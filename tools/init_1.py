'''
Initialize settings 
'''
import logging
import math
import os
from pathlib import Path

import accelerate
import diffusers
import torch
import transformers
# import lpips
from accelerate import Accelerator, InitProcessGroupKwargs
from datetime import timedelta
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.utils.environment import check_cuda_p2p_ib_support
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler, # noqa: F401
    UNet2DConditionModel,
    UNet2DModel, # noqa: F401
)
from diffusers.optimization import get_scheduler
from diffusers.utils import (
    check_min_version,
)
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import create_repo
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from config.train_config import set_config
from data.dataset import get_dataloader
from models.bcd_pipeline import BodyCorrectionInpaintingPipeline, AlignmentPipeline
from models.image_encoder import ImageProjModel
from models.refinenet import RefineNet
from tools.utils_1 import load_attn_model, load_refine_net, save_yaml
from torch.nn import Conv2d
from torch.nn.parameter import Parameter

from peft import PeftModel, LoraConfig, get_peft_model

from tools.utils_1 import is_torch2_available

if is_torch2_available():
    from models.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from models.attention_processor import IPAttnProcessor, AttnProcessor

def initialize_vae(stage='train'):
    if stage == 'train':
        # Will error if the minimal version of diffusers is not installed. Remove at your own risks.
        check_min_version("0.30.0")

        # if RTX 4000 series is used, disable P2P and IB
        if torch.cuda.is_available() and not check_cuda_p2p_ib_support():
            os.environ["NCCL_P2P_DISABLE"] = "1"
            os.environ["NCCL_IB_DISABLE"] = "1"

        logger = get_logger(__name__, log_level="INFO")

        cfg, hub_token = set_config()

        if cfg.tracker.report_to == "wandb" and hub_token is not None:
            raise ValueError(
                "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
                " Please use `huggingface-cli login` to authenticate with the Hub."
            )

        logging_dir = os.path.join(cfg.env.output_dir, cfg.tracker.logging_dir)

        accelerator_project_config = ProjectConfiguration(project_dir=cfg.env.output_dir, logging_dir=logging_dir)

        accelerator = Accelerator(
            gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
            mixed_precision=cfg.train.mixed_precision,
            log_with=cfg.tracker.report_to,
            project_config=accelerator_project_config,
        )

        # Disable AMP for MPS.
        if torch.backends.mps.is_available():
            accelerator.native_amp = False

        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state, main_process_only=False)

        if accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        # If passed along, set the training seed now.
        if cfg.env.seed is not None:
            set_seed(cfg.env.seed)
        
        # Handle the repository creation
        repo_id = None
        if accelerator.is_main_process:
            if cfg.env.output_dir is not None:
                os.makedirs(cfg.env.output_dir, exist_ok=True)

            if cfg.huggingface.push_to_hub:
                repo_id = create_repo(
                    repo_id=cfg.huggingface.hub_model_id or Path(cfg.env.output_dir).name, 
                    exist_ok=True, 
                    token=hub_token
                ).repo_id
        
        # train vae only
        if cfg.model.use_4x_downsample_vae:
            vae = AutoencoderKL.from_pretrained(
                pretrained_model_name_or_path='models/vae_kl_f4', 
                subfolder="vae", 
            )
            vae_bin_file = os.path.join('models/vae_kl_f4', 'diffusion_pytorch_model.bin')
            if not os.path.exists(vae_bin_file):
                raise ValueError(f"File {vae_bin_file} does not exist. Please download the model from Hugging Face Hub.")
            ckpt_sd = torch.load(vae_bin_file)
            new_sd = {}
            for k, v in ckpt_sd.items():
                if 'mid_block.attentions.0' in k:
                    new_k = k.replace('query', 'to_q').replace('key', 'to_k').replace('value', 'to_v').replace('proj_attn', 'to_out.0')
                    new_sd[new_k] = v
                else:
                    new_sd[k] = v
            
            vae.load_state_dict(new_sd)
            logger.info(f"Loaded 4x downsampled VAE from {vae_bin_file}")
        else:
            vae = AutoencoderKL.from_pretrained(
                cfg.model.pretrained_model_name_or_path, 
                subfolder="vae", 
                revision=cfg.model.revision, 
                variant=cfg.model.variant,
            )

        # `accelerate` 0.16.0 will have better support for customized saving
        if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
            # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
            def save_model_hook(models, weights, output_dir):
                if accelerator.is_main_process:
                    for i, model in enumerate(models):
                        sub_dir = "vae" 
                        model.save_pretrained(os.path.join(output_dir, sub_dir))
                        # make sure to pop weight so that corresponding model is not saved again
                        weights.pop()

            def load_model_hook(models, input_dir):
                for _ in range(len(models)):
                    # pop models so that they are not loaded again
                    model = models.pop()
                    model_cls = AutoencoderKL
                    load_model = model_cls.from_pretrained(input_dir, subfolder="vae")
                    model.register_to_config(**load_model.config)

                    model.load_state_dict(load_model.state_dict())
                    del load_model

            accelerator.register_save_state_pre_hook(save_model_hook)
            accelerator.register_load_state_pre_hook(load_model_hook)
        
        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if cfg.train.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        if cfg.lr.scale_lr:
            cfg.lr.learning_rate = (
                cfg.lr.learning_rate * cfg.train.gradient_accumulation_steps * cfg.train.batch_size * accelerator.num_processes
            )
        
        # Initialize the optimizer
        if cfg.train.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
                )

            optimizer_cls = bnb.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW

        optimizer = optimizer_cls(
            vae.parameters(),
            lr=cfg.lr.learning_rate,
            betas=(cfg.optim.adam_beta1, cfg.optim.adam_beta2),
            weight_decay=cfg.optim.adam_weight_decay,
            eps=cfg.optim.adam_epsilon,
        )

        # Prepare DataLoader
        train_dataloader, test_dataloader = get_dataloader(cfg)

        # Scheduler and math around the number of training steps.
        # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
        num_warmup_steps_for_scheduler = cfg.lr.lr_warmup_steps * accelerator.num_processes
        if cfg.train.max_train_steps is None:
            len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
            num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / cfg.train.gradient_accumulation_steps)
            num_training_steps_for_scheduler = (
                cfg.train.epochs * num_update_steps_per_epoch * accelerator.num_processes
            )
        else:
            num_training_steps_for_scheduler = cfg.train.max_train_steps * accelerator.num_processes

        lr_scheduler = get_scheduler(
            cfg.lr.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps_for_scheduler,
            num_training_steps=num_training_steps_for_scheduler,
        )

        # Prepare everything with our `accelerator`.
        vae, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            vae, optimizer, train_dataloader, lr_scheduler
        )

        # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
            cfg.train.mixed_precision = accelerator.mixed_precision
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
            cfg.train.mixed_precision = accelerator.mixed_precision
        
        # Move vae to gpu
        vae.to(accelerator.device, dtype=weight_dtype)

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.train.gradient_accumulation_steps)
        if cfg.train.max_train_steps is None:
            cfg.train.max_train_steps = cfg.train.epochs * num_update_steps_per_epoch
            if num_training_steps_for_scheduler != cfg.train.max_train_steps * accelerator.num_processes:
                logger.warning(
                    f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                    f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                    f"This inconsistency may result in the learning rate scheduler not functioning properly."
                )
        # Afterwards we recalculate our number of training epochs
        cfg.train.epochs = math.ceil(cfg.train.max_train_steps / num_update_steps_per_epoch)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if accelerator.is_main_process:
            tracker_config = dict(vars(cfg))
            init_kwargs = {"wandb": {"name": f"{cfg.env.signature}"}}
            accelerator.init_trackers(cfg.tracker.project, tracker_config, init_kwargs=init_kwargs)

        save_yaml(cfg)

        lpips_loss_fn = None
        # lpips_loss_fn = lpips.LPIPS(net='alex').to(accelerator.device)
        # lpips_loss_fn.eval()

        return (cfg, logger, accelerator,
                train_dataloader, test_dataloader,
                vae, optimizer, lr_scheduler, lpips_loss_fn,
                weight_dtype, num_update_steps_per_epoch, repo_id)
    
    elif stage == 'test':
        # Will error if the minimal version of diffusers is not installed. Remove at your own risks.
        check_min_version("0.30.0")

        # if RTX 4000 series is used, disable P2P and IB
        if torch.cuda.is_available() and not check_cuda_p2p_ib_support():
            os.environ["NCCL_P2P_DISABLE"] = "1"
            os.environ["NCCL_IB_DISABLE"] = "1"

        cfg, hub_token = set_config()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        vae = AutoencoderKL.from_pretrained(
            cfg.model.pretrained_model_name_or_path, 
            subfolder="vae", 
            revision=None, 
            variant=None,
        )

        if cfg.model.pretrained_model_name_or_path == 'models/vae_kl_f4':
            vae_bin_file = os.path.join('models/vae_kl_f4', 'diffusion_pytorch_model.bin')
            if not os.path.exists(vae_bin_file):
                raise ValueError(f"File {vae_bin_file} does not exist. Please download the model from Hugging Face Hub.")
            ckpt_sd = torch.load(vae_bin_file)
            new_sd = {}
            for k, v in ckpt_sd.items():
                if 'mid_block.attentions.0' in k:
                    new_k = k.replace('query', 'to_q').replace('key', 'to_k').replace('value', 'to_v').replace('proj_attn', 'to_out.0')
                    new_sd[new_k] = v
                else:
                    new_sd[k] = v
            
            vae.load_state_dict(new_sd)
        
        vae = vae.to(device)
        
        vae.eval()

        # Prepare DataLoader
        train_dataloader, test_dataloader = get_dataloader(cfg)

        save_yaml(cfg)
        
        return cfg, vae, test_dataloader, device


def initialize(model_type='ldm'):
    # Will error if the minimal version of diffusers is not installed. Remove at your own risks.
    check_min_version("0.30.0")

    # if RTX 4000 series is used, disable P2P and IB
    if torch.cuda.is_available() and not check_cuda_p2p_ib_support():
        os.environ["NCCL_P2P_DISABLE"] = "1"
        os.environ["NCCL_IB_DISABLE"] = "1"

    logger = get_logger(__name__, log_level="INFO")

    cfg, hub_token = set_config()

    if cfg.tracker.report_to == "wandb" and hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = os.path.join(cfg.env.output_dir, cfg.tracker.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=cfg.env.output_dir, logging_dir=logging_dir)

    # NCCL timeout is set to 3 hours for long time validation
    init_process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=10800))  # 3 hrs

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        mixed_precision=cfg.train.mixed_precision,
        log_with=cfg.tracker.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[init_process_group_kwargs]
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if cfg.env.seed is not None:
        set_seed(cfg.env.seed)
    
    # Handle the repository creation
    repo_id = None
    if accelerator.is_main_process:  
        if cfg.huggingface.push_to_hub:
            repo_id = create_repo(
                repo_id=cfg.huggingface.hub_model_id or Path(cfg.env.output_dir).name, 
                exist_ok=True, 
                token=hub_token
            ).repo_id
    
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDIMScheduler.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="scheduler")
    noise_scheduler.beta_schedule = cfg.train.beta_schedule
    noise_scheduler.timestep_spacing="trailing"
    # noise_scheduler = DDPMScheduler.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        cfg.model.pretrained_model_name_or_path, 
        subfolder="tokenizer", 
        revision=cfg.model.revision,
        clean_up_tokenization_spaces=True, 
        use_fast=False,
    )
    safety_checker = None

    text_encoder = CLIPTextModel.from_pretrained(
        cfg.model.pretrained_model_name_or_path, 
        subfolder="text_encoder", 
        revision=cfg.model.revision, 
        variant=cfg.model.variant
    )
        
    # VAE
    vae = AutoencoderKL.from_pretrained(
        cfg.model.pretrained_model_name_or_path, 
        subfolder="vae", 
        revision=cfg.model.revision, 
        variant=cfg.model.variant,
    )
    unet_inchannels = vae.config.latent_channels * 3 
    if cfg.env.task == 'bcd-inpainting':
        unet_inchannels += 1 # add one channel for mask 
    unet_outchannels = vae.config.latent_channels
    # Unet
    unet = UNet2DConditionModel.from_pretrained(
        cfg.model.pretrained_model_name_or_path, 
        subfolder="unet",
        out_channels=unet_outchannels,
        revision=cfg.model.revision,
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True,
    )

    assert unet_inchannels == 12, "Only support 12 channels for 8xVAE"
    unet = _replace_unet_conv_in(unet, 12)
    
    # Freeze vae
    if cfg.train.enable_vae_decoder:
        vae.encoder.requires_grad_(False)
    else:
        vae.requires_grad_(False)

    if cfg.train.freeze_mid_unet:
        # only train the first block and the last block of the unet
        for name, param in unet.named_parameters(): 
            exclude_keywords = ["conv_in", "time_embedding", "down_blocks.0", "up_blocks.3", "conv_norm_out", "conv_out"]
            if all(keyword not in name for keyword in exclude_keywords):
                param.requires_grad = False

    if cfg.train.enable_lora:
        unet_config = LoraConfig(
            r=cfg.lora.lora_rank,
            lora_alpha=cfg.lora.lora_alpha,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            lora_dropout=cfg.lora.lora_dropout,
            bias='none' if cfg.lora.lora_bias is None else cfg.lora.lora_bias,
        )
        unet = get_peft_model(unet, unet_config)

    # Freeze text_encoder
    text_encoder.requires_grad_(False)
     
    if cfg.env.decoupled_attn:
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(cfg.model.clip_name_or_path)
        image_encoder.requires_grad_(False)

        #ip-adapter
        image_proj_model = ImageProjModel(
            cross_attention_dim=unet.config.cross_attention_dim,
            clip_embeddings_dim=image_encoder.config.projection_dim,
            clip_extra_context_tokens=4,
        )

        # init adapter modules
        attn_procs = {}
        unet_sd = unet.state_dict()
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                layer_name = name.split(".processor")[0]
                weights = {
                    "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                    "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                }
                attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
                attn_procs[name].load_state_dict(weights)
        unet.set_attn_processor(attn_procs)
        adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())

    if cfg.train.enable_refine_net:
        refine_net_config = {}
        refine_net_config['latent_channels'] = vae.config['latent_channels']
        refine_net_config['out_channels'] = vae.config['out_channels']
        refine_net_config['up_block_types'] = vae.config['up_block_types']
        refine_net_config['block_out_channels'] = vae.config['block_out_channels']
        refine_net_config['layers_per_block'] = vae.config['layers_per_block']
        refine_net_config['norm_num_groups'] = vae.config['norm_num_groups']
        refine_net_config['act_fn'] = vae.config['act_fn']
        refine_net_config['mid_block_add_attention'] = vae.config['mid_block_add_attention']
        refine_net = RefineNet(
            in_channels=refine_net_config['latent_channels'],
            out_channels=refine_net_config['out_channels'],
            up_block_types=refine_net_config['up_block_types'],
            block_out_channels=refine_net_config['block_out_channels'],
            layers_per_block=refine_net_config['layers_per_block'],
            norm_num_groups=refine_net_config['norm_num_groups'],
            act_fn=refine_net_config['act_fn'],
            mid_block_add_attention=refine_net_config['mid_block_add_attention'],
        )

    if cfg.train.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                for i, model in enumerate(models):
                    if cfg.env.task == 'bcd-inpainting':
                        if hasattr(model, 'base_model') and model.base_model is not None: # LoRA unet & text_encoder
                            sub_dir = "text_encoder" if isinstance(model.base_model.model, type(accelerator.unwrap_model(text_encoder).base_model.model)) else "unet"
                        else: # non-LoRA unet
                            if isinstance(model, type(accelerator.unwrap_model(unet))):
                                sub_dir = "unet"
                            else:
                                raise ValueError("Text encoder is not supported without LoRA")
                    else:
                        if cfg.env.decoupled_attn:
                            if isinstance(model, type(accelerator.unwrap_model(unet))):
                                sub_dir = "unet"
                                model.save_pretrained(os.path.join(output_dir, sub_dir))
                            elif isinstance(model, type(accelerator.unwrap_model(image_proj_model))):
                                sub_dir = "image_proj_model"
                                # Save the custom model's state dict manually
                                save_dir = os.path.join(output_dir, sub_dir)
                                if not os.path.exists(save_dir):
                                    os.makedirs(save_dir)
                                torch.save(model.state_dict(), os.path.join(save_dir, 'pytorch_model.bin'))
                            elif isinstance(model, type(accelerator.unwrap_model(adapter_modules))):
                                sub_dir = "adapter_modules" 
                                # Save the custom model's state dict manually
                                save_dir = os.path.join(output_dir, sub_dir)
                                if not os.path.exists(save_dir):
                                    os.makedirs(save_dir)
                                torch.save(model.state_dict(), os.path.join(save_dir, 'pytorch_model.bin'))
                            elif isinstance(model, type(accelerator.unwrap_model(refine_net))):
                                sub_dir = "refine_net"
                                # save the custom model's state dict manually
                                save_dir = os.path.join(output_dir, sub_dir)
                                if not os.path.exists(save_dir):
                                    os.makedirs(save_dir)
                                torch.save(model.state_dict(), os.path.join(save_dir, 'pytorch_model.bin'))
                        else:
                            if isinstance(model, type(accelerator.unwrap_model(unet))):
                                sub_dir = "unet"
                                model.save_pretrained(os.path.join(output_dir, sub_dir))
                            elif isinstance(model, type(accelerator.unwrap_model(refine_net))):
                                sub_dir = "refine_net"
                                # save the custom model's state dict manually
                                save_dir = os.path.join(output_dir, sub_dir)
                                if not os.path.exists(save_dir):
                                    os.makedirs(save_dir)
                                torch.save(model.state_dict(), os.path.join(save_dir, 'pytorch_model.bin'))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            for _ in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()
                if cfg.env.task == 'bcd-inpainting':
                    if hasattr(model, 'base_model') and model.base_model is not None: # LoRA unet & text_encoder
                        sub_dir = "text_encoder" if isinstance(model.base_model.model, type(accelerator.unwrap_model(text_encoder).base_model.model)) else "unet"
                        model_cls = UNet2DConditionModel if isinstance(model.base_model.model, model_type(accelerator.unwrap_model(unet).base_model.model)) else CLIPTextModel
                        load_model = model_cls.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder=sub_dir)
                        load_model = PeftModel.from_pretrained(load_model, input_dir, subfolder=sub_dir)
                    else: # non-LoRA unet
                        if isinstance(model, type(accelerator.unwrap_model(unet))):
                            sub_dir = "unet"
                            model_cls = UNet2DConditionModel
                            load_model = model_cls.from_pretrained(input_dir, subfolder=sub_dir)
                else:
                    if isinstance(model, type(accelerator.unwrap_model(unet))):
                        sub_dir = "unet"
                        model_cls = UNet2DConditionModel
                        load_model = model_cls.from_pretrained(input_dir, subfolder=sub_dir)
                        model.register_to_config(**load_model.config)
                    elif isinstance(model, type(accelerator.unwrap_model(image_proj_model))):
                        # Unsupport from pretrained method
                        sub_dir = "image_proj_model"
                        load_model = image_proj_model.load_state_dict(torch.load(os.path.join(input_dir, sub_dir, "pytorch_model.bin")))
                    elif isinstance(model, type(accelerator.unwrap_model(adapter_modules))):
                        # Unsupport from pretrained method
                        sub_dir = "adapter_modules"
                        load_model = adapter_modules.load_state_dict(torch.load(os.path.join(input_dir, sub_dir, "pytorch_model.bin")))
                    elif isinstance(model, type(accelerator.unwrap_model(refine_net))):
                        # Unsupport from pretrained method
                        sub_dir = "refine_net"
                        load_model = refine_net.load_state_dict(torch.load(os.path.join(input_dir, sub_dir, "pytorch_model.bin")))
                
                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)
    
    if cfg.train.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if cfg.env.task == 'bcd-inpainting':
            text_encoder.enable_gradient_checkpointing()
    
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if cfg.train.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if cfg.lr.scale_lr:
        cfg.lr.unet_learning_rate = (
            cfg.lr.unet_learning_rate * cfg.train.gradient_accumulation_steps * cfg.train.batch_size * accelerator.num_processes
        )

        cfg.lr.text_encoder_learning_rate = (
            cfg.lr.text_encoder_learning_rate * cfg.lr.gradient_accumulation_steps * cfg.lr.train_batch_size * accelerator.num_processes
        )

        if cfg.env.decoupled_attn:
            cfg.lr.attn_learning_rate = (
                cfg.lr.attn_learning_rate * cfg.lr.gradient_accumulation_steps * cfg.lr.train_batch_size * accelerator.num_processes
            )
        
        if cfg.train.enable_refine_net:
            cfg.lr.refine_net_learning_rate = (
                cfg.lr.refine_net_learning_rate * cfg.train.gradient_accumulation_steps * cfg.train.batch_size * accelerator.num_processes
            )
    
    # Initialize the optimizer
    if cfg.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    params = [{"params": unet.parameters(), "lr": cfg.lr.unet_learning_rate}]
    if cfg.env.task == 'bcd-inpainting':
        params.append({"params": text_encoder.parameters(), "lr": cfg.lr.text_encoder_learning_rate})
    if cfg.env.decoupled_attn:
        params.append({"params": image_proj_model.parameters(), "lr": cfg.lr.attn_learning_rate})
        params.append({"params": adapter_modules.parameters(), "lr": cfg.lr.attn_learning_rate})
    if cfg.train.enable_refine_net:
        params.append({"params": refine_net.parameters(), "lr": cfg.lr.refine_net_learning_rate})

    optimizer = optimizer_cls(
        params,
        betas=(cfg.optim.adam_beta1, cfg.optim.adam_beta2),
        weight_decay=cfg.optim.adam_weight_decay,
        eps=cfg.optim.adam_epsilon,
    )

    # load additional module state dict
    if cfg.model.load_addition_from_pretrained:
        # base model like SD2-base don't have additional modules to load
        checkpoint_dir = cfg.model.pretrained_model_name_or_path

        # Image attn
        if cfg.env.decoupled_attn and accelerator.is_main_process:
            load_attn_model(checkpoint_dir, image_proj_model, adapter_modules)
        
        # Refine net
        if cfg.train.enable_refine_net and accelerator.is_main_process:
            load_refine_net(checkpoint_dir, refine_net)

    else:
        print("No additional modules loaded")

    # Prepare DataLoader
    train_dataloader, test_dataloader = get_dataloader(cfg)

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = cfg.lr.lr_warmup_steps * accelerator.num_processes
    if cfg.train.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / cfg.train.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            cfg.train.epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = cfg.train.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        cfg.lr.lr_scheduler,
        optimizer=optimizer,
        step_rules=cfg.lr.step_rules,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    # Prepare everything with our `accelerator`.
    if cfg.env.task == 'bcd-inpainting':
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    elif cfg.env.task == 'bcd':
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )
        if cfg.env.decoupled_attn:
            image_proj_model, adapter_modules = accelerator.prepare(image_proj_model, adapter_modules)
        if cfg.train.enable_refine_net:
            refine_net = accelerator.prepare(refine_net)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        cfg.train.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        cfg.train.mixed_precision = accelerator.mixed_precision

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    if model_type=='ldm':
        vae.to(accelerator.device, dtype=weight_dtype)
    if cfg.env.decoupled_attn:
        image_encoder.to(accelerator.device, dtype=weight_dtype)
    if cfg.train.enable_refine_net:
        refine_net.to(accelerator.device, dtype=weight_dtype)
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.train.gradient_accumulation_steps)
    if cfg.train.max_train_steps is None:
        cfg.train.max_train_steps = cfg.train.epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != cfg.train.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    cfg.train.epochs = math.ceil(cfg.train.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(cfg))
        init_kwargs = {"wandb": {"name": f"{cfg.env.signature}"}}
        accelerator.init_trackers(cfg.tracker.project, tracker_config, init_kwargs=init_kwargs)

    save_yaml(cfg)


    init_dict = {
        "cfg": cfg,
        "logger": logger,
        "accelerator": accelerator,
        "safety_checker": safety_checker,
        "train_dataloader": train_dataloader,
        "test_dataloader": test_dataloader,
        "unet": unet,
        "vae": vae,
        "text_encoder": text_encoder,
        "noise_scheduler": noise_scheduler,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
        "tokenizer": tokenizer,
        "weight_dtype": weight_dtype,
        "num_update_steps_per_epoch": num_update_steps_per_epoch,
        "repo_id": repo_id,
    }

    # add modules according to config
    if cfg.env.decoupled_attn:
        init_dict["image_encoder"] = image_encoder
        init_dict["image_proj_model"] = image_proj_model
        init_dict["adapter_modules"] = adapter_modules
    
    if cfg.train.enable_refine_net:
        init_dict["refine_net"] = refine_net

    return init_dict

def initialize_test():
    # Will error if the minimal version of diffusers is not installed. Remove at your own risks.
    check_min_version("0.30.0")

    # if RTX 4000 series is used, disable P2P and IB
    if torch.cuda.is_available() and not check_cuda_p2p_ib_support():
        os.environ["NCCL_P2P_DISABLE"] = "1"
        os.environ["NCCL_IB_DISABLE"] = "1"

    logger = get_logger(__name__, log_level="INFO")

    cfg, hub_token = set_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.env.task == 'bcd-inpainting':
        use_rnd = cfg.model.use_rnd
    else:
        use_rnd = False

    assert cfg.env.mode in ['test', 'inference'], "This initialization is only for testing"

    if cfg.tracker.report_to == "wandb" and hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # pipeline
    if cfg.env.task == 'bcd':
        pipeline = AlignmentPipeline.from_pretrained(
            cfg.model.pretrained_model_name_or_path,
            refine_net=None,
            safety_checker=None,
            torch_dtype=torch.float32,
            revision=None,
        )

        if cfg.env.decoupled_attn:
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(cfg.model.clip_name_or_path).to(device)
            image_encoder.requires_grad_(False)

            #ip-adapter
            image_proj_model = ImageProjModel(
                cross_attention_dim=pipeline.unet.config.cross_attention_dim,
                clip_embeddings_dim=image_encoder.config.projection_dim,
                clip_extra_context_tokens=4,
            ).to(device)

            # init adapter modules
            attn_procs = {}
            unet_sd = pipeline.unet.state_dict()
            for name in pipeline.unet.attn_processors.keys():
                cross_attention_dim = None if name.endswith("attn1.processor") else pipeline.unet.config.cross_attention_dim
                if name.startswith("mid_block"):
                    hidden_size = pipeline.unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(pipeline.unet.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = pipeline.unet.config.block_out_channels[block_id]
                if cross_attention_dim is None:
                    attn_procs[name] = AttnProcessor()
                else:
                    layer_name = name.split(".processor")[0]
                    weights = {
                        "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                        "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                    }
                    attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
                    attn_procs[name].load_state_dict(weights)
            pipeline.unet.set_attn_processor(attn_procs)
            adapter_modules = torch.nn.ModuleList(pipeline.unet.attn_processors.values()).to(device)

            load_attn_model(cfg.model.pretrained_model_name_or_path, image_proj_model, adapter_modules)
        else:
            image_encoder = None
            image_proj_model = None
            adapter_modules = None

        pipeline.image_proj_model = image_proj_model
        pipeline.adapter_modules = adapter_modules
        
    elif cfg.env.task == 'bcd-inpainting':
        pipeline = BodyCorrectionInpaintingPipeline.from_pretrained(
            cfg.model.pretrained_model_name_or_path,
            safety_checker=None,
            torch_dtype=torch.float32,
            revision=None,
        )
    else:
        raise ValueError(f"Task {cfg.env.task} is not supported")
    
    if cfg.model.enable_refine_net:
        refine_net_config = {}
        refine_net_config['latent_channels'] = pipeline.vae.config['latent_channels']
        refine_net_config['out_channels'] = pipeline.vae.config['out_channels']
        refine_net_config['up_block_types'] = pipeline.vae.config['up_block_types']
        refine_net_config['block_out_channels'] = pipeline.vae.config['block_out_channels']
        refine_net_config['layers_per_block'] = pipeline.vae.config['layers_per_block']
        refine_net_config['norm_num_groups'] = pipeline.vae.config['norm_num_groups']
        refine_net_config['act_fn'] = pipeline.vae.config['act_fn']
        refine_net_config['mid_block_add_attention'] = pipeline.vae.config['mid_block_add_attention']
        refine_net = RefineNet(
            in_channels=refine_net_config['latent_channels'],
            out_channels=refine_net_config['out_channels'],
            up_block_types=refine_net_config['up_block_types'],
            block_out_channels=refine_net_config['block_out_channels'],
            layers_per_block=refine_net_config['layers_per_block'],
            norm_num_groups=refine_net_config['norm_num_groups'],
            act_fn=refine_net_config['act_fn'],
            mid_block_add_attention=refine_net_config['mid_block_add_attention'],
        )
        load_refine_net(cfg.model.pretrained_model_name_or_path, refine_net)
        refine_net.to(device)
        pipeline.refine_net = refine_net

    if cfg.model.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            pipeline.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
    
    if cfg.model.enable_model_cpu_offload:
        pipeline.enable_model_cpu_offload()
    else:
        pipeline.to(device)

    # Prepare DataLoader
    train_dataloader, test_dataloader = get_dataloader(cfg)

    save_yaml(cfg)

    return cfg, logger, pipeline, image_encoder, train_dataloader, test_dataloader, use_rnd, device

def deepspeed_zero_init_disabled_context_manager():
    """
    returns either a context list that includes one that will disable zero.Init or an empty context list
    """
    deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
    if deepspeed_plugin is None:
        return []
    
    return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

        
def _replace_unet_conv_in(unet, target_channels):
    # replace the first layer to accept target in_channels
    _weight = unet.conv_in.weight.clone()  # [320, 4, 3, 3]
    _bias = unet.conv_in.bias.clone()  # [320]

    if target_channels == 8:
        _weight = _weight.repeat((1, 2, 1, 1))  # Keep selected channel(s)
    elif target_channels == 12:
        _weight = _weight.repeat((1, 3, 1, 1))
    elif target_channels == 6:
        # drop last channel
        _weight = _weight[:, :-1, :, :].repeat((1, 2, 1, 1))

    # half the activation magnitude
    _weight *= 0.5
    # new conv_in channel
    _n_convin_out_channel = unet.conv_in.out_channels
    _new_conv_in = Conv2d(
        target_channels, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
    )
    _new_conv_in.weight = Parameter(_weight)
    _new_conv_in.bias = Parameter(_bias)
    unet.conv_in = _new_conv_in
    logging.info("Unet conv_in layer is replaced")
    # replace config
    unet.config["in_channels"] = target_channels
    logging.info("Unet config is updated")
    return unet