'''
Initialize settings 
'''
import os
import warnings
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.utils.environment import check_cuda_p2p_ib_support
from diffusers import (
    AutoencoderKL,
    PNDMScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import create_repo
from packaging import version
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, CLIPTextModel

from data.dataset import get_dataloader
from tools.utils import replace_unet_conv_in, save_yaml
from tools.wandb import wandb_init


def initialize(cfg):
    # Will error if the minimal version of diffusers is not installed. Remove at your own risks.
    check_min_version("0.20.1")

    # if RTX 4000 series is used, disable P2P and IB
    if torch.cuda.is_available() and not check_cuda_p2p_ib_support():
        os.environ["NCCL_P2P_DISABLE"] = "1"
        os.environ["NCCL_IB_DISABLE"] = "1"

    # Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        mixed_precision=cfg.train.mixed_precision,
        log_with="wandb" if cfg.env.use_wandb else None,
        project_dir=cfg.env.log_dir,
    )

    # set seed
    set_seed(cfg.env.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if cfg.huggingface.push_to_hub:
            repo_id = create_repo(
                repo_id=cfg.huggingface.hub_model_id, 
                exist_ok=True, 
                token=cfg.huggingface.token, 
            ).repo_id
            cfg.env.repo_id = repo_id
        
    tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=cfg.model.revision,
            use_fast=False,
            clean_up_tokenization_spaces=False,
        )
    
    # Load scheduler and models
    noise_scheduler = PNDMScheduler.from_pretrained(cfg.model.pretrained_model_name_or_path, 
                                                    subfolder="scheduler"
                                                    )
    noise_scheduler.config.prediction_type = cfg.model.prediction_type
    text_encoder = CLIPTextModel.from_pretrained(cfg.model.pretrained_model_name_or_path, 
                                                 subfolder="text_encoder", 
                                                 revision=cfg.model.revision
                                                 )
    vae = AutoencoderKL.from_pretrained(cfg.model.pretrained_model_name_or_path, 
                                        subfolder="vae", 
                                        revision=cfg.model.revision
                                        )
    unet = UNet2DConditionModel.from_pretrained(cfg.model.pretrained_model_name_or_path, 
                                                subfolder="unet", 
                                                in_channels=8,
                                                low_cpu_mem_usage=False,
                                                ignore_mismatched_sizes=True,
                                                revision=cfg.model.revision
                                                )

    # if 8 != unet.config["in_channels"]:
    #     unet = replace_unet_conv_in(unet)

    if cfg.lora.use_lora:
        unet_config = LoraConfig(
            r=cfg.lora.lora_rank,
            lora_alpha=cfg.lora.lora_alpha,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            lora_dropout=cfg.lora.lora_dropout,
            bias=cfg.lora.lora_bias,
        )
        unet = get_peft_model(unet, unet_config)

    if cfg.train.enable_xformers:
        # xFormers memory-efficient attention
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version <= version.parse("0.0.16"):
                warnings.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
    
    if cfg.train.gradient_checkpointing:
        # Enable gradient checkpointing to save memory
        unet.enable_gradient_checkpointing()

    
    if cfg.train.allow_tf32:
        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        torch.backends.cuda.matmul.allow_tf32 = True

    if cfg.lr.scale_lr:
        cfg.lr.unet_learning_rate = (
            cfg.lr.unet_learning_rate * cfg.train.gradient_accumulation_steps * cfg.train.batch_size * accelerator.num_processes
        )

    # freeze the VAE and text encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Optimizer
    learning_rate = float(cfg.lr.unet_learning_rate)
    if cfg.train.use_8bit_adam:
        # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer = bnb.optim.AdamW8bit(unet.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-08)
    else:
        optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-08)
    
    # Dataloader
    train_loader, test_loader = get_dataloader(cfg)

    lr_scheduler = get_scheduler(
        cfg.lr.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr.lr_warmup_steps * cfg.train.gradient_accumulation_steps,
        num_training_steps=cfg.train.max_steps * cfg.train.gradient_accumulation_steps,
        num_cycles=cfg.lr.lr_num_cycles,
        power=cfg.lr.lr_power,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_loader, test_loader = accelerator.prepare(
        unet, optimizer, train_loader, test_loader
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    data_type = torch.float32
    if accelerator.mixed_precision == "fp16":
        data_type = torch.float16
    elif accelerator.mixed_precision == "bf16":
        data_type = torch.bfloat16

    # Move vae to device and cast to data_type
    vae.to(accelerator.device, dtype=data_type)
    text_encoder.to(accelerator.device, dtype=data_type)

    # resume load ckpt and optimizer
    resume_last_epoch = 0
    resume_max_psnr = -1
    if cfg.env.resume:
        ckpt = torch.load(cfg.model.resume_path, map_location='cpu')
        unet.load_state_dict(ckpt['unet_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        lr_scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])
        resume_last_epoch = ckpt['epoch']
        resume_max_psnr = ckpt['max_psnr']
        print(f"Resuming from epoch {resume_last_epoch} with min loss {resume_max_psnr}")

    save_yaml(cfg)
    if cfg.env.use_wandb and is_wandb_available():
        wandb_init(accelerator, cfg)

    # We return everything we need to train our model.
    return (
        accelerator,
        data_type,
        unet,
        vae,
        optimizer,
        text_encoder,
        tokenizer,
        lr_scheduler,
        train_loader,
        test_loader,
        noise_scheduler,
        resume_last_epoch,
        resume_max_psnr,
    )




    


