import inspect
import json
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from diffusers import DiffusionPipeline
from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from diffusers.schedulers import DDPMScheduler, DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from diffusers.utils import deprecate, logging
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from dataclasses import dataclass
from diffusers.utils.outputs import BaseOutput
from models.image_encoder import ImageProjModel
from models.refinenet import extract_decoder_features, extract_encoder_features

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def prepare_mask_and_masked_image(image, mask):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    return mask, masked_image


def overlay_inner_image(image:PIL.Image.Image, 
                        inner_image:PIL.Image.Image, 
                        paste_offset: Tuple[int] = (0, 0)):
    inner_image = inner_image.convert("RGBA")
    image = image.convert("RGB")

    image.paste(inner_image, paste_offset, inner_image)
    image = image.convert("RGB")

    return image

def pil_to_tensor(self, image: PIL.Image.Image) -> torch.Tensor:
        """
        Convert a PIL image to a tensor.

        Args:
            image (`PIL.Image.Image`):
                The input image to be converted.
        Returns:
            `torch.Tensor`: The converted tensor.
        """
        transform_to_tensor = transforms.ToTensor()
        image = transform_to_tensor(image)
        image = image.to(self.device)
        return image

class BodyCorrectionPipelineWithoutVAE(DiffusionPipeline):
    r"""
    Pipeline for body distortion correction using Diffusion without VAE.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        unet ([`UNet2DMonoModel`]): U-Net architecture to denoise the input image.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latens. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
    """
    def __init__(
        self,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDPMScheduler, DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        self.register_modules(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        input_image: Union[torch.Tensor, PIL.Image.Image],
        num_inference_steps: int = 50,
        eta: float = 0.0,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str`):
                The prompt text for the model. 'a photo of sks'
            input_image (`torch.Tensor` or `PIL.Image.Image`):
                `Image`, the condition input for the model.
            hole_mask (`torch.Tensor` or `PIL.Image.Image`):
                `Image`, the mask for the holes in the image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image`, `np.array` or 'torch.Tensor'.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """

        if isinstance(input_image, PIL.Image.Image):
            # to tensor
            input_image = pil_to_tensor(input_image) * 2 - 1
        *_, height, width = input_image.shape

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
        
        if prompt == "":
            # get empty prompt text embeddings
            text_embs = encode_empty_text(self.tokenizer, self.text_encoder)
        else:
            text_embs = encode_text(self.tokenizer, self.text_encoder, [prompt])

        # get the initial random noise
        pred_image = torch.randn_like(input_image, device=self.device)

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        # Some schedulers like PNDM have timesteps as arrays
        # It's more optimized to move all timesteps to correct device beforehand
        timesteps_tensor = self.scheduler.timesteps.to(self.device)

        # scale the initial noise by the standard deviation required by the scheduler
        pred_image = pred_image * self.scheduler.init_noise_sigma

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        for i, t in enumerate(self.progress_bar(timesteps_tensor)):
            # concat latents, mask, masked_image_latents in the channel dimension
            model_input = torch.cat([pred_image, input_image], dim=1)

            # predict the noise residual
            noise_pred = self.unet(model_input, t, encoder_hidden_states=text_embs).sample

            # compute the previous noisy sample x_t -> x_t-1
            pred_image = self.scheduler.step(noise_pred, t, pred_image, **extra_step_kwargs).prev_sample

        pred_image = pred_image * 0.5 + 0.5
        pred_image = pred_image.clamp(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        pred_image = pred_image.cpu().permute(0, 2, 3, 1).float().numpy()
        
        if output_type == "pil":
            pred_image = self.numpy_to_pil(pred_image)
            
        if not return_dict:
            return pred_image
        
        return AlignmentPipelineOutput(pred_image, None, None)

class BodyCorrectionInpaintingPipelineWithoutVAE(DiffusionPipeline):
    # TODO: finish this
    pass

class AlignmentPipeline(DiffusionPipeline):
    r"""
    Pipeline for body distortion correction using Stable Diffusion. 

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latens. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPImageProcessor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
        image_proj_model ([`ImageProjModel`], *optional*, defaults to `None`):
            Projection layer for image embeddings.
        adapter_modules ([`torch.nn.Module`], *optional*, defaults to `None`):
            Adapter modules to be used in the image cross-attention.
        
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDPMScheduler, DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        refine_net = None,
        image_proj_model: Optional[ImageProjModel] = None,
        adapter_modules: Optional[torch.nn.Module] = None,
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)
            
        scheduler.timestep_spacing="trailing"

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            refine_net=refine_net,
            image_proj_model=image_proj_model,
            adapter_modules=adapter_modules
        )

    @torch.no_grad()
    def __call__(
        self,
        img1: Union[torch.Tensor, PIL.Image.Image],
        img2: Union[torch.Tensor, PIL.Image.Image],
        num_inference_steps: int = 50,
        eta: float = 0.0,
        image_embeds: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        return_seq: bool = False,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str`):
                The prompt text for the model. 'a photo of sks'
            img1 (`torch.Tensor` or `PIL.Image.Image`):
                `Image`, the condition input of view1 image
            img2 (`torch.Tensor` or `PIL.Image.Image`):
                `Image`, the condition input of view2 image
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """

        if isinstance(img1, PIL.Image.Image):
            # to tensor
            img1 = pil_to_tensor(img1) * 2 - 1
        if isinstance(img2, PIL.Image.Image):
            # to tensor
            img2 = pil_to_tensor(img2) * 2 - 1
        *_, height, width = img1.shape

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if self.image_proj_model is not None:
            image_tokens = self.image_proj_model(image_embeds)
            text_embs = image_embeds.unsqueeze(1)
            encoder_hidden_states = torch.cat([text_embs, image_tokens], dim=1)
        else:
            text_embs = encode_empty_text(self.tokenizer, self.text_encoder)
            encoder_hidden_states = text_embs

        # encode the mask image into latents space so we can concatenate it to the latents
        image_latent1 = self.vae.encode(img1).latent_dist.sample(generator=generator) * self.vae.config.scaling_factor
        image_latent2 = self.vae.encode(img2).latent_dist.sample(generator=generator) * self.vae.config.scaling_factor
        image_latent = torch.cat([image_latent1, image_latent2], dim=1)

        # refine net
        if self.refine_net is not None:
            feature_maps = extract_encoder_features(self.vae.encoder, img1)

        # get the initial random noise
        pred_latent = torch.randn_like(image_latent1, device=self.device)

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        # Some schedulers like PNDM have timesteps as arrays
        # It's more optimized to move all timesteps to correct device beforehand
        timesteps_tensor = self.scheduler.timesteps.to(self.device)

        # scale the initial noise by the standard deviation required by the scheduler
        pred_latent = pred_latent * self.scheduler.init_noise_sigma

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        pred_seq = []
        for i, t in enumerate(self.progress_bar(timesteps_tensor)):
            # concat latents, mask, masked_image_latents in the channel dimension
            latent_model_input = torch.cat([pred_latent, image_latent], dim=1)

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=encoder_hidden_states).sample

            # compute the previous noisy sample x_t -> x_t-1
            pred_latent = self.scheduler.step(noise_pred, t, pred_latent, **extra_step_kwargs).prev_sample

            pred_seq.append((self.vae.decode(pred_latent / self.vae.config.scaling_factor).sample * 0.5 + 0.5).clamp(0, 1))

        pred_latent = 1 / self.vae.config.scaling_factor * pred_latent

        # Refine net
        if self.refine_net is not None:
            if self.vae.post_quant_conv is not None:
                pred_latent = self.vae.post_quant_conv(pred_latent)
            feature_maps.update(extract_decoder_features(self.vae.decoder, pred_latent))
            pred_image = self.refine_net(pred_latent, feature_maps)
        else:
            pred_image = self.vae.decode(pred_latent).sample
            
        pred_image = pred_image * 0.5 + 0.5
        pred_image = pred_image.clamp(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        pred_image = pred_image.cpu().permute(0, 2, 3, 1).float().numpy()

        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(pred_image), return_tensors="pt").to(
                self.device
            )
            pred_image, has_nsfw_concept = self.safety_checker(
                images=pred_image, clip_input=safety_checker_input.pixel_values.to(text_embs.dtype)
            )
        else:
            has_nsfw_concept = None

        if output_type == "pil":
            pred_image = self.numpy_to_pil(pred_image)

        if not return_dict:
            if return_seq:
                return (pred_image, has_nsfw_concept, pred_seq)
            else:
                return (pred_image, has_nsfw_concept)

        if return_seq:
            return AlignmentPipelineOutput(pred_image, None,has_nsfw_concept), pred_seq
        else:
            return AlignmentPipelineOutput(pred_image, None, has_nsfw_concept)

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        push_to_hub: bool = False,
        **kwargs,
    ):
        super().save_pretrained(save_directory, safe_serialization, variant, push_to_hub, **kwargs)
        
        model_index_path = os.path.join(save_directory, "model_index.json")
        if os.path.exists(model_index_path):
            with open(model_index_path, "r") as f:
                model_index = json.load(f)
            
            if "adapter_model" in model_index:
                del model_index["adapter_model"]
            
            if "image_proj_model" in model_index:
                del model_index["image_proj_model"]
            
            if "refine_net" in model_index:
                del model_index["refine_net"]

            with open(model_index_path, "w") as f:
                json.dump(model_index, f, indent=2)

class BodyCorrectionInpaintingPipeline(DiffusionPipeline):
    r"""
    Pipeline for body distortion correction (Inpainting) using Stable Diffusion. 

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latens. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPImageProcessor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDPMScheduler, DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        input_image: Union[torch.Tensor, PIL.Image.Image],
        hole_mask: Union[torch.Tensor, PIL.Image.Image] = None,
        num_inference_steps: int = 50,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        only_recon_hole: bool = True,
        use_rnd: bool = False,
        type: str = "ldm"
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str`):
                The prompt text for the model.
            input_image (`torch.Tensor` or `PIL.Image.Image`):
                `Image`, the condition input for the model.
            hole_mask (`torch.Tensor` or `PIL.Image.Image`):
                `Image`, the mask for the holes in the image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            only_recon_hole (`bool`, *optional*, defaults to `True`):
                Whether or not to only reconstruct the holes in the image. If `False`, the entire image will be
                reconstructed.
            use_rnd (`bool`, *optional*, defaults to `False`):
                Whether or not to use the RND method to fill the holes in the image.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """

        if isinstance(input_image, PIL.Image.Image):
            # to tensor
            input_image = pil_to_tensor(input_image) * 2 - 1
        *_, height, width = input_image.shape

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if prompt == "":
            # get empty prompt text embeddings
            text_embs = encode_empty_text(self.tokenizer, self.text_encoder)
        else:
            text_embs = encode_text(self.tokenizer, self.text_encoder, [prompt])

        # encode the mask image into latents space so we can concatenate it to the latents
        image_latent = self.vae.encode(input_image).latent_dist.sample(generator=generator)
        image_latent = image_latent * self.vae.config.scaling_factor

        if type == 'ldm' and hole_mask is not None:
            # TODO: Check if need 'nearst' interpolation
            # resize the mask to latents shape as we concatenate the mask to the latents
            hole_mask = F.interpolate(hole_mask, size=image_latent.shape[2:]).to(self.device)

        # get the initial random noise
        pred_latent = torch.randn_like(image_latent, device=self.device)

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        # Some schedulers like PNDM have timesteps as arrays
        # It's more optimized to move all timesteps to correct device beforehand
        timesteps_tensor = self.scheduler.timesteps.to(self.device)

        # scale the initial noise by the standard deviation required by the scheduler
        pred_latent = pred_latent * self.scheduler.init_noise_sigma

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        for i, t in enumerate(self.progress_bar(timesteps_tensor)):
            # concat latents, mask, masked_image_latents in the channel dimension
            latent_model_input = torch.cat([pred_latent, hole_mask, image_latent], dim=1)

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embs).sample

            # compute the previous noisy sample x_t -> x_t-1
            pred_latent = self.scheduler.step(noise_pred, t, pred_latent, **extra_step_kwargs).prev_sample

            if use_rnd:
                # RND
                pred_latent = image_latent * (1 - hole_mask) + pred_latent * hole_mask

        pred_latent = 1 / self.vae.config.scaling_factor * pred_latent
        pred_image = self.vae.decode(pred_latent).sample

        if only_recon_hole:
            hole_mask = hole_mask.to(pred_image.device)
            pred_image_only_holes = pred_image * hole_mask + input_image * (1 - hole_mask)
            
            pred_image = pred_image * 0.5 + 0.5
            pred_image = pred_image.clamp(0, 1)
            pred_image_only_holes = pred_image_only_holes * 0.5 + 0.5
            pred_image_only_holes = pred_image_only_holes.clamp(0, 1)
        else:
            pred_image = pred_image * 0.5 + 0.5
            pred_image = pred_image.clamp(0, 1)
            pred_image_only_holes = torch.zeros_like(pred_image)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        pred_image = pred_image.cpu().permute(0, 2, 3, 1).float().numpy()
        if only_recon_hole:
            pred_image_only_holes = pred_image_only_holes.cpu().permute(0, 2, 3, 1).float().numpy()

        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(pred_image), return_tensors="pt").to(
                self.device
            )
            pred_image, has_nsfw_concept = self.safety_checker(
                images=pred_image, clip_input=safety_checker_input.pixel_values.to(text_embs.dtype)
            )
        else:
            has_nsfw_concept = None

        if output_type == "pil":
            pred_image = self.numpy_to_pil(pred_image)
            pred_image_only_holes = self.numpy_to_pil(pred_image_only_holes)

        if not return_dict:
            return (pred_image, pred_image_only_holes, has_nsfw_concept)

        return AlignmentPipelineOutput(pred_image, pred_image_only_holes, has_nsfw_concept)


@dataclass
class AlignmentPipelineOutput(BaseOutput):
    """
    Output class for Body Diffusion pipelines.

    Args:
        pred_images (`List[PIL.Image.Image]`, `np.ndarray` or `torch.Tensor`)
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)` or PyTorch tensor of shape `(batch_size, num_channels, height, width)`.
        pred_image_only_holes (`List[PIL.Image.Image]`, `np.ndarray` or `torch.Tensor`, *optional*, defaults to `None`)
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)` or PyTorch tensor of shape `(batch_size, num_channels, height, width)`. Only available when `only_recon_hole` is `True` in the pipeline.
        nsfw_content_detected (`List[bool]`)
            List indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content or
            `None` if safety checking could not be performed.
    """

    pred_images: Union[List[PIL.Image.Image], np.ndarray, torch.Tensor]
    pred_image_only_holes: Optional[Union[List[PIL.Image.Image], np.ndarray, torch.Tensor]]
    nsfw_content_detected: Optional[List[bool]]

def encode_empty_text(tokenizer, text_encoder) -> torch.Tensor:
    """
    Encode text embedding for empty prompt
    """
    prompt = ""
    text_inputs = tokenizer(
        prompt,
        padding="do_not_pad",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(text_encoder.device)
    empty_text_embs = text_encoder(text_input_ids)[0]
    return empty_text_embs

def encode_text(tokenizer, text_encoder, batch_text: List[str]) -> torch.Tensor:
    """
    Encode text embedding for given text
    """
    text_embs = []
    for text in batch_text:
        text_inputs = tokenizer(
            text,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(text_encoder.device)
        text_embs.append(text_encoder(text_input_ids)[0])
    text_embs = torch.cat(text_embs, dim=0)
    return text_embs