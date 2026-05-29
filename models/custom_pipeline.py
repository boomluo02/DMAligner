from typing import Optional, Union

from matplotlib import pyplot as plt
import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    PNDMScheduler,
    UNet2DConditionModel,
)
from transformers import AutoTokenizer, CLIPTextModel

from tools.utils import img_postprocess

class CustomPipeline(DiffusionPipeline):
    '''
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        unet (`UNet2DConditionModel`):
            Conditional U-Net to denoise the depth latent, conditioned on image latent.
        vae (`AutoencoderKL`):
            Variational Auto-Encoder (VAE) Model to encode and decode images and depth maps
            to and from latent representations.
        scheduler (`DDIMScheduler`):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        text_encoder (`CLIPTextModel`):
            Text-encoder, for empty text embedding.
        tokenizer (`CLIPTokenizer`):
            CLIP tokenizer.
        default_denoising_steps (`int`, *optional*):
            The minimum number of denoising diffusion steps that are required to produce a prediction of reasonable
            quality with the given model. This value must be set in the model config. When the pipeline is called
            without explicitly setting `num_inference_steps`, the default value is used. This is required to ensure
            reasonable results with various model flavors compatible with the pipeline, such as those relying on very
            short denoising schedules (`LCMScheduler`) and those with full diffusion schedules (`DDIMScheduler`).
        default_processing_resolution (`int`, *optional*):
            The recommended value of the `processing_resolution` parameter of the pipeline. This value must be set in
            the model config. When the pipeline is called without explicitly setting `processing_resolution`, the
            default value is used. This is required to ensure reasonable results with various model flavors trained
            with varying optimal processing resolution values.
    '''

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: Union[DDIMScheduler, PNDMScheduler],
        text_encoder: CLIPTextModel,
        tokenizer: AutoTokenizer,
        default_num_inference_steps: Optional[int] = 200,
    ):
        super().__init__()
        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        self.register_to_config(
            default_num_inference_steps = default_num_inference_steps,
        )

        self.default_num_inference_steps = default_num_inference_steps

    @torch.no_grad()
    def __call__(
        self,
        input_image: torch.Tensor,
        num_inference_steps: Optional[int] = None,
        generator: Union[torch.Generator, None] = None,
    ):
        """
        Function invoked when calling the pipeline.

        Args:
            input_image (`torch.Tensor`):
                The input image to be denoised. These images have already been preprocessed.
            num_inference_steps (`int`, *optional*):
                The number of denoising diffusion steps to run. If not provided, the default value is used.
            batch_size (`int`, *optional*, defaults to 1):
                The batch size to use for inference.
            generator (`torch.Generator`, *optional*):
                A generator object to be used for inference.

        Returns:
            `torch.Tensor`: The denoised image.
        """
        if num_inference_steps is None:
            num_inference_steps = self.default_num_inference_steps

        # Encode the input image into latent representations
        image_latent = self.vae.encode(input_image.unsqueeze(0)).latent_dist.sample() * 0.18215

        # print(f'latent img shape: {pred_latent.shape}')
        # # visual
        # fig, axs = plt.subplots(1, 4)
        # for c in range(4):
        #     axs[c].imshow(pred_latent[0][c].cpu().numpy(),cmap='gray')
        # # save plt
        # plt.savefig('./test_img/pred_latent.jpg')

        # x_t
        pred_latents = torch.randn_like(image_latent, device=self.device)

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps  # [T]

        # Denoising loop
        for i, t in enumerate(timesteps.tolist()):
            unet_input = torch.cat([pred_latents, image_latent], dim=1)
            prompt = ""
            text_inputs = self.tokenizer(
                prompt,
                padding="do_not_pad",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
            empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype).to(self.device)

            noise_pred = self.unet(unet_input, t, empty_text_embed).sample

            pred_latents = self.scheduler.step(pred_latents, t, noise_pred).prev_sample

        # Decode the final predicted latent representation into an image 
        denoised_image = self.vae.decode(pred_latents / 0.18215).sample.squeeze(0)
        denoised_image = img_postprocess(denoised_image)

        # # save denoised_image
        # denoised_image_name = "./test_img/diffusion_denoised_image.jpg"
        # denoised_image_pil = denoised_image.cpu().numpy().transpose(1, 2, 0)
        # plt.imsave(denoised_image_name, denoised_image_pil)
        
        return denoised_image

    def encode_empty_text(self):
        """
        Encode text embedding for empty prompt
        """
        prompt = ""
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype)
