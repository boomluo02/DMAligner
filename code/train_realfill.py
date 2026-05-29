import itertools
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import yaml
import shutil
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from diffusers.training_utils import (
    compute_snr,
)
from huggingface_hub import upload_folder
from tqdm.auto import tqdm

from tools.init_1 import initialize
from tools.utils_1 import log_validation_ldm, read_data, save_model_card, test_vae, unwrap_model  # noqa: F401
from tools.loss import ReconLoss, calc_recon_loss, calc_texture_loss, get_pred_img

from models.bcd_pipeline import BodyCorrectionInpaintingPipeline, encode_empty_text, encode_text
import cv2
import numpy as np

def main():
    # =========== initialize ===========
    cfg, logger, accelerator, safety_checker, \
    train_dataloader, test_dataloader,\
    unet, vae, text_encoder, \
    noise_scheduler, optimizer, lr_scheduler, tokenizer,\
    weight_dtype, num_update_steps_per_epoch, repo_id = initialize(model_type="ldm")

    # ========== Train ===========
    total_batch_size = cfg.train.batch_size * accelerator.num_processes * cfg.train.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples(Per GPU) = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {cfg.train.epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.train.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg.train.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps(Per GPU) = {cfg.train.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if cfg.model.resume_from_checkpoint:
        path = cfg.model.resume_from_checkpoint
        if not os.path.exists(path):
            raise ValueError(f"Can't find [{path}]")
        accelerator.print(f"Resuming from checkpoint [{path}]")
        accelerator.load_state(cfg.model.resume_from_checkpoint)

    progress_bar = tqdm(
        range(0, cfg.train.max_train_steps),
        initial=0,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, cfg.train.epochs):
        unet.train()
        text_encoder.train()

        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet, text_encoder):
                # read data
                input_imgs, shape_imgs, \
                mask_imgs, hole_masks, \
                hole_imgs, prompts, target_shapes, img_ids = read_data(batch, accelerator.device, weight_dtype, phase='train')

                # save hole masks & hole imgs for visualization
                # v_hole_mask = hole_masks[0].detach().cpu().numpy().transpose(1, 2, 0) * 255
                # v_hole_img = hole_imgs[0] * 0.5 + 0.5
                # v_hole_img = v_hole_img.detach().cpu().numpy().transpose(1, 2, 0) * 255
                # # BGR to RGB
                # v_hole_img = v_hole_img[..., ::-1]
                # v_hole_img = v_hole_img.astype(np.uint8)
                # cv2.imwrite(f"test_img/hole_mask_{img_ids[0]}.png", v_hole_mask)
                # cv2.imwrite(f"test_img/hole_img_{img_ids[0]}.png", v_hole_img)

                # Convert images to latent space
                latent_inputs = vae.encode(input_imgs).latent_dist.sample()
                latent_inputs = latent_inputs * vae.config.scaling_factor
                # test_vae(vae, latent_inputs)

                # conditions 
                latent_img_with_holes = vae.encode(hole_imgs).latent_dist.sample()
                latent_img_with_holes = latent_img_with_holes * vae.config.scaling_factor

                # Downsample mask and weighting so that they match with the latents
                hole_masks = F.interpolate(hole_masks, size=latent_inputs.shape[2:])

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latent_inputs)

                if cfg.train.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += cfg.train.noise_offset * torch.randn(
                        (latent_inputs.shape[0], latent_inputs.shape[1], 1, 1), device=latent_inputs.device
                    )
                if cfg.train.input_perturbation:
                    new_noise = noise + cfg.train.input_perturbation * torch.randn_like(noise)
                bsz = latent_inputs.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, 
                                          noise_scheduler.config.num_train_timesteps, 
                                          (bsz,), 
                                          device=latent_inputs.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                if cfg.train.input_perturbation:
                    noisy_latents = noise_scheduler.add_noise(latent_inputs, new_noise, timesteps)
                else:
                    noisy_latents = noise_scheduler.add_noise(latent_inputs, noise, timesteps)
                
                # Concatenate noisy latents and conditionings to get inputs to unet
                unet_inputs = torch.cat([noisy_latents, hole_masks, latent_img_with_holes], dim=1)

                # Get the text embedding for conditioning
                text_embeds = encode_text(tokenizer, text_encoder, prompts).to(weight_dtype)

                # Get the target for loss depending on the prediction type
                if cfg.model.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=cfg.model.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latent_inputs, noise, timesteps)
                elif noise_scheduler.config.prediction_type == "sample":
                    target = latent_inputs
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                # Predict the noise residual and compute loss
                model_pred = unet(unet_inputs, timesteps, text_embeds, return_dict=False)[0]

                if cfg.model.use_rnd:
                    # RND
                    model_pred = model_pred * hole_masks + target * (1 - hole_masks)
                    model_pred = model_pred.contiguous()

                if cfg.train.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    # loss = hole_masks * F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    # loss = loss.mean()
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, cfg.train.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                if cfg.loss.use_recon_loss or cfg.loss.use_texture_loss or cfg.loss.use_fidelity_loss:
                    pred_imgs = get_pred_img(cfg, model_pred, noise_scheduler, vae, bsz, timesteps, noisy_latents)

                # Fidelity Loss
                if cfg.loss.use_fidelity_loss:
                    fidelity_loss = ReconLoss()(pred_imgs, input_imgs, coeff=cfg.loss.fidelity_loss_weight)
                    loss += fidelity_loss

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(cfg.train.batch_size)).mean()
                train_loss += avg_loss.item() / cfg.train.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = itertools.chain(
                        unet.parameters(), text_encoder.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, cfg.model.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % cfg.train.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if cfg.train.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(cfg.env.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= cfg.train.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - cfg.train.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(cfg.env.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(cfg.env.output_dir, f"checkpoint-{global_step}")
                        # save to yaml file
                        with open(os.path.join(cfg.env.output_dir, "num_update_steps_per_epoch.yaml"), "w") as f:
                            yaml.dump(num_update_steps_per_epoch, f)
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.train.max_train_steps:
                break

        if accelerator.is_main_process:
            if epoch % cfg.train.validation_epochs == 0:
                log_validation_ldm(cfg,
                               logger,
                               safety_checker,
                               test_dataloader,
                               vae, 
                               text_encoder, 
                               tokenizer, 
                               unet, 
                               accelerator,
                               pipeline_type=BodyCorrectionInpaintingPipeline,
                               noise_scheduler=noise_scheduler,
                               weight_dtype=weight_dtype, 
                               step=global_step,
                               use_rnd=cfg.model.use_rnd)

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        pipeline = BodyCorrectionInpaintingPipeline.from_pretrained(
            cfg.model.pretrained_model_name_or_path,
            vae=vae,
            text_encoder=unwrap_model(accelerator, text_encoder, enable_lora=True),
            tokenizer=tokenizer,
            unet=unwrap_model(accelerator, unet, cfg.train.enable_lora),
            safety_checker=safety_checker,
            revision=cfg.model.revision,
            variant=cfg.model.variant,
        )
        pipeline.scheduler = noise_scheduler
        
        final_model_path = f"{cfg.env.output_dir}/final_model"
        pipeline.save_pretrained(final_model_path)
    
        pred_images, pred_image_only_holes = log_validation_ldm(cfg,
                               logger,
                               safety_checker,
                               test_dataloader,
                               vae, 
                               text_encoder, 
                               tokenizer, 
                               unet, 
                               accelerator,
                               pipeline_type=BodyCorrectionInpaintingPipeline,
                               noise_scheduler=noise_scheduler,
                               weight_dtype=weight_dtype, 
                               step=global_step
                               )

        if cfg.huggingface.push_to_hub:
            save_model_card(cfg, repo_id, pred_images, repo_folder=cfg.env.output_dir)
            upload_folder(
                repo_id=repo_id,
                folder_path=cfg.env.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )


    accelerator.end_training()
    logger.info("Training completed.")

if __name__ == "__main__":
    main()
