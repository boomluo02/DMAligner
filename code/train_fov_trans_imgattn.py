import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import random
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import yaml
import shutil

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from diffusers.training_utils import compute_snr
from huggingface_hub import upload_folder
from tqdm.auto import tqdm

from models.bcd_pipeline import AlignmentPipeline, encode_empty_text, encode_text # noqa
from tools.init_1 import initialize
from tools.loss import L2Loss, ReconLoss, SobelLoss, calc_texture_loss, get_pred_img
from tools.utils_1 import log_validation_ldm, read_data, save_attn_model, save_model_card, test_vae, unwrap_model # noqa


def main():
    # =========== initialize ===========
    init_dict = initialize(model_type="ldm")

    cfg = init_dict["cfg"]
    logger = init_dict["logger"]
    accelerator = init_dict["accelerator"]
    safety_checker = init_dict["safety_checker"]
    train_dataloader = init_dict["train_dataloader"]
    test_dataloader = init_dict["test_dataloader"]
    unet = init_dict["unet"]
    vae = init_dict["vae"]
    text_encoder = init_dict["text_encoder"]
    noise_scheduler = init_dict["noise_scheduler"]
    optimizer = init_dict["optimizer"]
    lr_scheduler = init_dict["lr_scheduler"]
    tokenizer = init_dict["tokenizer"]
    weight_dtype = init_dict["weight_dtype"]
    num_update_steps_per_epoch = init_dict["num_update_steps_per_epoch"]
    repo_id = init_dict["repo_id"]

    if cfg.env.decoupled_attn:
        image_encoder = init_dict["image_encoder"]
        image_proj_model = init_dict["image_proj_model"]
        adapter_modules = init_dict["adapter_modules"]

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
        train_loss = 0.0
        train_mse_loss = 0.0
        train_fidelity_loss = 0.0
        train_struct_loss = 0.0
        train_texture_loss = 0.0
        train_recon_loss = 0.0
        train_feature_sobel_loss = 0.0
        train_sobel_loss = 0.0
        train_adaptive_recon_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet, image_encoder, image_proj_model, adapter_modules):
            # with accelerator.accumulate(unet):
                # read data
                input_imgs, shape_imgs, mask_imgs, hole_masks, \
                hole_imgs, prompts, clip_images, target_shapes, img_ids = read_data(batch, accelerator.device, weight_dtype, phase='train')
                
                # Convert images to latent space
                latent_inputs = vae.encode(shape_imgs).latent_dist.sample()
                latent_inputs = latent_inputs * vae.config.scaling_factor
                # test_vae(vae, latent_inputs, shape_imgs)

                # conditions 
                latent_condition = vae.encode(input_imgs).latent_dist.sample()
                latent_condition = latent_condition * vae.config.scaling_factor

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
                unet_inputs = torch.cat([noisy_latents, latent_condition], dim=1)

                with torch.no_grad():
                    image_embeds = image_encoder(clip_images.to(accelerator.device, dtype=weight_dtype)).image_embeds
                image_embeds_ = []
                for image_embed in image_embeds:
                    if random.random() < cfg.train.image_embed_dropout:
                        image_embeds_.append(torch.zeros_like(image_embed))
                    else:
                        image_embeds_.append(image_embed)
                image_embeds = torch.stack(image_embeds_)
                image_tokens = image_proj_model(image_embeds)

                # Get the text embedding for conditioning
                text_embeds = image_embeds.unsqueeze(1)

                encoder_hidden_states = torch.cat([text_embeds, image_tokens], dim=1)

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
                model_pred = unet(unet_inputs, timesteps, encoder_hidden_states, return_dict=False)[0]

                if cfg.train.snr_gamma is None:
                    mse_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
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

                    mse_loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    mse_loss = mse_loss.mean(dim=list(range(1, len(mse_loss.shape)))) * mse_loss_weights
                    mse_loss = mse_loss.mean()
                
                mse_loss = mse_loss * cfg.loss.mse_loss_weight
                loss = mse_loss.clone()

                if cfg.loss.use_recon_loss or cfg.loss.use_texture_loss or cfg.loss.use_fidelity_loss or cfg.loss.use_struct_loss or cfg.loss.use_adaptive_recon_loss or cfg.loss.use_sobel_loss:
                    pred_imgs = get_pred_img(cfg, model_pred, noise_scheduler, vae, bsz, timesteps, noisy_latents)
                
                if cfg.loss.use_recon_loss:
                    recon_loss = L2Loss()(pred_imgs, shape_imgs, coeff=cfg.loss.recon_loss_weight)
                    loss += recon_loss
                else:
                    recon_loss = torch.tensor(0.0, device=accelerator.device)

                if cfg.loss.use_sobel_loss:
                    feature_sobel_loss = SobelLoss()(model_pred.float(), target.float(), coeff=cfg.loss.sobel_loss_weight)
                    sobel_loss = SobelLoss()(pred_imgs, shape_imgs, coeff=cfg.loss.sobel_loss_weight)
                    loss = loss + feature_sobel_loss + sobel_loss
                else:
                    feature_sobel_loss = torch.tensor(0.0, device=accelerator.device)
                    sobel_loss = torch.tensor(0.0, device=accelerator.device)

                if cfg.loss.use_adaptive_recon_loss:
                    if global_step < cfg.loss.struct_step:
                        mode = 'structural'
                    else:
                        mode = 'fidelity'
                    adaptive_recon_loss = ReconLoss(mode=mode)(pred_imgs, shape_imgs, coeff=cfg.loss.adaptive_recon_loss_weight)
                    loss += adaptive_recon_loss
                    fidelity_loss = torch.tensor(0.0, device=accelerator.device)
                    struct_loss = torch.tensor(0.0, device=accelerator.device)
                else:
                    adaptive_recon_loss = torch.tensor(0.0, device=accelerator.device)
                    if cfg.loss.use_fidelity_loss:
                        fidelity_loss = ReconLoss(mode='fidelity')(pred_imgs, shape_imgs, coeff=cfg.loss.fidelity_loss_weight)
                        loss += fidelity_loss
                    else:
                        fidelity_loss = torch.tensor(0.0, device=accelerator.device)
                    
                    if cfg.loss.use_struct_loss:
                        struct_loss = ReconLoss(mode='structural')(pred_imgs, shape_imgs, coeff=cfg.loss.struct_loss_weight)
                        loss += struct_loss
                    else:
                        struct_loss = torch.tensor(0.0, device=accelerator.device)

                # Texture loss
                if cfg.loss.use_texture_loss:
                    texture_loss = calc_texture_loss(cfg, pred_imgs, shape_imgs)
                    loss += texture_loss
                else:
                    texture_loss = torch.tensor(0.0, device=accelerator.device)
                    
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(cfg.train.batch_size)).mean()
                avg_mse_loss = accelerator.gather(mse_loss.repeat(cfg.train.batch_size)).mean()
                avg_fidelity_loss = accelerator.gather(fidelity_loss.repeat(cfg.train.batch_size)).mean()
                avg_struct_loss = accelerator.gather(struct_loss.repeat(cfg.train.batch_size)).mean()
                avg_texture_loss = accelerator.gather(texture_loss.repeat(cfg.train.batch_size)).mean()
                avg_recon_loss = accelerator.gather(recon_loss.repeat(cfg.train.batch_size)).mean()
                avg_feature_sobel_loss = accelerator.gather(feature_sobel_loss.repeat(cfg.train.batch_size)).mean()
                avg_sobel_loss = accelerator.gather(sobel_loss.repeat(cfg.train.batch_size)).mean()
                avg_adaptive_recon_loss = accelerator.gather(adaptive_recon_loss.repeat(cfg.train.batch_size)).mean()

                train_loss += avg_loss.item() / cfg.train.gradient_accumulation_steps
                train_mse_loss += avg_mse_loss.item() / cfg.train.gradient_accumulation_steps
                train_fidelity_loss += avg_fidelity_loss.item() / cfg.train.gradient_accumulation_steps
                train_struct_loss += avg_struct_loss.item() / cfg.train.gradient_accumulation_steps
                train_texture_loss += avg_texture_loss.item() / cfg.train.gradient_accumulation_steps
                train_recon_loss += avg_recon_loss.item() / cfg.train.gradient_accumulation_steps
                train_feature_sobel_loss += avg_feature_sobel_loss.item() / cfg.train.gradient_accumulation_steps
                train_sobel_loss += avg_sobel_loss.item() / cfg.train.gradient_accumulation_steps
                train_adaptive_recon_loss += avg_adaptive_recon_loss.item() / cfg.train.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), cfg.model.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss, 
                                 "mse_loss": train_mse_loss,
                                 "fidelity_loss": train_fidelity_loss, 
                                 "struct_loss": train_struct_loss, 
                                 "texture_loss": train_texture_loss, 
                                 "recon_loss": train_recon_loss,
                                 "feature_sobel_loss": train_feature_sobel_loss,
                                 "sobel_loss": train_sobel_loss,
                                 "adaptive_recon_loss": train_adaptive_recon_loss, 
                                 "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
                train_loss = 0.0
                train_mse_loss = 0.0
                train_fidelity_loss = 0.0
                train_struct_loss = 0.0
                train_texture_loss = 0.0
                train_recon_loss = 0.0
                train_feature_sobel_loss = 0.0
                train_sobel_loss = 0.0
                train_adaptive_recon_loss = 0.0

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
                # if validation time too long will raise NCCL time out error
                log_validation_ldm(cfg,
                               logger,
                               safety_checker,
                               test_dataloader,
                               vae, 
                               text_encoder, 
                               tokenizer, 
                               unet, 
                               accelerator,
                               pipeline_type=AlignmentPipeline,
                               noise_scheduler=noise_scheduler,
                               weight_dtype=weight_dtype, 
                               step=global_step,
                               image_proj_model=image_proj_model,
                               adapter_modules=adapter_modules,
                               image_encoder=image_encoder,)

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        pipeline = AlignmentPipeline.from_pretrained(
            cfg.model.pretrained_model_name_or_path,
            vae=vae,
            text_encoder=unwrap_model(accelerator, text_encoder),
            tokenizer=tokenizer,
            unet=unwrap_model(accelerator, unet, cfg.train.enable_lora),
            safety_checker=safety_checker,
            image_proj_model=unwrap_model(accelerator, image_proj_model),
            adapter_modules=unwrap_model(accelerator, adapter_modules),
            revision=cfg.model.revision,
            variant=cfg.model.variant,
        )
        pipeline.scheduler = noise_scheduler

        final_model_path = f"{cfg.env.output_dir}/final_model"
        pipeline.save_pretrained(final_model_path)
        save_attn_model(final_model_path, pipeline.image_proj_model, adapter_modules)

        pred_images, _ = log_validation_ldm(cfg,
                               logger,
                               safety_checker,
                               test_dataloader,
                               vae, 
                               text_encoder, 
                               tokenizer, 
                               unet, 
                               accelerator,
                               pipeline_type=AlignmentPipeline,
                               noise_scheduler=noise_scheduler,
                               weight_dtype=weight_dtype, 
                               step=global_step,
                               image_proj_model=image_proj_model,
                               adapter_modules=adapter_modules,
                               image_encoder=image_encoder,)

        if cfg.huggingface.push_to_hub:
            save_model_card(cfg, repo_id, pred_images, repo_folder=cfg.env.output_dir)
            upload_folder(
                repo_id=repo_id,
                folder_path=cfg.env.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    main()
