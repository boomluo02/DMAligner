import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch.nn.functional as F
from huggingface_hub import upload_folder
from tqdm.auto import tqdm

from tools.init_1 import initialize_vae
from tools.utils_1 import compute_criterions, img_postprocess, log_validation_ldm, log_validation_vae, read_data, save_model_card, test_vae  # noqa: F401

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

def main():
    # =========== initialize ===========
    cfg, logger, accelerator, \
    train_dataloader, test_dataloader, \
    vae, optimizer, lr_scheduler, lpips_loss_fn, \
    weight_dtype, num_update_steps_per_epoch, repo_id = initialize_vae()

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

    max_psnr = 0

    for epoch in range(first_epoch, cfg.train.epochs):
        vae.train()

        train_loss = 0.0
        train_mse = 0.0
        train_ssim = 0.0
        train_psnr = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(vae):
                # read data
                input_imgs, shape_imgs, mask_imgs, hole_masks, \
                hole_imgs, prompts, clip_images, target_shapes, img_ids = read_data(batch, accelerator.device, weight_dtype, phase='train')
                
                try:
                    posterior = vae.encode(input_imgs).latent_dist
                    z = posterior.sample()
                    model_preds = vae.decode(z).sample
                except AttributeError: # for DDP
                    posterior = vae.module.encode(input_imgs).latent_dist
                    z = posterior.sample()
                    model_preds = vae.module.decode(z).sample

                # # Reconstruction Loss
                # rec_loss = torch.abs(model_preds - input_imgs).mean()
                # lpips_loss = lpips_loss_fn(model_preds, input_imgs).mean()
                # loss = rec_loss + lpips_loss
                
                loss = F.mse_loss(model_preds.contiguous(), input_imgs.contiguous(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(cfg.train.batch_size)).mean()
                train_loss = avg_loss.item() / cfg.train.gradient_accumulation_steps

                # Compute the metrics
                mse_value, ssim_value, psnr_value = compute_criterions(model_preds, input_imgs)
                avg_mse = accelerator.gather(mse_value.repeat(cfg.train.batch_size)).mean()
                avg_ssim = accelerator.gather(ssim_value.repeat(cfg.train.batch_size)).mean()
                avg_psnr = accelerator.gather(psnr_value.repeat(cfg.train.batch_size)).mean()
                train_mse = avg_mse.item() / cfg.train.gradient_accumulation_steps
                train_ssim = avg_ssim.item() / cfg.train.gradient_accumulation_steps
                train_psnr = avg_psnr.item() / cfg.train.gradient_accumulation_steps
                
                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(vae.parameters(), cfg.model.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss, "train_mse": train_mse, "train_ssim": train_ssim, "train_psnr": train_psnr})

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.train.max_train_steps:
                break

        if accelerator.is_main_process:
            if epoch % cfg.train.validation_epochs == 0:
                _, mse, ssim, psnr = log_validation_vae(cfg,
                                   logger,
                                   test_dataloader,
                                   vae,
                                   accelerator,
                                   weight_dtype=weight_dtype,
                                   step=global_step)
                # best
                if psnr > max_psnr:
                    max_psnr = psnr
                    save_path = f"{cfg.env.output_dir}/checkpoint_best_psnr"
                    accelerator.save_state(save_path)
                    logger.info(f"Best PSNR: {max_psnr} at epoch {epoch}")

                # last
                save_path = f"{cfg.env.output_dir}/checkpoint_last"
                accelerator.save_state(save_path)
                logger.info(f"Last checkpoint saved at epoch {epoch}")


    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        pred_images, mse, ssim, psnr = log_validation_vae(cfg,
                                   logger,
                                   test_dataloader,
                                   vae,
                                   accelerator,
                                   weight_dtype=weight_dtype,
                                   step=global_step)

        logger.info("-" * 50)
        logger.info("Last time save")
        # best
        if psnr > max_psnr:
            max_psnr = psnr
            save_path = f"{cfg.env.output_dir}/checkpoint_best_psnr"
            accelerator.save_state(save_path)
            logger.info(f"Best PSNR: {max_psnr} at epoch {epoch}")
        else:
            logger.info("Last epoch is not the best.")

        # last
        save_path = f"{cfg.env.output_dir}/checkpoint_last"
        accelerator.save_state(save_path)
        logger.info(f"Last checkpoint saved at epoch {epoch}")

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
