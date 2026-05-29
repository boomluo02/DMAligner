import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"  # 设置可见的GPU编号
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch
import torch.nn.functional as F
import wandb
import json
from tqdm import tqdm
from diffusers import PNDMScheduler
from config.config import set_config
from models.custom_pipeline import CustomPipeline
from tools.init import initialize
from tools.utils import compute_criterions, img_postprocess, read_data, save_model_output
from tools.wandb import show_imgs_on_wandb, wandb_log_criterion


def main(cfg):
    # ============ init ================
    accelerator, data_type, \
    unet, vae, optimizer, text_encoder, \
    tokenizer, lr_scheduler, train_loader, \
    test_loader, noise_scheduler, \
    resume_last_epoch, resume_max_psnr = initialize(cfg)

    # create pipeline for validation
    pipeline = CustomPipeline.from_pretrained(
        cfg.model.pretrained_model_name_or_path,
        revision=cfg.model.revision,
    )
    pipeline.set_progress_bar_config(disable=True)

    # ============ train ================
    if cfg.env.resume:
        max_psnr = resume_max_psnr
        start_epoch = resume_last_epoch + 1
        end_epoch = cfg.train.epochs
    else:
        max_psnr = -1
        start_epoch = 0
        end_epoch = cfg.train.epochs

    print('Start training ...')
    for epoch in range(start_epoch, end_epoch):

        # tracking progress
        if cfg.env.use_wandb and accelerator.is_local_main_process:
            wandb.log({"epoch": epoch+1}, step=epoch+1)
            wandb.log({"learning_rate": optimizer.param_groups[0]['lr']}, step=epoch+1)

        # train
        unet.train()

        # avg tracker
        avg_loss = 0

        with tqdm(total=len(train_loader), desc=f'Epoch[{epoch+1}/{end_epoch}](train)', unit='batch') as pbar:
            for step, batch in enumerate(train_loader):
                # 1. read data
                input_imgs, shape_imgs, \
                line_mask_imgs, hole_mask_imgs, \
                hole_imgs, img_ids = read_data(batch, data_type, phase='train')

                # Convert images to latent space
                latent_inputs = vae.encode(input_imgs).latent_dist.sample() * 0.18215

                # conditions 
                latent_img_with_holes = vae.encode(hole_imgs).latent_dist.sample() * 0.18215
                # masks = F.interpolate(line_mask_imgs.float(), size=latent_inputs.shape[-2:])

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latent_inputs)
                batch_size = latent_inputs.shape[0]
                device = latent_inputs.device

                # Sample a random timestep for each image
                timesteps = torch.randint(0, 
                                          noise_scheduler.config.num_train_timesteps, 
                                          (batch_size,),
                                          device=device)
                timesteps = timesteps.long()
                
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latent_inputs, noise, timesteps)

                # Concatenate noisy latents, masks and conditionings to get inputs to unet
                inputs = torch.cat([noisy_latents, latent_img_with_holes], dim=1)
                # inputs = torch.cat([noisy_latents, masks, latent_img_with_holes], dim=1)

                # prompts TODO: 检查一下是否有必要在每个step都重新生成
                prompt = ""
                text_inputs = tokenizer(
                    prompt,
                    padding="do_not_pad",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids.to(text_encoder.device)
                empty_text_embed = text_encoder(text_input_ids)[0].to(data_type)
                empty_text_embed = empty_text_embed.repeat(batch_size, 1, 1)

                # Predict
                model_pred = unet(inputs, timesteps, empty_text_embed).sample

                # Compute the loss  
                if noise_scheduler.config.prediction_type == 'sample':
                    target = input_imgs
                elif noise_scheduler.config.prediction_type == 'epsilon':
                    target = noise
                elif noise_scheduler.config.prediction_type == 'v_prediction':
                    target = noise_scheduler.get_velocity(latent_inputs, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                loss = F.mse_loss(model_pred.float(), target.float()).mean()
                
                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if cfg.model.max_grad_norm != -1:
                        accelerator.clip_grad_norm_(unet.parameters(), cfg.model.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                # log loss
                if accelerator.sync_gradients:
                    pbar.set_postfix({'loss': loss.item()})
                    pbar.update(1)
                    avg_loss += loss.item()

            avg_loss /= len(train_loader)
            print(f"Epoch {epoch+1}, Avg Loss: {avg_loss}")
            if accelerator.is_main_process and cfg.env.use_wandb:
                wandb_log_criterion({'train_loss': avg_loss}, epoch+1, 'train')
            
            # ============ save last epoch ============
            if accelerator.is_main_process:   
                model_save_pth = f'{cfg.env.CKPT_DIR}/{cfg.env.run_id}-last'
                accelerator.save_state(model_save_pth)
                resume_dict = {
                        'epoch': epoch,
                        'max_psnr': max_psnr,
                        'unet_path': model_save_pth,
                    }
                # to json
                with open(f'{cfg.env.CKPT_DIR}/{cfg.env.run_id}-last.json', 'w') as f:
                    json.dump(resume_dict, f)

            if ((epoch+1) % cfg.train.val_interval != 0 and epoch+1 != end_epoch):
                continue
            # ============ validation ================
            # every val epoch, update saved img count
            wandb_save_num = 6
            wandb_save_count = 0

            unet.eval()
            avg_val_psnr = 0
            avg_val_ssim = 0

            # update pipeline
            device = accelerator.device
            pipeline.unet = accelerator.unwrap_model(unet, keep_fp32_wrapper=True).to(device)
            pipeline.text_encoder = accelerator.unwrap_model(text_encoder, keep_fp32_wrapper=True)
            pipeline.scheduler = PNDMScheduler.from_config(pipeline.scheduler.config)

            # check device
            pipeline.to(device)
            
            # 2. predict TODO: seed的设置要改，当seed=-1的时候，不设置seed
            generator = None if cfg.env.seed == -1 else torch.Generator(device=accelerator.device).manual_seed(cfg.env.seed)
            
            with torch.no_grad():
                with tqdm(total=len(test_loader), desc=f'Epoch[{epoch+1}/{end_epoch}](val)', unit='batch') as pbar:
                    for step, batch in enumerate(test_loader):
                        # 1. read data
                        input_imgs, shape_imgs, \
                        line_mask_imgs, hole_mask_imgs, \
                        hole_imgs, img_ids = read_data(batch, phase='test')

                        # 2. predict        
                        pred_imgs = []
                        gt_imgs = []
                        input_holes = []
                        for i, hole_img in enumerate(hole_imgs):
                            pred_img = pipeline(input_image=hole_img,
                                                num_inference_steps=200, 
                                                generator=generator)
                            
                            # postprocess
                            pred_imgs.append(pred_img)
                            gt_imgs.append(img_postprocess(input_imgs[i]))
                            input_holes.append(img_postprocess(hole_img))
                        
                        pred_imgs = torch.stack(pred_imgs).to(accelerator.device)
                        gt_imgs = torch.stack(gt_imgs).to(accelerator.device)
                        input_holes = torch.stack(input_holes).to(accelerator.device)
                        _, psnr, ssim = compute_criterions(pred_imgs, gt_imgs)
                        
                        # 3. log psnr and ssim
                        avg_val_psnr += psnr.item()
                        avg_val_ssim += ssim.item()

                        # 3. save img
                        if cfg.train.val_save_output:
                            save_dir = cfg.train.val_save_dir
                            last_id, img_dict = save_model_output(img_ids, 
                                                                    save_dir,
                                                                    input_img=input_holes,
                                                                    pred=pred_imgs,
                                                                    gt=gt_imgs,)
    
                        if accelerator.is_main_process:
                            if cfg.env.use_wandb and (wandb_save_count < wandb_save_num):
                                show_imgs_on_wandb(last_id,
                                                   input_img=input_holes,
                                                   pred=pred_imgs,
                                                   gt=gt_imgs,
                                                   epoch=epoch)
                                wandb_save_count += 1

                        pbar.set_postfix({'psnr': psnr.item(), 'ssim': ssim.item()})
                        pbar.update(1)
                    
                    # log validation metrics
                    avg_val_psnr /= len(test_loader)
                    avg_val_ssim /= len(test_loader)

                    print(f"Epoch {epoch+1}, Avg Val PSNR: {avg_val_psnr}, Avg Val SSIM: {avg_val_ssim}\n")

                    if accelerator.is_main_process and cfg.env.use_wandb:
                        wandb_log_criterion({'val_psnr': avg_val_psnr, 'val_ssim': avg_val_ssim}, epoch+1, 'val')

                    # save best epoch
                    if avg_val_psnr > max_psnr:
                        max_psnr = avg_val_psnr
                        if accelerator.is_main_process:
                            model_save_pth = f'{cfg.env.CKPT_DIR}/{cfg.env.run_id}-best'
                            accelerator.save_state(model_save_pth)

                        print('-----------------------------------')
                        print(f"Model saved! Epoch {epoch+1},Best Val PSNR: {max_psnr}")
                        print('-----------------------------------')
            
            torch.cuda.empty_cache()
                
    # Save the lora layers
    if cfg.lora.use_lora:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            pipeline = CustomPipeline.from_pretrained(
                cfg.model.pretrained_model_name_or_path,
                unet=accelerator.unwrap_model(unet, keep_fp32_wrapper=True).merge_and_unload(),
                revision=cfg.model.revision,
            )

            pipeline.save_pretrained(cfg.env.log_dir)
            print(f"Saved LoRA layers to {cfg.env.log_dir}")

    # TODO: 要修改huggingface的上传设置
    # if cfg.huggingface.push_to_hub:
    #     save_model_card(
    #         repo_id,
    #         images=images,
    #         base_model=args.pretrained_model_name_or_path,
    #         repo_folder=args.output_dir,
    #     )
    #     upload_folder(
    #         repo_id=repo_id,
    #         folder_path=args.output_dir,
    #         commit_message="End of training",
    #         ignore_patterns=["step_*", "epoch_*"],
    #     )

    accelerator.end_training()
    print('Training finished!')

# program method
# we define program as a method to make it easier to run wandb sweep
def program():
    cfg = set_config()
    main(cfg)

if __name__ == '__main__':
    program()
