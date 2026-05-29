import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import yaml
from PIL import Image
from accelerate import Accelerator
from diffusers.utils import (
    is_wandb_available,
    make_image_grid,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from easydict import EasyDict as edict
from torchvision.transforms import ToPILImage
from torchvision.transforms.functional import crop
from math import exp
from typing import Tuple, Union
from models.bcd_pipeline import BodyCorrectionInpaintingPipelineWithoutVAE, AlignmentPipeline, BodyCorrectionInpaintingPipeline, BodyCorrectionPipelineWithoutVAE
from tools.wandb import wandb_log

if is_wandb_available():
    import wandb

def resize_tensor(tensor, shape_list) -> torch.Tensor:
    b, c, h, w = tensor.shape
    resized_tensors = []
    for i in range(b):
        target_shape = shape_list[i]
        resized_tensor = F.interpolate(tensor[i].unsqueeze(0), size=target_shape, mode='bilinear', align_corners=False)
        resized_tensors.append(resized_tensor.squeeze(0))
    return torch.stack(resized_tensors)

def save_refine_net(save_path, refine_net):
    refine_net_dir = os.path.join(save_path, 'refine_net')
    if not os.path.exists(refine_net_dir):
        os.makedirs(refine_net_dir)
    torch.save(refine_net.state_dict(), f'{refine_net_dir}/pytorch_model.bin')
    print(f"[refine_net] weights saved in {refine_net_dir}/pytorch_model.bin")

def load_refine_net(save_path, refine_net):
    refine_net_dir = os.path.join(save_path, 'refine_net')
    refine_net.load_state_dict(torch.load(f'{refine_net_dir}/pytorch_model.bin'))
    print(f"[refine_net] weights loaded from {refine_net_dir}/pytorch_model.bin")
    return refine_net

def save_attn_model(save_path, image_proj_model, adapter_modules):
    image_proj_model_dir = os.path.join(save_path, 'image_proj_model')
    adapter_modules_dir = os.path.join(save_path, 'adapter_modules')
    if not os.path.exists(image_proj_model_dir):
        os.makedirs(image_proj_model_dir)
    if not os.path.exists(adapter_modules_dir):
        os.makedirs(adapter_modules_dir)
    torch.save(image_proj_model.state_dict(), f'{image_proj_model_dir}/pytorch_model.pth')
    torch.save(adapter_modules.state_dict(), f'{adapter_modules_dir}/pytorch_model.pth')
    print(f"[img attn] image proj model weights saved in {image_proj_model_dir}/pytorch_model.pth")
    print(f"[img attn] adapter modules weights saved in {adapter_modules_dir}/pytorch_model.pth")

def load_attn_model(save_path, image_proj_model, adapter_modules):
    image_proj_model_dir = os.path.join(save_path, 'image_proj_model')
    adapter_modules_dir = os.path.join(save_path, 'adapter_modules')
    image_proj_model.load_state_dict(torch.load(f'{image_proj_model_dir}/pytorch_model.pth'))
    adapter_modules.load_state_dict(torch.load(f'{adapter_modules_dir}/pytorch_model.pth'))
    print(f"[img attn] image proj model weights loaded from {image_proj_model_dir}/pytorch_model.pth")
    print(f"[img attn] adapter modules weights loaded from {adapter_modules_dir}/pytorch_model.pth")
    return image_proj_model, adapter_modules

def make_mask(resolution, allow_region_mask, times=30, sampling_mode='uniform'):
    """
    :param resolution: resolution of the images (H, W)
    :param allow_region_mask: mask of the region where the mask can be applied [H, W]
    :param times: number of mask patches to apply
    :param sampling_mode: sampling mode for the mask patches, 'uniform' or 'random'
    :return: mask to apply to the images [H, W]
    """
    # Create an initial mask of ones
    mask, times = np.zeros_like(allow_region_mask), np.random.randint(2, times)
    
    # Define size limits
    min_size_h, max_size_h, margin_h = np.array([0.03, 0.25, 0.01]) * resolution[0]
    min_size_w, max_size_w, margin_w = np.array([0.03, 0.25, 0.01]) * resolution[1]
    max_size_h = min(max_size_h, resolution[0] - margin_h * 2)
    max_size_w = min(max_size_w, resolution[1] - margin_w * 2)

    # Get valid positions from the allow_region_mask (locations where allow_region_mask == 1)
    valid_positions = np.nonzero(allow_region_mask)

    if valid_positions[0].size == 0:
        raise ValueError("No valid positions in the allow_region_mask.")

    selected_centers = []
    
    if sampling_mode == 'uniform':
        # Automatically calculate interval based on valid positions and times
        interval = valid_positions[0].size // times
        start_idx = np.random.randint(0, valid_positions[0].size - interval)
        for i in range(times):
            center_idx = (start_idx + i * interval) % valid_positions[0].size  # Loop around if necessary
            selected_centers.append((valid_positions[0][center_idx], valid_positions[1][center_idx]))
    elif sampling_mode == 'random':
        # Random sampling: shuffle and pick the first 'times' centers
        shuffled_indices = np.random.permutation(valid_positions[0].size)
        selected_centers = [(valid_positions[0][i], valid_positions[1][i]) for i in shuffled_indices[:times]]
    else:
        raise ValueError(f"Unknown sampling_mode: {sampling_mode}. Choose 'uniform' or 'random'.")

    # Generate mask patches based on the selected centers
    for center_y, center_x in selected_centers:
        # Randomly choose the width and height of the patch
        width = np.random.randint(int(min_size_w), int(max_size_w))
        height = np.random.randint(int(min_size_h), int(max_size_h))

        # Calculate the top-left corner of the patch based on the center position
        x_start = max(int(center_x - width // 2), int(margin_w))
        y_start = max(int(center_y - height // 2), int(margin_h))

        # Ensure the patch stays within the image boundary
        x_start = min(x_start, resolution[1] - width - int(margin_w))
        y_start = min(y_start, resolution[0] - height - int(margin_h))

        # Apply the mask
        mask[y_start:y_start + height, x_start:x_start + width] = 1

    # mask = 1 - mask if random.random() < 0.5 else mask

    return mask

def test_vae(vae, latent, input_img=None):
    # test vae
    pred_latent = 1 / vae.config.scaling_factor * latent
    pred_image = vae.decode(pred_latent).sample
    pred_image = pred_image * 0.5 + 0.5
    pred_image = pred_image.clamp(0, 1)
    save_image = pred_image.permute(0, 2, 3, 1).detach().cpu().numpy()[0] * 255
    # to BGR
    save_image = cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR)
    if not os.path.exists('test_img'):
        os.makedirs('test_img')
    cv2.imwrite('test_img/vae_recon_img.jpg', save_image)
    if input_img is not None:
        # [-1, 1] -> [0, 1]
        save_image = input_img * 0.5 + 0.5
        save_image = save_image.permute(0, 2, 3, 1).detach().cpu().numpy()[0] * 255
        # to BGR
        save_image = cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite('test_img/vae_input_img.jpg', save_image)

def log_validation_dm(cfg,
                      logger,
                      test_dataloader,
                      text_encoder,
                      tokenizer,
                      unet,
                      accelerator:Accelerator,
                      pipeline_type: Union[BodyCorrectionPipelineWithoutVAE,BodyCorrectionInpaintingPipelineWithoutVAE],
                      noise_scheduler,
                      weight_dtype,
                      step,
                      use_rnd=False):
    logger.info("Running validation... ")

    pipeline = pipeline_type(
        text_encoder=unwrap_model(accelerator, text_encoder),
        tokenizer=tokenizer,
        unet=unwrap_model(accelerator, unet),
        scheduler=noise_scheduler
    )

    # Keep the prediction type of the scheduler consistent with training
    if cfg.model.prediction_type is not None:
        # set prediction_type of scheduler if defined
        pipeline.scheduler.register_to_config(prediction_type=cfg.model.prediction_type)

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if cfg.train.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()
    
    if cfg.env.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(cfg.env.seed)

    # save num
    track_img_num = 6
    track_img_count = 0

    avg_mse = 0
    avg_psnr = 0
    avg_ssim = 0
    mse_list = []
    psnr_list = []
    ssim_list = []

    # metrices
    metrices_df = pd.DataFrame(columns=['img_id', 'MSE', 'SSIM', 'PSNR'])
    
    for batch in test_dataloader:
        pred_images = []
        pred_image_only_holes = []
        # read data
        input_imgs, shape_imgs, \
        line_mask_imgs, hole_masks, \
        hole_imgs, target_shapes, img_ids = read_data(batch, accelerator.device, weight_dtype, phase='test')

        if cfg.env.task == 'bcd-inpainting':
            conditions = hole_imgs
            only_recon_hole = False
        elif cfg.env.task == 'bcd':
            # resize
            input_imgs = F.interpolate(input_imgs, size=(cfg.data.infer_h, cfg.data.infer_w), mode='bilinear', align_corners=True)
            shape_imgs = F.interpolate(shape_imgs, size=(cfg.data.infer_h, cfg.data.infer_w), mode='bilinear', align_corners=True)
            conditions = input_imgs
            only_recon_hole = False
        for condition in conditions:
            condition = condition.unsqueeze(0).to(accelerator.device)
            if isinstance(pipeline, BodyCorrectionInpaintingPipelineWithoutVAE):
                BCD_outputs = pipeline("a photo of sks",
                                    condition, 
                                    hole_masks, 
                                    num_inference_steps=cfg.train.val_sample_steps, 
                                    generator=generator,
                                    output_type='ndarray', 
                                    only_recon_hole=only_recon_hole,
                                    use_rnd=use_rnd,
                                    type='ldm')
            elif isinstance(pipeline, BodyCorrectionPipelineWithoutVAE):
                BCD_outputs = pipeline("a photo of sks",
                                    condition, 
                                    num_inference_steps=cfg.train.val_sample_steps, 
                                    output_type='ndarray')
            pred_image = BCD_outputs.pred_images[0]
            pred_image = torch.from_numpy(pred_image).permute(2, 0, 1)
            pred_images.append(pred_image)
            
            if only_recon_hole:
                pred_image_only_hole = BCD_outputs.pred_image_only_holes[0]
                pred_image_only_hole = torch.from_numpy(pred_image_only_hole).permute(2, 0, 1)
                pred_image_only_holes.append(pred_image_only_hole)
            
        pred_images = torch.stack(pred_images).to(accelerator.device)
        if only_recon_hole:
            pred_image_only_holes = torch.stack(pred_image_only_holes).to(accelerator.device)
        
        if cfg.env.task == 'bcd-inpainting':
            gt_imgs = img_postprocess(input_imgs).to(accelerator.device)
            hole_imgs = img_postprocess(hole_imgs).to(accelerator.device)
        elif cfg.env.task == 'bcd':
            gt_imgs = img_postprocess(shape_imgs).to(accelerator.device)
            input_imgs = img_postprocess(input_imgs).to(accelerator.device)
            hole_imgs = None

        # compute criterions
        if only_recon_hole:
            mse_value, ssim_value, psnr_value = compute_criterions(pred_image_only_holes, hole_imgs)
        else:
            mse_value, ssim_value, psnr_value = compute_criterions(pred_images, gt_imgs)
        mse_list.append(mse_value.item())
        ssim_list.append(ssim_value.item())
        psnr_list.append(psnr_value.item())

        # save to pd
        row = pd.DataFrame({'img_id':[img_ids[0]],
                            'MSE':[mse_value.item()],
                            'SSIM':[ssim_value.item()],
                            'PSNR':[psnr_value.item()]})
        metrices_df = pd.concat([metrices_df, row], ignore_index=True)

        # save img
        if cfg.train.val_save_output:
            save_dir = cfg.train.val_save_dir
            if cfg.env.task == 'bcd-inpainting':
                if only_recon_hole:
                    kwargs = {'input_img': hole_imgs,
                            'pred': pred_images,
                            'pred_hole': pred_image_only_holes,
                            'gt': gt_imgs}
                else:
                    kwargs = {'input_img': hole_imgs,
                            'pred': pred_images,
                            'gt': gt_imgs}
                last_id, img_dict = save_model_output(img_ids, 
                                                    save_dir,
                                                    **kwargs,)
            elif cfg.env.task == 'bcd':
                last_id, img_dict = save_model_output(img_ids, 
                                                    save_dir,
                                                    input_img=input_imgs,
                                                    pred=pred_images,
                                                    gt=gt_imgs,)
                                                
        if track_img_count < track_img_num:
            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    np_images = np.stack([np.asarray(img) for img in pred_images])
                    tracker.writer.add_images("validation", np_images, step, dataformats="NHWC")
                elif tracker.name == "wandb":
                    wandb_log(tracker,
                              input_image=img_dict['input_img'],
                              pred_image=img_dict['pred_hole'] if only_recon_hole else img_dict['pred'],
                              gt_image=img_dict['gt'],
                              img_id=last_id,
                              step=step)
                else:
                    logger.warning(f"image logging not implemented for {tracker.name}")
                
            track_img_count += 1

    # average criterions
    avg_mse = sum(mse_list) / len(mse_list)
    avg_ssim = sum(ssim_list) / len(ssim_list)
    avg_psnr = sum(psnr_list) / len(psnr_list)

    row = pd.DataFrame({'img_id': ['average'],
                        'MSE': [avg_mse],
                        'SSIM': [avg_ssim],
                        'PSNR': [avg_psnr]})
    metrices_df = pd.concat([metrices_df, row], ignore_index=True)
    metrices_df.to_csv(f'{cfg.env.log_dir}/metrices.csv', index=False)

    logger.info(f"Validation on {len(test_dataloader)} images")
    logger.info(f"Val Avg MSE: {avg_mse:.4f}")
    logger.info(f"Val Avg SSIM: {avg_ssim:.4f}")
    logger.info(f"Val Avg PSNR: {avg_psnr:.4f}")
    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            tracker.log({"val/MSE": avg_mse, "val/SSIM": avg_ssim, "val/PSNR": avg_psnr}, step=step)
    
    del pipeline
    torch.cuda.empty_cache()

    return pred_images, pred_image_only_holes

def log_validation_ldm(cfg,
                   logger,
                   safety_checker,
                   test_dataloader,
                   vae, 
                   text_encoder, 
                   tokenizer, 
                   unet, 
                   accelerator:Accelerator, 
                   pipeline_type: Union[AlignmentPipeline, BodyCorrectionInpaintingPipeline],
                   noise_scheduler,
                   weight_dtype, 
                   step,
                   use_rnd=False,
                   refine_net=None,
                   image_proj_model=None,
                   adapter_modules=None,
                   image_encoder=None,
                   ):
    logger.info("Running validation... ")

    pipeline = pipeline_type.from_pretrained(
        cfg.model.pretrained_model_name_or_path,
        scheduler=noise_scheduler,
        vae=vae,
        text_encoder=unwrap_model(accelerator, text_encoder),
        tokenizer=tokenizer,
        unet=unwrap_model(accelerator, unet),
        safety_checker=safety_checker,
        refine_net=unwrap_model(accelerator, refine_net) if refine_net is not None else None,
        image_proj_model=unwrap_model(accelerator, image_proj_model) if image_proj_model is not None else None,
        adapter_modules=unwrap_model(accelerator, adapter_modules) if adapter_modules is not None else None,
        revision=cfg.model.revision,
        variant=cfg.model.variant,
        torch_dtype=weight_dtype,
    )

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # Keep the prediction type of the scheduler consistent with training
    if cfg.model.prediction_type is not None:
        # set prediction_type of scheduler if defined
        pipeline.scheduler.register_to_config(prediction_type=cfg.model.prediction_type)

    if cfg.train.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if cfg.env.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(cfg.env.seed)

    # save num
    track_img_num = 6
    track_img_count = 0

    avg_mse = 0
    avg_psnr = 0
    avg_ssim = 0
    mse_list = []
    psnr_list = []
    ssim_list = []

    # metrices file
    metrices_df = pd.DataFrame(columns=['img_id', 'MSE', 'SSIM', 'PSNR'])

    for batch in test_dataloader:
        pred_images = []
        pred_image_only_holes = []
        # read data
        img1s, img2s, gt_imgs, mask_imgs, prompts, img_ids = read_data(batch, accelerator.device, weight_dtype, phase='test')

        # resize
        img1s = F.interpolate(img1s, size=(cfg.data.infer_h, cfg.data.infer_w), mode='bilinear', align_corners=True)
        img2s = F.interpolate(img2s, size=(cfg.data.infer_h, cfg.data.infer_w), mode='bilinear', align_corners=True)
        gt_imgs = F.interpolate(gt_imgs, size=(cfg.data.infer_h, cfg.data.infer_w), mode='bilinear', align_corners=True)
        only_recon_hole = False

        image_embeds = None
        
        assert len(img1s) == len(img2s), "img1s and img2s should have the same length"
        for img1, img2 in zip(img1s, img2s):
            img1 = img1.unsqueeze(0).to(accelerator.device)
            img2 = img2.unsqueeze(0).to(accelerator.device)
            Align_outputs = pipeline(img1, img2, 
                                num_inference_steps=cfg.train.val_sample_steps, 
                                image_embeds=image_embeds,
                                generator=generator,
                                output_type='ndarray')
            pred_image = Align_outputs.pred_images[0]
            pred_image = torch.from_numpy(pred_image).permute(2, 0, 1)
            pred_images.append(pred_image)
            
            if only_recon_hole:
                pred_image_only_hole = Align_outputs.pred_image_only_holes[0]
                pred_image_only_hole = torch.from_numpy(pred_image_only_hole).permute(2, 0, 1)
                pred_image_only_holes.append(pred_image_only_hole)
            
        pred_images = torch.stack(pred_images).to(accelerator.device)
        if only_recon_hole:
            pred_image_only_holes = torch.stack(pred_image_only_holes).to(accelerator.device)
        
        gt_imgs = img_postprocess(gt_imgs).to(accelerator.device)
        img1s = img_postprocess(img1s).to(accelerator.device)
        img2s = img_postprocess(img2s).to(accelerator.device)

        # compute criterions
        mse_value, ssim_value, psnr_value = compute_criterions(pred_images, gt_imgs)
        mse_list.append(mse_value.item())
        ssim_list.append(ssim_value.item())
        psnr_list.append(psnr_value.item())

        # save to pd
        row = pd.DataFrame({'img_id':[img_ids[0]],
                            'MSE':[mse_value.item()],
                            'SSIM':[ssim_value.item()],
                            'PSNR':[psnr_value.item()]})
        metrices_df = pd.concat([metrices_df, row], ignore_index=True)

        # save img
        if cfg.train.val_save_output:
            save_dir = cfg.train.val_save_dir
            last_id, img_dict = save_model_output(img_ids, 
                                                save_dir,
                                                img1=img1s,
                                                img2=img2s,
                                                pred=pred_images,
                                                gt=gt_imgs,)
                                                
        if track_img_count < track_img_num:
            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    np_images = np.stack([np.asarray(img) for img in pred_images])
                    tracker.writer.add_images("validation", np_images, step, dataformats="NHWC")
                elif tracker.name == "wandb":
                    wandb_log(tracker,
                              img_1 = img_dict['img1'],
                              img_2 = img_dict['img2'],
                              pred_image=img_dict['pred'],
                              gt_image=img_dict['gt'],
                              img_id=last_id,
                              step=step)
                else:
                    logger.warning(f"image logging not implemented for {tracker.name}")
                
            track_img_count += 1

    # average criterions
    avg_mse = sum(mse_list) / len(mse_list)
    avg_ssim = sum(ssim_list) / len(ssim_list)
    avg_psnr = sum(psnr_list) / len(psnr_list)

    row = pd.DataFrame({'img_id': ['average'],
                        'MSE': [avg_mse],
                        'SSIM': [avg_ssim],
                        'PSNR': [avg_psnr]})
    metrices_df = pd.concat([metrices_df, row], ignore_index=True)
    metrices_df.to_csv(f'{cfg.env.log_dir}/metrices.csv', index=False)

    # log criterions
    logger.info(f"Validation on {len(test_dataloader)} images")
    logger.info(f"Val Avg MSE: {avg_mse:.4f}")
    logger.info(f"Val Avg SSIM: {avg_ssim:.4f}")
    logger.info(f"Val Avg PSNR: {avg_psnr:.4f}")
    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            tracker.log({"val/MSE": avg_mse, "val/SSIM": avg_ssim, "val/PSNR": avg_psnr}, step=step)
    
    del pipeline
    torch.cuda.empty_cache()

    return pred_images, pred_image_only_holes

def log_validation_vae(cfg,
                       logger,
                       test_dataloader,
                       vae,
                       accelerator:Accelerator,
                       weight_dtype,
                       step):
    logger.info("Running validation... ")

    vae.eval()
    
    # save num
    track_img_num = 6
    track_img_count = 0

    avg_mse = 0
    avg_psnr = 0
    avg_ssim = 0
    mse_list = []
    psnr_list = []
    ssim_list = []

    # metrices file
    metrices_df = pd.DataFrame(columns=['img_id', 'MSE', 'SSIM', 'PSNR'])

    for batch in test_dataloader:
        with torch.no_grad():
            # read data
            input_imgs, shape_imgs, mask_imgs, hole_masks, \
            hole_imgs, prompts, clip_images, target_shapes, img_ids = read_data(batch, accelerator.device, weight_dtype, phase='test')

            # resize
            input_imgs = F.interpolate(input_imgs, size=(cfg.data.infer_h, cfg.data.infer_w), mode='bilinear', align_corners=True)
            
            # forward
            try:
                posterior = vae.encode(input_imgs).latent_dist
                z = posterior.sample()
                model_preds = vae.decode(z).sample
            except AttributeError:
                posterior = vae.module.encode(input_imgs).latent_dist
                z = posterior.sample()
                model_preds = vae.module.decode(z).sample

            # Compute the metrics
            mse_value, ssim_value, psnr_value = compute_criterions(model_preds, input_imgs)
            mse_list.append(mse_value.item())
            ssim_list.append(ssim_value.item())
            psnr_list.append(psnr_value.item())

        # save to pd
        row = pd.DataFrame({'img_id':[img_ids[0]],
                            'MSE':[mse_value.item()],
                            'SSIM':[ssim_value.item()],
                            'PSNR':[psnr_value.item()]})
        metrices_df = pd.concat([metrices_df, row], ignore_index=True)
        
        # unnormalize
        input_imgs = img_postprocess(input_imgs)
        model_preds = img_postprocess(model_preds)

        # save img
        if cfg.train.val_save_output:
            save_dir = cfg.train.val_save_dir
            last_id, img_dict = save_model_output(img_ids, 
                                                save_dir,
                                                gt=input_imgs,
                                                pred=model_preds,)
            
        if track_img_count < track_img_num:
            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    np_images = np.stack([np.asarray(img) for img in model_preds])
                    tracker.writer.add_images("validation", np_images, step, dataformats="NHWC")
                elif tracker.name == "wandb":
                    wandb_log(tracker,
                              input_image=img_dict['gt'],
                              gt_image=img_dict['gt'],
                              pred_image=img_dict['pred'],
                              img_id=last_id,
                              step=step)
                else:
                    logger.warning(f"image logging not implemented for {tracker.name}")
                
            track_img_count += 1

    # average criterions
    avg_mse = sum(mse_list) / len(mse_list)
    avg_ssim = sum(ssim_list) / len(ssim_list)
    avg_psnr = sum(psnr_list) / len(psnr_list)

    row = pd.DataFrame({'img_id': ['average'],
                        'MSE': [avg_mse],
                        'SSIM': [avg_ssim],
                        'PSNR': [avg_psnr]})
    metrices_df = pd.concat([metrices_df, row], ignore_index=True)
    metrices_df.to_csv(f'{cfg.env.log_dir}/metrices.csv', index=False)

    # log criterions
    logger.info(f"Validation on {len(test_dataloader)} images")
    logger.info(f"Val Avg MSE: {avg_mse:.4f}")
    logger.info(f"Val Avg SSIM: {avg_ssim:.4f}")
    logger.info(f"Val Avg PSNR: {avg_psnr:.4f}")
    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            tracker.log({"val/MSE": avg_mse, "val/SSIM": avg_ssim, "val/PSNR": avg_psnr}, step=step)

    return model_preds, avg_mse, avg_ssim, avg_psnr

def save_model_card(
    cfg,
    repo_id: str,
    distortion_inputs: list = None,
    images: list = None,
    repo_folder: str = None,
):
    img_str = ""

    for i, image_pair in enumerate(zip(distortion_inputs, images)):
        image_list = [image_pair[0], image_pair[1]]
        image_grid = make_image_grid(image_list, 1, 2)
        image_grid.save(os.path.join(repo_folder, f"correcture_pair_{i}.png"))
        img_str += f"![correcture_pair_{i}](./correcture_pair_{i}.png)\n"

    model_description = f"""
# Body Image Correction with Diffusion

This pipeline was finetuned from **{cfg.model.pretrained_model_name_or_path}**. 
You can find some example images in the following. \n
{img_str}

## Pipeline usage

You can use the pipeline like so:

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("{repo_id}", torch_dtype=torch.float16)
distortion_input = "{distortion_inputs[0]}"
image = pipeline(distortion_input).images[0]
image.save("corrected_image.png")
```

## Training info

These are the key hyperparameters used during training:

* Epochs: {cfg.train.epochs}
* Learning rate: {cfg.lr.learning_rate}
* Batch size: {cfg.train.batch_size}
* Gradient accumulation steps: {cfg.train.gradient_accumulation_steps}
* Image resolution(HxW): {cfg.data.img_size_h}x{cfg.data.img_size_w}
* Mixed-precision: {cfg.train.mixed_precision}

"""
    wandb_info = ""
    if is_wandb_available():
        wandb_run_url = None
        if wandb.run is not None:
            wandb_run_url = wandb.run.url

    if wandb_run_url is not None:
        wandb_info = f"""
More information on all the CLI arguments and the environment are available on your [`wandb` run page]({wandb_run_url}).
"""

    model_description += wandb_info

    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=cfg.model.pretrained_model_name_or_path,
        model_description=model_description,
        inference=True,
    )

    tags = ["stable-diffusion", "stable-diffusion-diffusers", "Distortion-Correction", "body-image-correction",  "inpainting", "diffusers", "diffusers-training"]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))

def unwrap_model(accelerator, model, enable_lora=False):
    if enable_lora:
        model = accelerator.unwrap_model(model, keep_fp32_wrapper=True).merge_and_unload()
    else:
        model = accelerator.unwrap_model(model, keep_fp32_wrapper=True)
    return model

def read_data(data:dict, device, dtype=torch.float32, phase='train'):
    if phase in ['train', 'test']:
        img1 = data['img1'].to(dtype).to(device)
        img2 = data['img2'].to(dtype).to(device)
        gt_img = data['gt_img'].to(dtype).to(device)
        mask_img = data['mask_img'].to(dtype).to(device)
        prompt = data['prompt']
        img_id = data['img_id']

        return img1, img2, gt_img, mask_img, prompt, img_id
    
    elif phase == 'inference':
        file_name = data['file_name']
        img1 = data['img1'].to(dtype).to(device)
        img2 = data['img2'].to(dtype).to(device)
        if data['clip_image'] is not None:
            clip_image = data['clip_image'].to(dtype).to(device)
        else:
            clip_image = None
        target_shape = data['target_shape']
        return file_name, img1, img2, clip_image, target_shape
    
    else:
        raise ValueError(f'phase [{phase}] not supported')
    
# ========== config tool =========
def edict_2_dict(x):
    '''
    This method recursively converts an edict to a dict.
    '''
    if isinstance(x, dict):
        xnew = {}
        for k in x:
            xnew[k] = edict_2_dict(x[k])
        return xnew
    elif isinstance(x, list):
        xnew = []
        for i in range(len(x)):
            xnew.append( edict_2_dict(x[i]))
        return xnew
    else:
        return x
    
def save_yaml(cfg: edict):
    config_pth = os.path.join(cfg.env.log_dir, 'config.yaml')
    with open(config_pth, 'w') as f:
        yaml.dump(edict_2_dict(cfg), f)
    
def save_one_img(img_ids, i, save_dir, **kwargs):
    img_id = img_ids[i]
    save_dir = f'{save_dir}/{img_id}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img_dict = {}
    # gts and masks
    for arg_k, arg_v in kwargs.items():
        img=arg_v[i]
        save_pth = f'{save_dir}/{img_id}_{arg_k}.png'
        if isinstance(img, torch.Tensor):
            img_pil = ToPILImage()(img)
        else:
            img_pil = img
        img_pil.save(save_pth)
        img_dict[arg_k] = img

    return img_id, img_dict

def save_model_output(img_ids, save_dir='debug_img/', **kwargs):
    for i in range(len(img_ids)): # batch size
        img_id, img_dict = save_one_img(img_ids, i, save_dir, **kwargs)
    return img_id, img_dict

def img_postprocess(img):
    '''
    de-normalize the image

    args:
        img (torch.Tensor): input image
        scale (float): scale factor of the input image
    
    returns:
        img (Image): de-normalized image
    '''
    img = img * 0.5 + 0.5
    img = img.clamp(0, 1)

    return img

class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = self.create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = self.create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return self.ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel=1):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def ssim(self, img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None) -> torch.Tensor:
        # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
        if val_range is None:
            if torch.max(img1) > 128:
                max_val = 255
            else:
                max_val = 1

            if torch.min(img1) < -0.5:
                min_val = -1
                max_val = 1
            else:
                min_val = 0
            L = max_val - min_val
        else:
            L = val_range

        padd = 0
        (_, channel, height, width) = img1.size()
        if window is None:
            real_size = min(window_size, height, width)
            window = self.create_window(real_size, channel=channel).to(img1.device)

        mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
        mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)  # contrast sensitivity

        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

        if size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)

        if full:
            return ret, cs
        
        return ret

def compute_criterions(x_img_batch, x_gt_img_batch, val_range=1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    Compute the mse ,psnr and ssim between x_img_batch and x_gt_img_batch
    '''
    if isinstance(x_img_batch, np.ndarray):
        x_img_batch = torch.tensor(x_img_batch)
    if isinstance(x_gt_img_batch, np.ndarray):
        x_gt_img_batch = torch.tensor(x_gt_img_batch)
    
    mse_value = torch.mean(torch.square(x_img_batch * val_range - x_gt_img_batch * val_range), dim=(1, 2, 3)).mean()
    psnr_value = 10 * torch.log10(val_range * val_range / mse_value).mean()
    ssim_value = SSIM()(x_img_batch, x_gt_img_batch)
    
    return mse_value, ssim_value, psnr_value

class RandomInsideCrop(nn.Module):
    '''
    Apply random crop to the input inside the image.
    This method is different from torch.nn.functional.RandomCrop. It won't crop the image outside the image.

    Args:
        init:
            crop_size (Tuple[int, int]): The size of the crop. [H, W]
        forward:
            image (torch.Tensor or PIL.Image.Image): The input image.
            seed (int): Random seed.

    Returns:
        image (torch.Tensor or PIL.Image.Image): The cropped image.

    '''
    def __init__(self, crop_size:Tuple[int, int]=(512, 512)):
        super().__init__()
        self.crop_h = crop_size[0]
        self.crop_w = crop_size[1]

    def forward(self, image, seed=None):
        if seed is not None:
            local_random = np.random.RandomState(seed)
        else:
            raise ValueError('For each data, random seed is required')
        
        # RandomCrop
        if isinstance(image, torch.Tensor):
            h = image.size(1)
            w = image.size(2)
        elif isinstance(image, Image.Image):
            w, h = image.size
        top = local_random.randint(0, h - self.crop_h)
        left = local_random.randint(0, w - self.crop_w)
        image = crop(image, top, left, self.crop_h, self.crop_w)

        return image


class CropbyMask(nn.Module):
    '''
    Apply random crop to the input inside the image.
    This method is different from torch.nn.functional.RandomCrop. It won't crop the image outside the image.

    Args:
        init:
            crop_size (Tuple[int, int]): The size of the crop. [H, W]
        forward:
            image (torch.Tensor or PIL.Image.Image): The input image.
            seed (int): Random seed.

    Returns:
        image (torch.Tensor or PIL.Image.Image): The cropped image.

    '''
    def __init__(self, crop_size:Tuple[int, int]=(512, 512)):
        super().__init__()
        self.crop_h = crop_size[0]
        self.crop_w = crop_size[1]
        self.patch_size_h = crop_size[0]
        self.patch_size_w = crop_size[1]

    def forward(self, image, input_mask):
        if isinstance(image, Image.Image):
            image = np.array(image)
        if isinstance(input_mask, Image.Image):
            input_mask = np.array(input_mask)
        flow_mask = np.argwhere(input_mask > 0)
        mean_coords = np.mean(flow_mask, axis=0)
        mean_coords = np.round(mean_coords).astype(int)

        ### crop
        crop_h = mean_coords[0] - self.patch_size_h//2
        crop_w = mean_coords[1] - self.patch_size_w//2
        if(crop_h<0):
            crop_h = 0
        if(crop_w<0):
            crop_w = 0

        if((crop_h+self.patch_size_h) > input_mask.shape[0]):
            crop_h = input_mask.shape[0] - self.patch_size_h
        if((crop_w+self.patch_size_w) > input_mask.shape[1]):
            crop_w = input_mask.shape[1] - self.patch_size_w

        image = image[crop_h:(crop_h+self.patch_size_h), crop_w:(crop_w+self.patch_size_w), :]

        image = Image.fromarray(image)
        return image
    
def is_torch2_available():
    return hasattr(F, "scaled_dot_product_attention")