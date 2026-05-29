import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import pandas as pd
from models.bcd_pipeline import BodyCorrectionInpaintingPipeline, AlignmentPipeline
from tools.init_1 import initialize_test
from tools.utils_1 import compute_criterions, img_postprocess, log_validation_ldm, read_data, resize_tensor, save_model_card, save_model_output, test_vae, unwrap_model # noqa

import cv2

def main():
    # =========== initialize ===========
    cfg, logger, pipeline, image_encoder,\
    train_dataloader, test_dataloader, use_rnd, device = initialize_test()

    if cfg.env.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=device).manual_seed(cfg.env.seed)

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

    for i, batch in enumerate(test_dataloader):
        pred_images = []
        pred_image_only_holes = []
        # read data
        input_imgs, shape_imgs, mask_imgs, hole_masks, \
        hole_imgs, prompts, clip_images, target_shapes, img_ids = read_data(batch, device, phase='test')

        if cfg.env.task == 'bcd-inpainting':
            conditions = hole_imgs
            only_recon_hole = False
        elif cfg.env.task == 'bcd':
            if cfg.data.change_infer_size:
                # resize
                input_imgs = F.interpolate(input_imgs, size=(cfg.data.infer_h, cfg.data.infer_w), mode='bilinear', align_corners=True)
                shape_imgs = F.interpolate(shape_imgs, size=(cfg.data.infer_h, cfg.data.infer_w), mode='bilinear', align_corners=True)
            conditions = input_imgs
            only_recon_hole = False
        
        if clip_images is not None:
            with torch.no_grad():
                image_embeds = image_encoder(clip_images.to(device, dtype=torch.float32)).image_embeds
            image_embeds_ = []
            for image_embed in image_embeds:
                image_embeds_.append(image_embed)
            image_embeds = torch.stack(image_embeds_)
        else:
            image_embeds = None

        for condition in conditions:
            print(f"Inferencing [{i+1}/{len(test_dataloader)}] image {img_ids[0]}:")
            condition = condition.unsqueeze(0).to(device)
            if isinstance(pipeline, BodyCorrectionInpaintingPipeline):
                BCD_outputs = pipeline("a photo of sks",
                                    condition, 
                                    hole_masks, 
                                    num_inference_steps=cfg.model.num_inference_steps, 
                                    generator=generator,
                                    output_type='ndarray', 
                                    only_recon_hole=only_recon_hole,
                                    use_rnd=use_rnd,
                                    type='ldm')
            elif isinstance(pipeline, AlignmentPipeline):
                BCD_outputs, pred_seq = pipeline(condition, 
                                    num_inference_steps=cfg.model.num_inference_steps, 
                                    image_embeds=image_embeds,
                                    generator=generator,
                                    output_type='ndarray',
                                    return_seq=True,)
                
            pred_image = BCD_outputs.pred_images[0]
            pred_image = torch.from_numpy(pred_image).permute(2, 0, 1)
            pred_images.append(pred_image)
            
            if only_recon_hole:
                pred_image_only_hole = BCD_outputs.pred_image_only_holes[0]
                pred_image_only_hole = torch.from_numpy(pred_image_only_hole).permute(2, 0, 1)
                pred_image_only_holes.append(pred_image_only_hole)
            
        pred_images = torch.stack(pred_images).to(device)
        if only_recon_hole:
            pred_image_only_holes = torch.stack(pred_image_only_holes).to(device)
        
        if cfg.env.task == 'bcd-inpainting':
            gt_imgs = img_postprocess(input_imgs).to(device)
            hole_imgs = img_postprocess(hole_imgs).to(device)
        elif cfg.env.task == 'bcd':
            gt_imgs = img_postprocess(shape_imgs).to(device)
            input_imgs = img_postprocess(input_imgs).to(device)
            hole_imgs = None
        
        # resize to target size
        gt_imgs = resize_tensor(gt_imgs, target_shapes)
        pred_images = resize_tensor(pred_images, target_shapes)
        if cfg.env.task == 'bcd-inpainting':
            hole_imgs = resize_tensor(hole_imgs, target_shapes)
        elif cfg.env.task == 'bcd':
            input_imgs = resize_tensor(input_imgs, target_shapes)

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
        save_dir = cfg.env.test_save_dir
        
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
        
        if cfg.env.save_seq:
            # save sequence img
            for i in range(len(pred_seq)):
                img = pred_seq[i].squeeze(0)
                img = img.permute(1, 2, 0)
                img = img.cpu().numpy()
                img = (img * 255).astype('uint8')
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                if not os.path.exists(f'{cfg.env.test_save_dir}/{img_ids[0]}/seq'):
                    os.makedirs(f'{cfg.env.test_save_dir}/{img_ids[0]}/seq')
                cv2.imwrite(f'{cfg.env.test_save_dir}/{img_ids[0]}/seq/{i}.png', img)

        if track_img_count < track_img_num:
            # TODO: wandb tracker
            pass
            # for tracker in accelerator.trackers:
            #     if tracker.name == "tensorboard":
            #         np_images = np.stack([np.asarray(img) for img in pred_images])
            #         tracker.writer.add_images("validation", np_images, step, dataformats="NHWC")
            #     elif tracker.name == "wandb":
            #         wandb_log(tracker,
            #                   input_image=img_dict['input_img'],
            #                   pred_image=img_dict['pred_hole'] if only_recon_hole else img_dict['pred'],
            #                   gt_image=img_dict['gt'],
            #                   img_id=last_id,
            #                   step=step)
            #     else:
            #         logger.warning(f"image logging not implemented for {tracker.name}")
                
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
    print(f"Test on {len(test_dataloader)} images")
    print(f"Test Avg MSE: {avg_mse:.4f}")
    print(f"Test Avg SSIM: {avg_ssim:.4f}")
    print(f"Test Avg PSNR: {avg_psnr:.4f}")

    del pipeline
    torch.cuda.empty_cache()

    print("Test finished!")

if __name__ == "__main__":
    main()
