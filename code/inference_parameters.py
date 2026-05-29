import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from tools.init_1 import initialize_test
from tools.utils_1 import compute_criterions, img_postprocess, log_validation_ldm, read_data, resize_tensor, save_model_card, save_model_output, test_vae, unwrap_model # noqa

import cv2

def main():
    # =========== initialize ===========
    cfg, logger, pipeline, image_encoder,\
    _, infer_dataloader, use_rnd, device = initialize_test()

    if cfg.env.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=device).manual_seed(cfg.env.seed)

    # save num
    track_img_num = 6
    track_img_count = 0

    for i, batch in enumerate(infer_dataloader):
        pred_images = []
        pred_image_only_holes = []
        # read data
        file_names, img1s, img2s, clip_images, target_shapes = read_data(batch, device, phase='inference')

        if cfg.data.change_infer_size:
            # resize
            origin_size_h, origin_size_w = img1s.shape[-2], img1s.shape[-1]
            # target_shapes = (origin_size_h, origin_size_w)
            img1s = F.interpolate(img1s, size=(cfg.data.infer_h, cfg.data.infer_w), mode='bilinear', align_corners=True)
            img2s = F.interpolate(img2s, size=(cfg.data.infer_h, cfg.data.infer_w), mode='bilinear', align_corners=True)

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

        with torch.no_grad():
            
            from thop import profile
            args = {"img1":img1s, "img2":img2s,
                    "num_inference_steps": cfg.model.num_inference_steps,
                    "image_embeds": image_embeds,
                    "generator": generator,
                    "output_type": 'ndarray',
                    "return_seq": True}
            
            flops, params = profile(pipeline, inputs=args, verbose=False)
            print('flops: %.3f G, params: %.3f M' % (flops / 1000 / 1000 / 1000, params / 1000 / 1000))

            # print(f"Inferencing [{i+1}/{len(infer_dataloader)}] image [{file_names[0]}]:")
            # # img1s = img1s.unsqueeze(0).to(device)
            # # img2s = img2s.unsqueeze(0).to(device)
            # BCD_outputs, pred_seq = pipeline(img1s, img2s,
            #                         num_inference_steps=cfg.model.num_inference_steps, 
            #                         image_embeds=image_embeds,
            #                         generator=generator,
            #                         output_type='ndarray',
            #                         return_seq=True)
        
            
        # pred_image = BCD_outputs.pred_images[0]
        # pred_image = torch.from_numpy(pred_image).permute(2, 0, 1)
        # pred_images.append(pred_image)
            
        # pred_images = torch.stack(pred_images).to(device)
        # if only_recon_hole:
        #     pred_image_only_holes = torch.stack(pred_image_only_holes).to(device)
        
        # input_imgs = img_postprocess(img1s).to(device)
        
        # # resize to target size
        # # input_imgs = resize_tensor(input_imgs, target_shapes)
        # # pred_images = resize_tensor(pred_images, target_shapes)
        # input_imgs = F.interpolate(input_imgs, size=(origin_size_h, origin_size_w), mode='bilinear', align_corners=True)
        # pred_images = F.interpolate(pred_images, size=(origin_size_h, origin_size_w), mode='bilinear', align_corners=True)
    
        # # save img
        # save_dir = cfg.env.test_save_dir
        
        # last_id, img_dict = save_model_output(file_names, 
        #                                     save_dir,
        #                                     input_img=input_imgs,
        #                                     pred=pred_images)
        
        # if cfg.env.save_seq:
        #     # save sequence img
        #     for i in range(len(pred_seq)):
        #         img = pred_seq[i].squeeze(0)
        #         img = img.permute(1, 2, 0)
        #         img = img.cpu().numpy()
        #         img = (img * 255).astype('uint8')
        #         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #         if not os.path.exists(f'{cfg.env.test_save_dir}/{file_names[0]}/seq'):
        #             os.makedirs(f'{cfg.env.test_save_dir}/{file_names[0]}/seq')
        #         cv2.imwrite(f'{cfg.env.test_save_dir}/{file_names[0]}/seq/{i}.png', img)

        # if track_img_count < track_img_num:
        #     # TODO: wandb tracker
        #     pass
        #     # for tracker in accelerator.trackers:
        #     #     if tracker.name == "tensorboard":
        #     #         np_images = np.stack([np.asarray(img) for img in pred_images])
        #     #         tracker.writer.add_images("validation", np_images, step, dataformats="NHWC")
        #     #     elif tracker.name == "wandb":
        #     #         wandb_log(tracker,
        #     #                   input_image=img_dict['input_img'],
        #     #                   pred_image=img_dict['pred_hole'] if only_recon_hole else img_dict['pred'],
        #     #                   gt_image=img_dict['gt'],
        #     #                   img_id=last_id,
        #     #                   step=step)
        #     #     else:
        #     #         logger.warning(f"image logging not implemented for {tracker.name}")
                
        #     track_img_count += 1

    del pipeline
    torch.cuda.empty_cache()

    print("Inferencing finished!")

if __name__ == "__main__":
    main()
