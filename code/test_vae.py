import os

from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
import torch.nn.functional as F
import pandas as pd
import cv2

from tools.init_1 import initialize_vae
from tools.utils_1 import compute_criterions, img_postprocess, log_validation_ldm, log_validation_vae, read_data, save_model_card, save_model_output, test_vae  # noqa: F401

def main():
    # =========== initialize ===========
    cfg, vae, test_dataloader, device = initialize_vae(stage='test')

    avg_mse = 0
    avg_psnr = 0
    avg_ssim = 0
    mse_list = []
    psnr_list = []
    ssim_list = []

    # metrices file
    metrices_df = pd.DataFrame(columns=['img_id', 'MSE', 'SSIM', 'PSNR'])

    for batch in tqdm(test_dataloader):
        with torch.no_grad():
            # read data
            input_imgs, shape_imgs, mask_imgs, hole_masks, \
            hole_imgs, prompts, clip_images, target_shapes, img_ids = read_data(batch, device, torch.float32, phase='test')

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
            
            # # save the model prediction for visualization
            # model_pred_ = z[0] * 0.5 + 0.5
            # model_pred_ = model_pred_.clamp(0, 1)
            # # resize [0, 1]
            # # model_pred_ = (model_pred_ - model_pred_.min()) / (model_pred_.max() - model_pred_.min())
            # for channel in range(model_pred_.shape[0]):
            #     channel_map = model_pred_[channel, :, :]
            #     channel_map = channel_map.unsqueeze(0)
            #     channel_map = channel_map.permute(1, 2, 0).detach().cpu().numpy()
            #     channel_map = channel_map * 255
            #     channel_map = channel_map.astype('uint8')
            #     if not os.path.exists('test_img'):
            #         os.makedirs('test_img')
            #     cv2.imwrite(f'test_img/{channel}.png', channel_map)
             

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
        if cfg.env.test_save_dir:
            save_dir = cfg.env.test_save_dir
            last_id, img_dict = save_model_output(img_ids, 
                                                save_dir,
                                                gt=input_imgs,
                                                pred=model_preds,)
                
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
    print(f"Avg MSE: {avg_mse:.4f}")
    print(f"Avg SSIM: {avg_ssim:.4f}")
    print(f"Avg PSNR: {avg_psnr:.4f}")

    print("Test finished!")


if __name__ == "__main__":
    main()
