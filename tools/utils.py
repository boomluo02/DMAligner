import logging
from math import exp
import os
import cv2
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
import yaml
from easydict import EasyDict as edict
from torch.nn import Conv2d

# ========== model tool =========
def replace_unet_conv_in(unet):
    # add a layer to transfer 9 channel to 8 channel
    _add_layer_in = Conv2d(
        8, 9, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)
    )
    # add layer 和 new conv in合并到一起
    two_layer = nn.Sequential(_add_layer_in, unet.conv_in)
    unet.conv_in = two_layer
    logging.info("Unet conv_in layer is replaced")
    # replace config
    unet.config["in_channels"] = 8
    logging.info("Unet config is updated")
    return unet

# ========== img tool =========
def img_postprocess(img):
    '''
    de-normalize the image

    args:
        img (torch.Tensor): input image
        scale (float): scale factor of the input image
    
    returns:
        img (torch.Tensor): de-normalized image
    '''
    img = img * 0.5 + 0.5
    img = img.clamp(0, 1)

    return img

def save_one_img(img_ids, i, save_dir, **kwargs):
    img_id = img_ids[i]
    save_dir = f'{save_dir}/{img_id}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img_dict = {}
    # gts and masks
    for arg_k, arg_v in kwargs.items():
        if arg_k == 'save_num':
            continue
        img=arg_v[i]
        # img = arg_v[i].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        save_pth = f'{save_dir}/{img_id}_{arg_k}.jpg'
        img_pil = ToPILImage()(img)
        # cv2.imwrite(save_pth, img)
        img_pil.save(save_pth)
        img_dict[arg_k] = img

    return img_id, img_dict

def save_model_output(img_ids, save_dir='debug_img/', **kwargs):
    gt = kwargs['gt']
    for i in range(gt.size(0)):
        img_id, img_dict = save_one_img(img_ids, i, save_dir, **kwargs)
    return img_id, img_dict

# ========== compute tool =========
class SSIM(nn.Module):
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

# ========== data tool =========
def read_data(data:dict, dtype=torch.float32, phase='train'):
    if phase in ['train', 'test']:
        input_img = data['input_img'].to(dtype)
        shape_img = data['shape_img'].to(dtype)
        hole_img = data['hole_img'].to(dtype)
        line_mask_img = data['line_mask_img'].to(dtype)
        hole_mask_img = data['hole_mask_img'].to(dtype)
        img_id = data['img_id']

        return input_img, shape_img, line_mask_img, hole_mask_img, hole_img, img_id
    
    elif phase == 'inference':
        file_name = data['file_name']
        input_img = data['input_img'].to(dtype)
        origin_img = data['origin_img'].to(dtype)
        return file_name, input_img, origin_img
    
    else:
        raise ValueError(f'phase [{phase}] not supported')

def generate_hole_mask(line_wholebodymask_pth, hole_mask_pth, shape):
    '''
    generate the hole mask from the line_wholebodymask

    Args:
        line_wholebodymask (str): line_wholebodymask path
        hole_mask_pth (str): hole_mask path
    '''
    line_wholebodymask = cv2.imread(line_wholebodymask_pth, cv2.IMREAD_GRAYSCALE)
    line_wholebodymask[line_wholebodymask > 0] = 1
    # resize
    line_wholebodymask = cv2.resize(line_wholebodymask, shape, interpolation=cv2.INTER_NEAREST)
    # save mask
    # cv2.imwrite("./test/test_mask.png", line_wholebodymask * 255)

    left_offset = np.random.randint(5, 15)
    right_offset = np.random.randint(5, 15)
    left_shift_mask = np.roll(line_wholebodymask, -left_offset, axis=1)
    right_shift_mask = np.roll(line_wholebodymask, right_offset, axis=1)
    uni_mask = left_shift_mask + right_shift_mask
    uni_mask = np.clip(uni_mask, 0, 1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    hole_mask_erosion = cv2.erode(line_wholebodymask, kernel, iterations=1)

    hole_mask = uni_mask - hole_mask_erosion

    cv2.imwrite(hole_mask_pth, hole_mask * 255)

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
