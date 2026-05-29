'''
include all loss functions used in model
'''
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    PNDMScheduler,
)
from scipy.signal import gaussian

def get_pred_img(cfg,
                pred, 
                noise_scheduler, 
                vae,
                bsz,
                timesteps,
                noisy_latents):
    noise_scheduler.set_timesteps(cfg.train.val_sample_steps)
    if isinstance(noise_scheduler, (DDIMScheduler, DDPMScheduler)):
        # if scheduler is ddim, we can directly sample x0 from the predicted noise
        pred_x0 = []
        for i in range(bsz):
            pred_x0_latent = noise_scheduler.step(pred[i:i+1], timesteps[i:i+1], noisy_latents[i:i+1]).pred_original_sample
            pred_x0.append(pred_x0_latent)
        pred_x0 = torch.cat(pred_x0, dim=0)
        if vae is not None:
            pred_x0 = 1 / vae.config.scaling_factor * pred_x0
            pred_x0 = vae.decode(pred_x0).sample
        
        pred_x0 = pred_x0 * 0.5 + 0.5
        pred_x0 = pred_x0.clamp(0, 1)
    else:
        raise TypeError("Unsupported noise scheduler type: {}".format(type(noise_scheduler)))

    return pred_x0 # [B, 3, H, W]

def calc_recon_loss(cfg,
                    pred_imgs, 
                    input_imgs,
                    hole_mask_imgs,
                    ):
    '''
    calculate reconstruction loss
    '''
    recon_x0_hole = pred_imgs * hole_mask_imgs
    gt_x0_hole = input_imgs * hole_mask_imgs
    recon_loss = F.mse_loss(recon_x0_hole, gt_x0_hole, reduction="sum") / hole_mask_imgs.sum()
    recon_loss = cfg.loss.recon_loss_weight * recon_loss
    return recon_loss
    
def calc_texture_loss(cfg, pred_imgs, gt_imgs):
    '''
    calculate texture loss
    '''
    texture_loss = TextureLoss(pred_imgs.device)(pred_imgs, gt_imgs, coeff=cfg.loss.texture_loss_weight)
    return texture_loss
    
class CannyFilter(nn.Module):
    '''
    Only Use Canny Edge Detection Algorithm to find fine details, so no need to do double threshold detection and Hysteresis operations.

    Args:
        gray_mode (bool): whether to convert the input image to grayscale or not. Default: False
        kernel_size (int): size of the Gaussian kernel used for smoothing. Default: 5
        std (float): standard deviation of the Gaussian kernel. Default: 1.0
    '''

    def __init__(self, gray_mode=False, kernel_size=5, std=1.):
        super(CannyFilter, self).__init__()
        self.gray_mode = gray_mode
        self.gaussian_filter_h, self.gaussian_filter_v = self.get_gaussian_filter(kernel_size=kernel_size, std=std)
        self.sobel_x_filter, self.sobel_y_filter = self.get_sobel_filter(gray_mode=gray_mode)
        self.directional_filter = self.get_directional_filters()
    
    def forward(self, image):
        if self.gray_mode:
            image = self.rgb_to_grayscale(image)
        canny_result = self.canny_edge_detection(image) 
        return canny_result

    def canny_edge_detection(self, image):
        # Apply Gaussian filter
        if self.gray_mode:
            blur_h = self.gaussian_filter_h(image)
            blur_img = self.gaussian_filter_v(blur_h)
        else:
            img_r = image[:,0:1]
            img_g = image[:,1:2]
            img_b = image[:,2:3]
            blur_rh = self.gaussian_filter_h(img_r)
            blur_gh = self.gaussian_filter_h(img_g)
            blur_bh = self.gaussian_filter_h(img_b)
            blur_r = self.gaussian_filter_v(blur_rh)
            blur_g = self.gaussian_filter_v(blur_gh)
            blur_b = self.gaussian_filter_v(blur_bh)
            blur_img = torch.cat([blur_r, blur_g, blur_b], dim=1)
            
        # Apply Sobel filters to get gradients
        grad_x = self.sobel_x_filter(blur_img)
        grad_y = self.sobel_y_filter(blur_img)

        # Compute magnitude and direction
        grad_mag, grad_orientation = self.get_magnitude_and_direction(grad_x, grad_y)

        # Apply directional filters to the gradient magnitude
        all_filtered = self.directional_filter(grad_mag)
        indices_positive = (grad_orientation / 45).long() % 8  # Map to 8 directions
        indices_negative = ((grad_orientation / 45) + 4).long() % 8  # Map opposite directions
        
        # Perform non-maximum suppression
        suppressed_image = self.non_maximum_suppression(all_filtered, grad_mag, indices_positive, indices_negative)

        # min max normalization
        suppressed_image = (suppressed_image - suppressed_image.min()) / (suppressed_image.max() - suppressed_image.min()) * 255.0

        return suppressed_image

    def get_directional_filters(self):
        # Define the 8 directional filters manually
        filter_0 = torch.tensor([[[0, 0, 0], 
                                  [0, 1, -1], 
                                  [0, 0, 0]]], dtype=torch.float32)
        
        filter_45 = torch.tensor([[[0, 0, 0], 
                                   [0, 1, 0], 
                                   [0, 0, -1]]], dtype=torch.float32)
        
        filter_90 = torch.tensor([[[0, 0, 0], 
                                   [0, 1, 0], 
                                   [0, -1, 0]]], dtype=torch.float32)
        
        filter_135 = torch.tensor([[[0, 0, 0], 
                                    [0, 1, 0], 
                                    [-1, 0, 0]]], dtype=torch.float32)
        
        filter_180 = torch.tensor([[[0, 0, 0], 
                                    [-1, 1, 0], 
                                    [0, 0, 0]]], dtype=torch.float32)
        
        filter_225 = torch.tensor([[[-1, 0, 0], 
                                    [0, 1, 0], 
                                    [0, 0, 0]]], dtype=torch.float32)
        
        filter_270 = torch.tensor([[[0, -1, 0], 
                                    [0, 1, 0], 
                                    [0, 0, 0]]], dtype=torch.float32)
        
        filter_315 = torch.tensor([[[0, 0, -1], 
                                    [0, 1, 0], 
                                    [0, 0, 0]]], dtype=torch.float32)

        all_filters = torch.stack([filter_0, filter_45, filter_90, filter_135, filter_180, filter_225, filter_270, filter_315])

        # Create a convolution layer with 8 output channels for the 8 filters
        directional_filter = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1, bias=False)
        directional_filter.weight.data.copy_(all_filters)
        directional_filter.weight.requires_grad = False
        return directional_filter

    def non_maximum_suppression(self, all_filtered, grad_mag, indices_positive, indices_negative):
        '''
        Perform non-maximum suppression based on gradient magnitude and filtered directions.
        '''
        B, C, H, W = grad_mag.shape
        height, width = grad_mag.shape[2], grad_mag.shape[3]
        pixel_count = height * width
        pixel_range = torch.FloatTensor(range(pixel_count)).to(grad_mag.device)

        # Positive direction
        indices = (indices_positive.view(B, -1).data * pixel_count + pixel_range)
        channel_select_filtered_positive = [all_filtered.view(-1)[indices[i].long()].view(1,height,width) for i in range(B)]
        channel_select_filtered_positive = torch.stack(channel_select_filtered_positive, dim=0)

        # Negative direction
        indices = (indices_negative.view(B, -1).data * pixel_count + pixel_range)
        channel_select_filtered_negative = [all_filtered.view(-1)[indices[i].long() ].view(1,height,width) for i in range(B)]
        channel_select_filtered_negative = torch.stack(channel_select_filtered_negative, dim=0)

        channel_select_filtered = torch.stack([channel_select_filtered_positive, channel_select_filtered_negative], dim=1)

        is_max = channel_select_filtered.min(dim=1)[0] > 0.0

        thin_edges = grad_mag.clone()
        thin_edges[is_max == 0] = 0.0

        return thin_edges

    @staticmethod
    def rgb_to_grayscale(image):
        r, g, b = image[:, 0, :, :], image[:, 1, :, :], image[:, 2, :, :]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        return gray.unsqueeze(1)

    @staticmethod
    def get_magnitude_and_direction(grad_x, grad_y):
        if grad_x.shape[1] == 1:
            grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)
            grad_orientation = torch.atan2(grad_y, grad_x) * 180.0/np.pi  # angle in degrees
            grad_orientation += 180.0
            grad_orientation = torch.round(grad_orientation / 45.0) * 45.0  # normalize to 8 directions
        elif grad_x.shape[1] == 3:
            grad_mag = torch.sqrt(grad_x[:, 0, :, :] ** 2 + grad_y[:, 0, :, :] ** 2)
            grad_mag = grad_mag + torch.sqrt(grad_x[:, 1, :, :] ** 2 + grad_y[:, 1, :, :] ** 2)
            grad_mag = grad_mag + torch.sqrt(grad_x[:, 2, :, :] ** 2 + grad_y[:, 2, :, :] ** 2)
            grad_mag = grad_mag.unsqueeze(1)
            grad_orientation = torch.atan2(grad_y[:, 0, :, :] + grad_y[:, 1, :, :] + grad_y[:, 2, :, :], grad_x[:, 0, :, :] + grad_x[:, 1, :, :] + grad_x[:, 2, :, :]) * 180.0/np.pi  # angle in degrees
            grad_orientation += 180.0
            grad_orientation = torch.round(grad_orientation / 45.0) * 45.0  # normalize to 8 directions
            grad_orientation = grad_orientation.unsqueeze(1)
        return grad_mag, grad_orientation
    
    @ staticmethod
    def get_gaussian_filter(kernel_size=5, std=1.0):
        kernel_h = torch.tensor(gaussian(kernel_size,std=std).reshape([1,kernel_size]), dtype=torch.float32)
        kernel_v = torch.tensor(gaussian(kernel_size,std=std).reshape([kernel_size,1]), dtype=torch.float32)

        filter_h = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, kernel_size), padding=(0, kernel_size//2), bias=False)
        filter_v = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(kernel_size, 1), padding=(kernel_size//2, 0), bias=False)
        filter_h.weight.data = kernel_h.unsqueeze(0).unsqueeze(0)
        filter_v.weight.data = kernel_v.unsqueeze(0).unsqueeze(0)
        
        return filter_h, filter_v
    
    @staticmethod
    def get_sobel_filter(gray_mode=False):
        sobel_x = torch.tensor([[1, 0, -1], 
                                [2, 0, -2], 
                                [1, 0, -1]], dtype=torch.float32)
        
        sobel_y = torch.tensor([[1, 2, 1], 
                                [0, 0, 0], 
                                [-1, -2, -1]], dtype=torch.float32)
        
        if gray_mode:
            sobel_x_filter = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
            sobel_y_filter = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
            sobel_x_filter.weight.data = sobel_x.unsqueeze(0).unsqueeze(0)
            sobel_y_filter.weight.data = sobel_y.unsqueeze(0).unsqueeze(0)
        else:
            sobel_x_filter = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, bias=False)
            sobel_y_filter = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, bias=False)
            sobel_x_filter.weight.data = sobel_x.unsqueeze(0).unsqueeze(0).repeat(3, 3, 1, 1)
            sobel_y_filter.weight.data = sobel_y.unsqueeze(0).unsqueeze(0).repeat(3, 3, 1, 1)
        return sobel_x_filter, sobel_y_filter

class TextureLoss(nn.Module):
    """Calculate the texture loss between the predicted and ground truth tensors.

    Args:
        pred (tensor): [B, C, H, W]
        gt (tensor): [B, C, H, W]
        weight (tensor, optional): [B, C, H, W]. Defaults to None.  
        coeff (int, optional): Coefficient for the loss. Defaults to 1.
        return_img (bool, optional): Whether to return the canny images. Defaults to False.

    Returns:
        mean_loss (tensor): mean texture loss
        pred_canny_img (tensor): canny image of predicted tensor
        gt_canny_img (tensor): canny image of ground truth tensor
    """
    def __init__(self, device='cuda'):
        super(TextureLoss, self).__init__()
        self.filter = CannyFilter().to(device)
        self.l1_loss = torch.nn.L1Loss()
    def forward(self, pred, gt, weight=None, coeff=1.0):
        pred_canny = self.filter(pred)
        gt_canny = self.filter(gt)
        # save for visualization
        # gt_canny_img = gt_canny.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
        # pred_canny_img = pred_canny.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
        # cv2.imwrite('test_img/render_gt_canny.png', gt_canny_img)
        # cv2.imwrite('test_img/render_pred_canny.png', pred_canny_img)

        loss = self.l1_loss(pred_canny, gt_canny)
        if weight is not None:
            loss = loss * weight.squeeze(1)
        return loss.mean() * coeff

class L2Loss(nn.Module):
    """Calculate the L2 loss between the predicted and ground truth tensors.

    Args:
        pred (tensor): [B, C, H, W]
        gt (tensor): [B, C, H, W]
        weight (tensor, optional): [B, C, H, W]. Defaults to None.
        coeff (int, optional): Coefficient for the loss. Defaults to 1.

    Returns:
        mean_loss: mean L2 loss
    """
    def __init__(self):
        super(L2Loss, self).__init__()
        self.l2_loss = nn.MSELoss()
    
    def forward(self, pred, gt, weight=None, coeff=1.0):
        pred_ = pred.clone()
        gt_ = gt.clone()
        if weight is not None:
            pred_ *= weight
            gt_ *= weight
        loss = self.l2_loss(pred_, gt_)
        return loss * coeff

class ReconLoss(nn.Module):
    """Calculate the Fidelity or Structural loss between the predicted and ground truth tensors.

    Args:
        pred (tensor): [B, C, H, W]
        gt (tensor): [B, C, H, W]
        weight (tensor, optional): [B, C, H, W]. Defaults to None.
        coeff (int, optional): Coefficient for the loss. Defaults to 5.

    Returns:
        mean_loss: mean fidelity or structural loss
    """
    def __init__(self, mode='fidelity'):
        super(ReconLoss, self).__init__()
        # mode: 'fidelity' or 'structural'
        self.mode = mode
        self.l2_loss = nn.MSELoss(reduction='none')
    
    def forward(self, pred, gt, coeff=1.0):
        B, C, H, W = pred.size()
        sobel_x = nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1, bias=False)
        sobel_y = nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1, bias=False)
        sobel_x.weight.requires_grad = False
        sobel_y.weight.requires_grad = False

        # Sobel kernel
        sobel_x_weight = torch.tensor([[[[1, 0, -1], 
                                        [2, 0, -2], 
                                        [1, 0, -1]]]], dtype=torch.float32)

        sobel_y_weight = torch.tensor([[[[1, 2, 1], 
                                        [0, 0, 0], 
                                        [-1, -2, -1]]]], dtype=torch.float32)

        
        sobel_x.weight.data = sobel_x_weight.repeat(C, C, 1, 1).to(pred.device)
        sobel_y.weight.data = sobel_y_weight.repeat(C, C, 1, 1).to(pred.device)

        img_stack = torch.cat((pred, gt), dim=0)
        pred_x, gt_x = sobel_x(img_stack[:B]), sobel_x(img_stack[B:])
        pred_y, gt_y = sobel_y(img_stack[:B]), sobel_y(img_stack[B:])
        L2X = self.l2_loss(pred_x, gt_x)
        L2Y = self.l2_loss(pred_y, gt_y)
        L2_sqrt = torch.sqrt(L2X + L2Y)
        # map to [0, 1]
        sobel_weight = torch.tanh(L2_sqrt)
        if self.mode == 'fidelity':
            weight = 1 - sobel_weight
        elif self.mode == 'structural':
            weight = sobel_weight
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        # pixel difference
        pixel_diff = self.l2_loss(pred, gt)
        loss = (weight * pixel_diff).mean()
        return loss * coeff


class SobelLoss(nn.Module):
    """Calculate the SobelLoss loss between the predicted and ground truth tensors.

    Args:
        pred (tensor): [B, C, H, W]
        gt (tensor): [B, C, H, W]
        weight (tensor, optional): [B, C, H, W]. Defaults to None.
        coeff (int, optional): Coefficient for the loss. Defaults to 5.

    Returns:
        mean_loss: mean fidelity or structural loss
    """
    def __init__(self, mode='fidelity'):
        super(SobelLoss, self).__init__()
        # mode: 'fidelity' or 'structural'
        self.mode = mode
        self.l2_loss = nn.MSELoss()
    
    def forward(self, pred, gt, coeff=1.0):
        B, C, H, W = pred.size()
        sobel_x = nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1, bias=False)
        sobel_y = nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1, bias=False)
        sobel_x.weight.requires_grad = False
        sobel_y.weight.requires_grad = False

        # Sobel kernel
        sobel_x_weight = torch.tensor([[[[1, 0, -1], 
                                        [2, 0, -2], 
                                        [1, 0, -1]]]], dtype=torch.float32)

        sobel_y_weight = torch.tensor([[[[1, 2, 1], 
                                        [0, 0, 0], 
                                        [-1, -2, -1]]]], dtype=torch.float32)

        
        sobel_x.weight.data = sobel_x_weight.repeat(C, C, 1, 1).to(pred.device)
        sobel_y.weight.data = sobel_y_weight.repeat(C, C, 1, 1).to(pred.device)

        img_stack = torch.cat((pred, gt), dim=0)
        pred_x, gt_x = sobel_x(img_stack[:B]), sobel_x(img_stack[B:])
        pred_y, gt_y = sobel_y(img_stack[:B]), sobel_y(img_stack[B:])
        L2X = self.l2_loss(pred_x, gt_x)
        L2Y = self.l2_loss(pred_y, gt_y)
        total_loss = L2X + L2Y
        return total_loss * coeff

class CosineLoss(nn.Module):
    """Calculate the cosine loss between the predicted and ground truth tensors.

    Args:
        pred (tensor): [B, C, H, W]
        gt (tensor): [B, C, H, W]
        weight (tensor, optional): [B, C, H, W]. Defaults to None.
        coeff (int, optional): Coefficient for the loss. Defaults to 1.

    Returns:
        mean_loss: mean cosine loss
    """
    def __init__(self):
        super(CosineLoss, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1) # calc along channel dim
    
    def forward(self, pred, gt, weight=None, coeff=1.0):
        pred_ = pred.clone()
        gt_ = gt.clone()
        loss = 1 - self.cos(pred_, gt_)
        if weight is not None:
            loss = loss * weight.squeeze(1)
        return loss.mean() * coeff
