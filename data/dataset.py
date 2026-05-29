import os
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPImageProcessor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from tools.utils_1 import CropbyMask, make_mask


def get_transform(cfg):
    '''
    Return the transform for the dataset
    '''
    train_transform = []
    if cfg.transform.resize:
        train_transform.append(transforms.Resize((cfg.data.img_size_h, cfg.data.img_size_w)))
    if cfg.transform.change_color:
        train_transform.append(transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.2, hue=0.5))
        train_transform.append(transforms.RandomGrayscale(p=0.1))
    if cfg.transform.random_flip:
        train_transform.append(transforms.RandomHorizontalFlip(p=0.5))
    if cfg.transform.center_crop:
        train_transform.append(transforms.CenterCrop((cfg.transform.crop_size_h, cfg.transform.crop_size_w)))
    elif cfg.transform.crop_by_mask:
        train_transform.append(CropbyMask((cfg.transform.crop_size_h, cfg.transform.crop_size_w)))
    elif cfg.transform.random_crop:
        train_transform.append(transforms.RandomCrop((cfg.transform.crop_size_h, cfg.transform.crop_size_w)))

    test_transform = [
        # transforms.Resize((cfg.data.img_size_h, cfg.data.img_size_w)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]

    train_transform.extend(test_transform)

    return train_transform, test_transform

def generate_infer_csv(infer_dir):
    '''
    Generate infer csv file

    Args:
        :param infer_dir: Path to the infer directory
    '''
    infer_csv = os.path.join('data/csv_files', 'infer.csv')
    img_1_list, img_2_list = [], []
    file_list = os.listdir(infer_dir)
    if len(file_list) == 0:
        raise ValueError(f'No image files found in the infer directory [{infer_dir}].')
    for file_dir in file_list:
        for file in os.listdir(f'{infer_dir}/{file_dir}'):
            if file == 'img1.png':
                img_1_list.append(os.path.join(infer_dir, file_dir, file))
                img2 = file.replace('img1', 'img2')
                img_2_list.append(os.path.join(infer_dir, file_dir, img2))
    infer_df = pd.DataFrame({"img1":img_1_list,"img2":img_2_list})
    infer_df.to_csv(infer_csv, index=False)
    return infer_csv

def get_dataloader(cfg):
    print("-----------------------------------")
    train_csv_name = 'train.csv'
    test_csv_name = 'test.csv'
    train_debug_csv_name = 'train_debug.csv'
    test_debug_csv_name = 'test_debug.csv'

    print('Loading dataset ...')
    if cfg.env.mode != 'inference':
        csv_root = os.path.join(cfg.data.DATA_ROOT, 'csv_files')
        if cfg.env.debug:
            train_csv = os.path.join(csv_root, train_debug_csv_name)
            test_csv = os.path.join(csv_root, test_debug_csv_name)

        else:
            train_csv = os.path.join(csv_root, train_csv_name)
            test_csv = os.path.join(csv_root, test_csv_name)

        # Data augmentation
        train_transform, test_transform = get_transform(cfg)
        for_vae = True if cfg.tracker.project == 'BCD_VAE' else False

        # dataset
        train_dataset = MyDataset(train_csv, 
                                         transform=train_transform, 
                                         decoupled_attn=cfg.env.decoupled_attn,
                                         clip_name_or_path=cfg.model.clip_name_or_path)
        test_dataset = MyDataset(test_csv, 
                                        transform=test_transform, 
                                        decoupled_attn=cfg.env.decoupled_attn,
                                        clip_name_or_path=cfg.model.clip_name_or_path)

        # dataloader
        if cfg.env.mode == 'train':
            batch_size = cfg.train.batch_size
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=cfg.data.num_workers, collate_fn=collate_fn)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
            print(f'Train dataset size: {len(train_dataset)}')
            print(f'Test dataset size: {len(test_dataset)}')
        else:
            train_loader = None
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
            print(f'Test dataset size: {len(test_dataset)}')
        print("-----------------------------------") 

        return train_loader, test_loader

    elif cfg.env.mode == 'inference':
        # Data augmentation
        train_transform, test_transform = get_transform(cfg)
        for_vae = True if cfg.tracker.project == 'BCD_VAE' else False
        if not os.path.exists(cfg.data.infer_dir):
            raise ValueError(f'Inference directory [{cfg.data.infer_dir}] not exists.')
        infer_csv = generate_infer_csv(cfg.data.infer_dir)
        infer_dataset = MyDataset(infer_csv, 
                                transform=test_transform, 
                                # data_type=cfg.env.data, 
                                # for_vae=for_vae, 
                                # random_mask=False, 
                                decoupled_attn=cfg.env.decoupled_attn,
                                clip_name_or_path=cfg.model.clip_name_or_path)
        infer_loader = DataLoader(infer_dataset, batch_size=1, shuffle=False, num_workers=cfg.data.num_workers, collate_fn=infer_collate_fn)
        print(f'Infer dataset size: {len(infer_dataset)}')
        print("-----------------------------------")
        
        return None, infer_loader

class MyDataset(Dataset):
    '''
    Dataset class
    '''
    def __init__(self, csv_file, transform=None, max_num=None, seed=42, decoupled_attn=False, clip_name_or_path=None):
        '''
        Constructor

        Args:
            :param csv_file: Path to the csv file
            :param transform: Transformations to be applied to the images
            :param max_num: Maximum number of images to be loaded
            :param seed: Seed for shuffling the data
            :param for_vae: If the data is for VAE
        '''
        self.phase = csv_file.split('/')[-1].split('.')[0]
        self.decoupled_attn = decoupled_attn
        self.data = pd.read_csv(csv_file)
        
        if decoupled_attn:
            self.clip_image_processor = CLIPImageProcessor.from_pretrained(clip_name_or_path)
        else:
            self.clip_image_processor = None

        if max_num is not None:
            # shuffle data
            self.data = self.data.sample(frac=1, random_state=seed).reset_index(drop=True)
            self.data = self.data.iloc[:max_num]
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        '''
        Get item method
        '''
        if self.phase in ['train', 'train_debug', 'test', 'test_debug']:
            # img file
            img1_path = self.data['img1'].iloc[idx]
            img2_path = self.data['img2'].iloc[idx]
            gt_path = self.data['gt'].iloc[idx]
            mask_path = self.data['mask'].iloc[idx]
            img_id = f"{img1_path.split('/')[3]}"

            # read image 0~1
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')
            gt_img = Image.open(gt_path).convert('RGB')
            mask_img = Image.open(mask_path).convert('L')
        
            if self.decoupled_attn:
                clip_image = self.clip_image_processor(img1, return_tensors='pt').pixel_values
            else:
                clip_image = None

            all_list = [img1, img2, gt_img, mask_img]

            # manual seed
            seed = torch.random.seed()
            torch.manual_seed(seed)
            # addition transform
            if self.transform: 
                for t in self.transform:
                    for i in range(len(all_list)): 
                        if all_list[i] is None:
                            continue
                        if isinstance(t, transforms.Resize):
                            if i not in [3]:
                                t.interpolation = transforms.InterpolationMode.BILINEAR
                            else:
                                t.interpolation = transforms.InterpolationMode.NEAREST
                            all_list[i] = t(all_list[i])
                        elif isinstance(t, transforms.ColorJitter) or isinstance(t, transforms.RandomGrayscale) or isinstance(t, transforms.Normalize):
                            if i not in [3]: 
                                # mask don't need to change color and normalize
                                all_list[i] = t(all_list[i])
                        elif isinstance(t, CropbyMask):
                            if i not in [3]:
                                all_list[i] = t(all_list[i], mask_img)
                        else:
                            # RandomFlip, CenterCrop, ToTensor
                            all_list[i] = t(all_list[i])
            
            # prompt
            prompt = "" if random.random() < 0.1 else "a photo of sks"
            data = {
                'img1': all_list[0],
                'img2': all_list[1],
                'gt_img': all_list[2],
                'mask_img': all_list[3],
                'prompt': prompt,
                'img_id': f'train_{img_id}'
            }

            return data
        
        elif self.phase == 'infer':
            # jpg file
            img1_path = self.data['img1'].iloc[idx]
            img2_path = self.data['img2'].iloc[idx]

            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')

            # file_name
            file_name = img1_path.split('/')[-2].split('.')[0]
            
            # read image 0~1
            shape = img1.size
            if shape[0] > shape[1]: # w > h
                target_shape = (384, 512)
            else:
                target_shape = (512, 384)

            if self.decoupled_attn:
                clip_image = self.clip_image_processor(img1, return_tensors='pt').pixel_values
            else:
                clip_image = None

            all_list = [img1, img2]

            if self.transform:
                for t in self.transform:
                    for i in range(len(all_list)):
                        all_list[i] = t(all_list[i])

            data = {
                'file_name': file_name,
                'img1': all_list[0],
                'img2': all_list[1],
                'clip_image': clip_image,
                'target_shape': target_shape
            }

            return data

        else:
            raise ValueError('Invalid phase, Please check the csv file name.')


def collate_fn(batch):
    img1 = torch.stack([item['img1'] for item in batch])
    img2 = torch.stack([item['img2'] for item in batch])
    gt_img = torch.stack([item['gt_img'] for item in batch])
    mask_img = torch.stack([item['mask_img'] for item in batch])
    prompt = [item['prompt'] for item in batch]
    img_id = [item['img_id'] for item in batch]
    
    batch = {
        'img1': img1,
        'img2': img2,
        'gt_img': gt_img,
        'mask_img': mask_img,
        'prompt': prompt,
        'img_id': img_id
    }
    return batch

def infer_collate_fn(batch):

    img1 = torch.stack([item['img1'] for item in batch])
    img1 = img1.to(memory_format=torch.contiguous_format).float()
    img2 = torch.stack([item['img2'] for item in batch])
    img2 = img2.to(memory_format=torch.contiguous_format).float()

    if batch[0]['clip_image'] is not None:
        clip_image = torch.cat([item["clip_image"] for item in batch], dim=0)
    else:
        clip_image = None

    target_shape = [item['target_shape'] for item in batch]

    batch = {
        'file_name': [item['file_name'] for item in batch],
        'img1': img1,
        'img2': img2,
        'clip_image': clip_image,
        'target_shape': target_shape
    }
    return batch