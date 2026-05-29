import os
# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import cv2

import random
import pandas as pd

from tqdm import tqdm

from tools.utils import generate_hole_mask


def generate_csv_file_for_align(data_root, create_debug=False):
    '''
    This method is used to generate a csv file with two fields: input and gt.
    Please configure the dataset in the following structure:
        data_root
        ├── train
        │   ├── 00000_1_0001
        │   │   ├── img1_warp_gt.png
        │   │   ├── img1.png
        │   │   ├── img2.png
        │   │   └── input_mask.png
        │   ├── ...
        ├── test
        │   ├── SL_00001_2_0006
        │   │   ├── img1_warp_gt.png
        │   │   ├── img1.png
        │   │   ├── img2.png
        │   │   └── input_mask.png

    :param data_root: Root directory of the dataset
    :param create_debug: Whether to create debug csv files
    :param train_val_split_ratio: Ratio of training and validation split
    '''

    train_df = pd.DataFrame(columns=['img1',
                                     'img2',
                                     'gt',
                                     'mask',
                                     ])
    
    test_df = pd.DataFrame(columns=['img1',
                                    'img2',
                                    'gt',
                                    'mask',
                                    ])

    data_type = ['train', 'test']
    for dt in data_type:
        input_dir = os.path.join(data_root, dt)
        file_dirs = os.listdir(input_dir)
        random.shuffle(file_dirs)
        print('Processing data ...')
        for file_dir in tqdm(file_dirs):
            img1 = os.path.join(input_dir, file_dir, 'img1.png')
            img2 = os.path.join(input_dir, file_dir, 'img2.png')
            gt = os.path.join(input_dir, file_dir, 'img1_warp_gt.png')
            mask = os.path.join(input_dir, file_dir, 'input_mask.png')

            # check file exist
            for f in [img1, img2, gt, mask]:
                if not os.path.exists(f):
                    raise FileNotFoundError(f"File {f} not found in {input_dir}")

            if dt == 'train':
                train_df = pd.concat([train_df, pd.DataFrame({'img1': [img1],
                                                              'img2': [img2],
                                                              'gt': [gt],
                                                              'mask': [mask],
                                                              })], ignore_index=True)
            else:
                test_df = pd.concat([test_df, pd.DataFrame({'img1': [img1],
                                                            'img2': [img2],
                                                            'gt': [gt],
                                                            'mask': [mask],
                                                            })], ignore_index=True)
                
    print("Train data length:", len(train_df))
    print("Test data length:", len(test_df))

    if not os.path.exists('data/csv_files'):
        os.makedirs('data/csv_files')
        
    # Check CSV files and generate (keep a copy of the previous one)
    if os.path.exists('data/csv_files/train.csv'):
        os.remove('data/csv_files/train_old.csv') if os.path.exists('data/csv_files/train_old.csv') else None
        os.rename('data/csv_files/train.csv', 'data/csv_files/train_old.csv')

    if os.path.exists('data/csv_files/test.csv'):
        os.remove('data/csv_files/test_old.csv') if os.path.exists('data/csv_files/test_old.csv') else None
        os.rename('data/csv_files/test.csv', 'data/csv_files/test_old.csv')

    train_df.to_csv('data/csv_files/train.csv', index=False)
    test_df.to_csv('data/csv_files/test.csv', index=False)

    print("CSV files generated successfully!")
    if create_debug:
        # random select 15 samples from train_df, val_df and test_df for each to create debug csv files
        train_debug_df = train_df.sample(n=15)
        test_debug_df = test_df.sample(n=3)

        if os.path.exists('data/csv_files/train_debug.csv'):
            os.remove('data/csv_files/train_debug.csv')
        if os.path.exists('data/csv_files/test_debug.csv'):
            os.remove('data/csv_files/test_debug.csv')
        
        train_debug_df.to_csv('data/csv_files/train_debug.csv', index=False)
        test_debug_df.to_csv('data/csv_files/test_debug.csv', index=False)
        
        print("Train debug data length:", len(train_debug_df))
        print("Test debug data length:", len(test_debug_df))
        print("Debug CSV files generated successfully!")

def generate_render_csv_file(data_root, create_debug=False, train_val_split_ratio=0.98):
    '''
    This method is used to generate a csv file with two fields: input and gt.
    Please configure the dataset in the following structure:
        data_root
        ├── 2_humanpose1_00005_norm.png
        ├── 2_humanpose1_00005_wideangle.jpg
        ├── ...

    :param data_root: Root directory of the dataset
    :param create_debug: Whether to create debug csv files
    :param train_val_split_ratio: Ratio of training and validation split
    '''
    train_df = pd.DataFrame(columns=['input',
                                     'shape',
                                     'people_num',
                                     ])
    
    test_df = pd.DataFrame(columns=['input',
                                    'shape',
                                     'people_num',
                                     ])
    
    input_files = os.listdir(data_root)
    
    random.shuffle(input_files)
    print('Processing data ...')
    people_dict = {}
    for file in tqdm(input_files):
        if file.endswith('wideangle.png'):
            people_num = file.split('_')[0]
            people_dict[people_num] = people_dict.get(people_num, 0) + 1
            
            input_file = data_root + '/' +  file
            shape = data_root + '/' +  file.replace('wideangle.png', 'norm.png')
            
            # check file exist
            for f in [input_file, shape]:
                if not os.path.exists(f):
                    raise FileNotFoundError(f"File {f} not found in {data_root}")
                
            train_df = pd.concat([train_df, pd.DataFrame({'input': [input_file],
                                                          'shape': [shape],
                                                          'people_num': [people_num],
                                                          })], ignore_index=True)              
        
    # Split according to people_num
    for people_num in people_dict.keys():
        people_num_df = train_df[train_df['people_num'] == people_num]
        train_len = int(len(people_num_df)*train_val_split_ratio)
        train_df = train_df.drop(people_num_df.index)
        train_df = pd.concat([train_df, people_num_df[:train_len]])
        test_df = pd.concat([test_df, people_num_df[train_len:]])

    print("Train data length:", len(train_df))
    print("Test data length:", len(test_df))

    if not os.path.exists('data/csv_files'):
        os.makedirs('data/csv_files')
        
    # Check CSV files and generate (keep a copy of the previous one)
    if os.path.exists('data/csv_files/train_render.csv'):
        os.remove('data/csv_files/train_render_old.csv') if os.path.exists('data/csv_files/train_render_old.csv') else None
        os.rename('data/csv_files/train_render.csv', 'data/csv_files/train_render_old.csv')

    if os.path.exists('data/csv_files/test_render.csv'):
        os.remove('data/csv_files/test_render_old.csv') if os.path.exists('data/csv_files/test_render_old.csv') else None
        os.rename('data/csv_files/test_render.csv', 'data/csv_files/test_render_old.csv')

    train_df.to_csv('data/csv_files/train_render.csv', index=False)
    test_df.to_csv('data/csv_files/test_render.csv', index=False)

    print("CSV files generated successfully!")
    if create_debug:
        # random select 15 samples from train_df, val_df and test_df for each to create debug csv files
        train_debug_df = train_df.sample(n=15)
        test_debug_df = test_df.sample(n=3)

        if os.path.exists('data/csv_files/train_render_debug.csv'):
            os.remove('data/csv_files/train_render_debug.csv')
        if os.path.exists('data/csv_files/test_render_debug.csv'):
            os.remove('data/csv_files/test_render_debug.csv')
        
        train_debug_df.to_csv('data/csv_files/train_render_debug.csv', index=False)
        test_debug_df.to_csv('data/csv_files/test_render_debug.csv', index=False)
        
        print("Train debug data length:", len(train_debug_df))
        print("Test debug data length:", len(test_debug_df))
        print("Debug CSV files generated successfully!")

def rename_file(data_root):
    '''
    00005_Xxxx_2_xxx_norm.png -> 2_xxx_00005_norm.png
    00005_Xxxx_2_xxx_wideangle.png -> 2_xxx_00005_wideangle.png
    '''
    input_files = os.listdir(data_root)
    for file in tqdm(input_files):
        if file.endswith('norm.png') or file.endswith('wideangle.png'):
            name_split = file.split('_')
            if len(name_split) == 5:
                new_name = name_split[2] + '_' + name_split[3] + '_' + name_split[0] + '_' + name_split[4]
                os.rename(os.path.join(data_root, file), os.path.join(data_root, new_name))
    
if __name__ == '__main__':
    # data_root = 'data/img_data'
    # generate_csv_file(data_root, create_debug=True)

    # data_root = 'data/img_render_v1/dataset'
    # generate_render_csv_file(data_root, create_debug=True)

    # data_root = 'data/img_render_v3'
    # # rename_file(data_root)
    # generate_render_csv_file(data_root, create_debug=True)

    data_root = 'data/DSIA'
    generate_csv_file_for_align(data_root, create_debug=True)

