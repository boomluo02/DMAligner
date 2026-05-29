import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from pathlib import Path
from argparse import ArgumentParser

from lpipsPyTorch import lpips


# os.environ["CUDA_VISIBLE_DEVICES"] = "3"  #代表只使用第3个gpu
device = torch.device("cuda:3")

def readImages(renders_path, gt_path):
    renders = []
    gts = []

    render = Image.open(renders_path)
    gt = Image.open(gt_path)
    renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].to(device))
    gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].to(device))
    return renders, gts

def evaluate(output_roots):

    full_dict = {}

    print("")

    for output_root in output_roots:
        for scene_dir in os.listdir(output_root):

            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}

            gt_path = os.path.join(output_root, scene_dir, "{:s}_img1.jpg".format(scene_dir))
            renders_path = os.path.join(output_root, scene_dir, "{:s}_pred.jpg".format(scene_dir))
            renders, gts = readImages(renders_path, gt_path)
            lpipss = []

            for idx in range(len(renders)):

                lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))


            print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean()))
            print("")

            full_dict[scene_dir].update({"LPIPS": torch.tensor(lpipss).mean().item()})



if __name__ == "__main__":
    # device = torch.device("cuda:3")
    # torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--output_root', '-o', nargs="+", type=str, default=["/root/Diff/output/train/train-20241223092440/val_output"])
    args = parser.parse_args()
    evaluate(args.output_root)