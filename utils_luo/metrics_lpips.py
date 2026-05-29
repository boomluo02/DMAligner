import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from argparse import ArgumentParser

import lpips

# device = torch.device("cuda:0")
device = torch.device("cpu")

def evaluate(result_root, gt_root, data_name):

    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device) # closer to "traditional" perceptual loss, when used for optimization

    print("")

    # result_dir = os.path.join(result_root, data_name)
    # gt_dir = os.path.join(gt_root, data_name)
    result_dir = result_root
    gt_dir = gt_root

    results = []
    meritcs_list = []

    dir_list = sorted(os.listdir(result_dir))

    for i, scene_dir in enumerate(dir_list):

        print("Scene:", scene_dir)

        renders_path = os.path.join(result_dir, scene_dir, "{:s}_pred.png".format(scene_dir))
        gt_path = os.path.join(gt_dir, scene_dir, "img1.png")

        gt = tf.to_tensor(Image.open(gt_path)).unsqueeze(0)[:, :3, :, :].to(device)
        pred = tf.to_tensor(Image.open(renders_path)).unsqueeze(0)[:, :3, :, :].to(device)
        distance = loss_fn_vgg(pred, gt) # The model takes an RGB image from [0, 1], size batch_sizex3x224x224
        meritcs = torch.tensor(distance).mean()

        print("s_name:{:s}, lpips: {:>12.7f}".format(scene_dir, meritcs))
        print("")

        results.append({
            "s_name": scene_dir,
            "lpips": float(meritcs),
        })
        meritcs_list.append(meritcs)
    
    results.append({"s_name": "mean", 
                    "lpips": torch.tensor(meritcs_list).mean()})

    csv_path = f"/root/Diff/inference_results/{data_name:s}_metrics_lpips.csv"
    import csv
    # 写入CSV文件
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = ["s_name", "lpips"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nCompleted! Results saved to {csv_path}")


if __name__ == "__main__":
    # device = torch.device("cuda:3")
    # torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    # parser.add_argument('--result_root', '-r', type=str, default="/root/Diff/output_DAVIS/inference/inference-20250218124718/test_output")
    # parser.add_argument('--gt_root', '-g', type=str, default="/root/Diff/data/DAVIS_makeup")
    # parser.add_argument('--dataset', '-d', type=str, default="DAVIS_makeup")
    # parser.add_argument('--result_root', '-r', type=str, default="/root/Diff/output_Sintel/clean/inference/inference-20250220060745/test_output")
    # parser.add_argument('--gt_root', '-g', type=str, default="/root/Diff/data/Sintel/test/clean_makeup")
    # parser.add_argument('--dataset', '-d', type=str, default="Sintel_clean_makeup")
    # parser.add_argument('--result_root', '-r', type=str, default="/root/Diff/output_Sintel/final/inference/inference-20250220070641/test_output")
    # parser.add_argument('--gt_root', '-g', type=str, default="/root/Diff/data/Sintel/test/final_makeup")
    # parser.add_argument('--dataset', '-d', type=str, default="Sintel_final_makeup")
    # parser.add_argument('--result_root', '-r', type=str, default="/root/Diff/output_Sintel_train/clean/inference/inference-20250301171528/test_output")
    # parser.add_argument('--gt_root', '-g', type=str, default="/root/Diff/data/Sintel/train/clean_makeup")
    # parser.add_argument('--dataset', '-d', type=str, default="Sintel_train_clean_makeup")
    parser.add_argument('--result_root', '-r', type=str, default="/root/Diff/output_Sintel_train/final/inference/inference-20250301171653/test_output")
    parser.add_argument('--gt_root', '-g', type=str, default="/root/Diff/data/Sintel/train/final_makeup")
    parser.add_argument('--dataset', '-d', type=str, default="Sintel_train_final_makeup")
    args = parser.parse_args()
    evaluate(args.result_root, args.gt_root, args.dataset)