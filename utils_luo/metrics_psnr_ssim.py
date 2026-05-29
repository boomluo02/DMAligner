import os
import cv2
import csv
import numpy as np
from calculate_psnr import calculate

def main():
    # 定义路径模板
    gen_base_dir = "/root/Diff/output_DSIA/inference/inference-20250218101419/test_output"
    gt_base_dir = "/root/Diff/data/DSIA/test"
    csv_path = "/root/Diff/output_DSIA/metrics_results.csv"

    # 获取所有s_name（生成目录中的子目录）
    s_names = []
    for name in os.listdir(gen_base_dir):
        gen_dir = os.path.join(gen_base_dir, name)
        gt_dir = os.path.join(gt_base_dir, name)
        
        if os.path.isdir(gen_dir) and os.path.isdir(gt_dir):
            s_names.append(name)
        else:
            print(f"Skipping {name}: directories not found")

    results = []
    idx = 0
    for s_name in s_names:
        # 构建完整文件路径
        gen_path = os.path.join(gen_base_dir, s_name, f"{s_name}_pred.jpg")
        gt_path = os.path.join(gt_base_dir, s_name, "img1_warp_gt.png")

        # 检查文件是否存在
        if not os.path.exists(gen_path):
            print(f"Missing generated image: {gen_path}")
            continue
        if not os.path.exists(gt_path):
            print(f"Missing ground truth: {gt_path}")
            continue

        # 读取并预处理图像
        try:
            img_gt = cv2.imread(gt_path).astype(np.float32) / 255.0
            img_gen = cv2.imread(gen_path).astype(np.float32) / 255.0
        except Exception as e:
            print(f"Error loading {s_name}: {str(e)}")
            continue

        # 验证图像尺寸
        if img_gt.shape != img_gen.shape:
            print(f"Shape mismatch in {s_name}: GT {img_gt.shape} vs Gen {img_gen.shape}")
            continue
        
        # 计算指标
        try:
            psnr, ssim = calculate(img_gt, img_gen)
            results.append({
                "s_name": s_name,
                "PSNR": float(psnr),
                "SSIM": float(ssim)
            })
            idx += 1
            print(f"Processed {idx:d}/{len(s_names):d}, name {s_name}: PSNR={psnr:.2f}, SSIM={ssim:.4f}")
        except Exception as e:
            print(f"Error calculating metrics for {s_name}: {str(e)}")
            continue

    # 写入CSV文件
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = ["s_name", "PSNR", "SSIM"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nCompleted! Results saved to {csv_path}")

if __name__ == "__main__":
    main()