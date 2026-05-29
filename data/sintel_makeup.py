import os
import shutil
import re

source_dir = '/root/Diff/data/Sintel/test/clean'  # 替换为实际路径
target_dir = '/root/Diff/data/Sintel/test/clean_makeup'

source_dir = '/root/Diff/data/Sintel/test/final'  # 替换为实际路径
target_dir = '/root/Diff/data/Sintel/test/final_makeup'

source_dir = '/root/Diff/data/Sintel/train/clean'  # 替换为实际路径
target_dir = '/root/Diff/data/Sintel/train/clean_makeup'

source_dir = '/root/Diff/data/Sintel/train/final'  # 替换为实际路径
target_dir = '/root/Diff/data/Sintel/train/final_makeup'

os.makedirs(target_dir, exist_ok=True)
frame_pattern = re.compile(r'frame_(\d{4})\.png')

for seq_name in os.listdir(source_dir):
    seq_path = os.path.join(source_dir, seq_name)
    if not os.path.isdir(seq_path):
        continue

    # 提取并排序帧文件
    frames = sorted([
        f for f in os.listdir(seq_path)
        if frame_pattern.match(f)
    ], key=lambda x: int(frame_pattern.match(x).group(1)))

    # 处理连续帧对
    for i in range(len(frames)-1):
        current_frame = frames[i]
        next_frame = frames[i+1]
        
        # 提取数字索引
        idx = int(frame_pattern.match(current_frame).group(1))
        
        # 创建目标文件夹
        new_folder = f"{seq_name}_{idx:04d}"
        new_folder_path = os.path.join(target_dir, new_folder)
        os.makedirs(new_folder_path, exist_ok=True)
        
        # 复制并重命名文件
        shutil.copy(
            os.path.join(seq_path, current_frame),
            os.path.join(new_folder_path, "img1.png")  # 当前帧 -> img1
        )
        shutil.copy(
            os.path.join(seq_path, next_frame),
            os.path.join(new_folder_path, "img2.png")  # 下一帧 -> img2
        )

print("处理完成！")