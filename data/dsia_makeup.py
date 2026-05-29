import os
import re
import shutil
from collections import defaultdict

def main():
    source_dir = '/root/Diff/data/DAVIS'
    target_root = '/root/Diff/data/DAVIS_makeup'
    file_pairs = defaultdict(dict)
    processed = 0
    errors = 0

    # 第一阶段：扫描文件
    print("🕵️ 扫描文件中...")
    for filename in os.listdir(source_dir):
        # 匹配 img1_数字.png
        img1_match = re.match(r'img1_(\d+)\.png', filename)
        if img1_match:
            idx = f"{int(img1_match.group(1)):04d}"
            file_pairs[idx]['img1'] = filename
            continue

        # 匹配 img2_数字.png
        img2_match = re.match(r'img2_(\d+)\.png', filename)
        if img2_match:
            idx = f"{int(img2_match.group(1)):04d}"
            file_pairs[idx]['img2'] = filename

    # 第二阶段：处理文件
    print(f"🔍 找到 {len(file_pairs)} 个潜在配对")
    for idx, files in file_pairs.items():
        try:
            # 验证配对完整性
            if 'img1' not in files or 'img2' not in files:
                print(f"⚠️  跳过不完整配对 {idx}")
                errors += 1
                continue

            # 创建目标目录
            target_dir = os.path.join(target_root, f"{idx}")
            os.makedirs(target_dir, exist_ok=True)

            # 复制文件
            shutil.copy2(
                os.path.join(source_dir, files['img1']),
                os.path.join(target_dir, 'img1.png')
            )
            shutil.copy2(
                os.path.join(source_dir, files['img2']),
                os.path.join(target_dir, 'img2.png')
            )
            
            processed += 1
            print(f"✅ 完成 {idx} 的复制")
            
        except Exception as e:
            errors += 1
            print(f"❌ 处理 {idx} 时出错: {str(e)}")

    # 最终报告
    print(f"\n操作完成！\n成功: {processed} 对\n失败: {errors} 对")

if __name__ == '__main__':
    main()