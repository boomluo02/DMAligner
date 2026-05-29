import pandas as pd
import pickle
import os
import shutil

# 读取CSV文件
df = pd.read_csv('/root/Diff/output_DSIA/metrics_results.csv')

# 使用字典推导式构建结果字典
result_dict = {
    key: df[df['s_name'].str[:2] == key][['s_name', 'PSNR', 'SSIM']]
          .sort_values('SSIM', ascending=False)
          .reset_index(drop=True)
    for key in ['SS', 'SL', 'LS', 'LL']
}

# 初始化存储结构
selected_samples_dict = {}  # 按键存储s_name列表
all_selected_s_names = []    # 合并所有s_name


key_range_dict = {"SS":4000, "SL":700, "LS":800, "LL":2000}
# 3. 提取每个键的前 500 个样本并计算平均 PSNR
avg_psnr = {}
avg_ssim = {}
for key in result_dict:
    # top_samples = result_dict[key].head(500) # 获取前 500 个样本（如果样本不足 500，则取全部）
    # top_samples = result_dict[key].iloc[key_range_dict[key]:key_range_dict[key]+1000:2]  # 注意Python切片左闭右开

    if(key in ["SS", "SL"]):
        top_samples = result_dict[key].iloc[key_range_dict[key]:key_range_dict[key]+500]
    else:
        top_samples = result_dict[key].iloc[::10]
    # 计算平均 PSNR，保留 4 位小数
    avg_psnr[key] = round(top_samples["PSNR"].mean(), 6)
    avg_ssim[key] = round(top_samples["SSIM"].mean(), 6)

    # 转换为列表并存储
    s_names = top_samples["s_name"].tolist()
    selected_samples_dict[key] = s_names
    all_selected_s_names.extend(s_names)

# 保存到CSV（所有s_name）
pd.DataFrame(all_selected_s_names, columns=["s_name"]).to_csv(
    "/root/Diff/output_DSIA/DSIA_test_selected_new.csv", index=False
)

# 保存到PKL（按键分类的字典）
# with open("/root/Diff/output_DSIA/DSIA_test_selected.pkl", "wb") as f:
#     pickle.dump(selected_samples_dict, f)
# print("保存完成：DSIA_test_selected.csv 和 DSIA_test_selected.pkl")

# 4. 打印结果
print("各键前 500 个样本的平均 PSNR和SSIM: ")
for key, value in avg_psnr.items():
    print(f"{key}: samples {len(selected_samples_dict[key])} PSNR {value}, SSIM {avg_ssim[key]}")

# # 复制操作
# source_root = "/root/Diff/output_DSIA/inference/inference-20250218101419/test_output"
# target_root = "/root/Diff/output_DSIA_test_selected"
# os.makedirs(target_root, exist_ok=True)
# for s_name in all_selected_s_names:
#     src_path = os.path.join(source_root, s_name)
#     dst_path = os.path.join(target_root, s_name)
    
#     if not os.path.exists(src_path):
#         print(f"警告：{s_name} 源文件夹不存在，已跳过")
#         continue
    
#     try:
#         # 复制整个文件夹（自动覆盖已存在的文件夹）
#         if os.path.exists(dst_path):
#             shutil.rmtree(dst_path)
#         shutil.copytree(src_path, dst_path)
#         print(f"成功复制：{s_name}")
#     except Exception as e:
#         print(f"错误：复制 {s_name} 失败 - {str(e)}")

# print("操作完成，请检查目标文件夹：", target_root)