import pandas as pd

# 读取CSV文件
csv_path = "/root/Diff/output_DSIA/metrics_results.csv"
df = pd.read_csv(csv_path)

# 添加类型列（取s_name前两位）
df['type'] = df['s_name'].str[:2]

# 定义目标分类顺序
categories = ['LL', 'LS', 'SL', 'SS']

# 预计算分组统计
results = []
for cat in categories:
    # 筛选当前分类的数据
    group_df = df[df['type'] == cat]
    
    # 计算统计指标
    stats = {
        "type": cat,
        "count": len(group_df),
        "psnr_mean": group_df['PSNR'].mean(),
        "psnr_std": group_df['PSNR'].std(),
        "ssim_mean": group_df['SSIM'].mean(),
        "ssim_std": group_df['SSIM'].std(),
        "samples_list": group_df['s_name'].tolist()  # 保存原始样本列表
    }
    results.append(stats)

# 转换结果为DataFrame
result_df = pd.DataFrame(results)

# 格式化输出
print("分类统计结果：")
print(result_df[['type', 'count', 'psnr_mean', 'psnr_std', 'ssim_mean', 'ssim_std']])

# 可选：保存完整结果到新CSV
result_df.to_csv(csv_path.replace('metrics_results', 'metrics_results_selected_test'), index=False)