import os
from pathlib import Path
import random

# 指定目录路径
dir_path = Path("/home/lzl/DeepSDF/implicit-shape-reconstruction/casename_files/Colon/Task10_Colon__CTPelvic1K__fold0_3dfullres_pred")

# 列出目录下的所有文件
all_files = [f.name[:-7] for f in dir_path.iterdir() if f.is_file()]

# 随机打乱文件名列表
random.shuffle(all_files)

# 计算训练集的大小（70%）
train_size = int(0.7 * len(all_files))

# 分割文件名列表
train_files = all_files[:train_size]
eval_files = all_files[train_size:]

# 指定保存文件的目录
save_dir = Path("/home/lzl/DeepSDF/implicit-shape-reconstruction/casename_files/Colon/")

# 保存文件名到文件中
with open(save_dir / 'train_cases.txt', 'w') as f:
    for file in train_files:
        f.write(file + '\n')

with open(save_dir / 'eval_cases.txt', 'w') as f:
    for file in eval_files:
        f.write(file + '\n')