import os
import numpy as np
import pandas as pd

base = r"G:\research\归一化流\msflow\1"

print("base:", base)

# 自动寻找包含 train/test 的 data 根目录
def find_data_root(base: str) -> str:
    candidates = [
        os.path.join(base, "data"),
        os.path.join(base, "data", "data"),
    ]
    for cand in candidates:
        if os.path.isdir(os.path.join(cand, "train")):
            return cand
    raise RuntimeError("在以下路径都没找到 train 目录，请确认：\n" + "\n".join(candidates))

data_root = find_data_root(base)
print("data_root:", data_root)

# 先看标注文件
labels_path = os.path.join(base, "labeled_anomalies.csv")
print("\n=== labeled_anomalies.csv 前几行 ===")
df = pd.read_csv(labels_path)
print(df.head())

used_dims = [0, 1, 2]
sum_ = np.zeros(len(used_dims), dtype=np.float64)
sq_sum = np.zeros(len(used_dims), dtype=np.float64)
count = 0

for split in ["train", "test"]:
    split_dir = os.path.join(data_root, split)
    files = sorted([f for f in os.listdir(split_dir) if f.endswith(".npy")])
    print(f"\n=== {split} 目录: {split_dir} ===")
    print(f"{split} 有 {len(files)} 个 .npy 文件")
    print(f"{split} 前 5 个文件名:", files[:5])

    if not files:
        continue

    if split == "train":
        for fname in files:
            fpath = os.path.join(split_dir, fname)
            arr = np.load(fpath)
            if arr.ndim == 1:
                arr = arr[:, None]
            T_len, D = arr.shape
            dims = [d for d in used_dims if d < D]
            arr = arr[:, dims]
            sum_[: len(dims)] += arr.sum(axis=0)
            sq_sum[: len(dims)] += (arr ** 2).sum(axis=0)
            count += T_len

    # 取第一个文件看一眼
    sample_name = files[0]
    sample_path = os.path.join(split_dir, sample_name)
    print(f"\n加载示例文件: {sample_name}")
    arr = np.load(sample_path)

    print("shape:", arr.shape)
    print("dtype:", arr.dtype)
    if arr.ndim == 1:
        print("一维序列，长度:", arr.shape[0])
    elif arr.ndim == 2:
        T, D = arr.shape
        print("时间长度 T:", T, "变量数 D:", D)
        print("前 3 行数据:\n", arr[:3])
    else:
        print("维度大于 2，实际维度:", arr.ndim)

if count > 0:
    mean = sum_ / count
    var = sq_sum / count - mean ** 2
    std = np.sqrt(np.maximum(var, 0.0))

    print("\n=== 统计结果（基于 train，dims", used_dims, ") ===")
    print("img_mean =", mean.tolist())
    print("img_std  =", std.tolist())