import os
import argparse
import ast

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


def load_scores(root_dir: str):
    """Load anomaly scores and maps saved by msflow test mode.

    Assumes the same ckpt_dir structure as export_smap_scores.py.
    """
    ckpt_dir = os.path.join(
        root_dir,
        "work_dirs",
        "msflow_wide_resnet50_2_avgpool_pl258",
        "smap",
        "SMAP",
    )
    scores_dir = os.path.join(ckpt_dir, "scores")

    anomaly_score = np.load(os.path.join(scores_dir, "anomaly_score.npy"))
    anomaly_map_add = np.load(os.path.join(scores_dir, "anomaly_score_map_add.npy"))
    filenames = np.load(os.path.join(scores_dir, "filenames.npy"))
    labels = np.load(os.path.join(scores_dir, "labels.npy"))

    return ckpt_dir, scores_dir, anomaly_score, anomaly_map_add, filenames, labels


def build_interval_map(base_dir: str):
    """Build mapping: chan_id -> list of [start, end] anomaly intervals."""
    labels_path = os.path.join(base_dir, "labeled_anomalies.csv")
    df = pd.read_csv(labels_path)
    df = df[df["spacecraft"] == "SMAP"]

    chan2intervals = {}
    for _, row in df.iterrows():
        chan = row["chan_id"]
        seq_str = str(row.get("anomaly_sequences", "[]"))
        try:
            intervals = ast.literal_eval(seq_str)
            if not isinstance(intervals, (list, tuple)):
                intervals = []
        except Exception:
            intervals = []
        chan2intervals[chan] = intervals

    return chan2intervals


def intervals_to_labels(T_len: int, intervals):
    """Convert list of [start, end] intervals to 0/1 label array of length T_len."""
    y = np.zeros(T_len, dtype=np.int8)
    for itv in intervals:
        if not isinstance(itv, (list, tuple)) or len(itv) != 2:
            continue
        s, e = int(itv[0]), int(itv[1])
        if e < 0 or s >= T_len:
            continue
        s = max(0, s)
        e = min(T_len - 1, e)
        if e >= s:
            y[s : e + 1] = 1
    return y


def resize_1d(vec: np.ndarray, target_len: int) -> np.ndarray:
    """Linearly resize 1D vector to target_len using np.interp."""
    src_len = vec.shape[0]
    if src_len == target_len:
        return vec.astype(np.float32)
    x_src = np.linspace(0.0, 1.0, src_len)
    x_tgt = np.linspace(0.0, 1.0, target_len)
    return np.interp(x_tgt, x_src, vec).astype(np.float32)


def build_timestep_scores(root_dir: str, anomaly_map_add, filenames, chan2intervals):
    """From 2D anomaly maps and intervals, build per-time-step labels and scores."""
    base_1 = os.path.join(root_dir, "1")
    data_root = os.path.join(base_1, "data", "data")
    test_dir = os.path.join(data_root, "test")

    all_labels = []
    all_scores = []

    for i in range(len(filenames)):
        path = filenames[i]
        # filenames.npy 里是相对路径，比如 './1/data/data/test/E-3.npy' 或绝对路径
        fname = os.path.basename(str(path))
        chan_id = os.path.splitext(fname)[0]

        ts_path = os.path.join(test_dir, fname)
        if not os.path.exists(ts_path):
            # 如果 filenames 已经是完整路径，直接用它
            if os.path.exists(path):
                ts_path = path
            else:
                print(f"[WARN] 找不到时间序列文件: {ts_path}, 跳过 {chan_id}")
                continue

        arr = np.load(ts_path)
        if arr.ndim == 1:
            T_len = arr.shape[0]
        else:
            T_len = arr.shape[0]

        intervals = chan2intervals.get(chan_id, [])
        y_t = intervals_to_labels(T_len, intervals)

        # 将 2D anomaly map 压成 1D，再插值到长度 T_len
        m = anomaly_map_add[i]  # (H, W) = (512, 512)
        s_1d = m.mean(axis=0)   # (W,)
        s_t = resize_1d(s_1d, T_len)

        all_labels.append(y_t.astype(np.int8))
        all_scores.append(s_t.astype(np.float32))

    if not all_labels:
        raise RuntimeError("没有成功构造任何时间点标签，请检查文件路径和 labeled_anomalies.csv。")

    y_all = np.concatenate(all_labels)
    s_all = np.concatenate(all_scores)
    return y_all, s_all


def main():
    parser = argparse.ArgumentParser(description="Evaluate SMAP time-step AUROC from msflow outputs")
    parser.add_argument("--root-dir", type=str, default=".", help="项目根目录 (包含 1/ 和 work_dirs/) 的路径")
    parser.add_argument("--save-npy", action="store_true", help="是否把拼接后的 y_all 和 s_all 保存为 npy")
    args = parser.parse_args()

    ckpt_dir, scores_dir, anomaly_score, anomaly_map_add, filenames, labels = load_scores(args.root_dir)
    print("ckpt_dir:", ckpt_dir)
    print("scores_dir:", scores_dir)
    print("样本数 (通道数):", len(filenames))

    chan2intervals = build_interval_map(os.path.join(args.root_dir, "1"))

    y_all, s_all = build_timestep_scores(args.root_dir, anomaly_map_add, filenames, chan2intervals)
    pos = int(y_all.sum())
    neg = int((y_all == 0).sum())
    print(f"总时间点数: {len(y_all)}, 正类(异常)时间点: {pos}, 负类(正常)时间点: {neg}")

    if len(np.unique(y_all)) < 2:
        print("只有一个类别，无法计算 ROC AUC。")
        return

    auc = roc_auc_score(y_all, s_all)
    ap = average_precision_score(y_all, s_all)
    print(f"Time-step ROC AUC: {auc:.4f}")
    print(f"Time-step Average Precision (AP): {ap:.4f}")

    if args.save_npy:
        np.save(os.path.join(scores_dir, "timestep_labels.npy"), y_all)
        np.save(os.path.join(scores_dir, "timestep_scores.npy"), s_all)
        print("已保存 timestep_labels.npy 和 timestep_scores.npy 到:", scores_dir)


if __name__ == "__main__":
    main()
