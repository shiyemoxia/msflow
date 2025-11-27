import os
import argparse
import numpy as np
import csv

import matplotlib.pyplot as plt


def load_scores(root_dir: str):
    ckpt_dir = os.path.join(root_dir, 'work_dirs', 'msflow_wide_resnet50_2_avgpool_pl258', 'smap', 'SMAP')
    scores_dir = os.path.join(ckpt_dir, 'scores')

    anomaly_score_path = os.path.join(scores_dir, 'anomaly_score.npy')
    anomaly_map_add_path = os.path.join(scores_dir, 'anomaly_score_map_add.npy')
    filenames_path = os.path.join(scores_dir, 'filenames.npy')
    labels_path = os.path.join(scores_dir, 'labels.npy')

    anomaly_score = np.load(anomaly_score_path)
    anomaly_map_add = np.load(anomaly_map_add_path)
    filenames = np.load(filenames_path)
    labels = np.load(labels_path)

    return ckpt_dir, scores_dir, anomaly_score, anomaly_map_add, filenames, labels


def export_csv(scores_dir: str, anomaly_score, filenames, labels, top_k: int = None):
    csv_path = os.path.join(scores_dir, 'scores.csv')

    # 排序：分数从高到低
    idx = np.argsort(-anomaly_score)
    if top_k is not None:
        idx = idx[:top_k]

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['rank', 'index', 'filename', 'label', 'anomaly_score'])
        for rank, i in enumerate(idx, start=1):
            writer.writerow([
                rank,
                int(i),
                filenames[i],
                int(labels[i]),
                float(anomaly_score[i]),
            ])

    print(f'CSV 导出完成: {csv_path}')


def save_heatmaps(scores_dir: str, anomaly_map_add, filenames, anomaly_score, top_n: int = 5):
    vis_dir = os.path.join(scores_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)

    idx = np.argsort(-anomaly_score)
    idx = idx[:top_n]

    for rank, i in enumerate(idx, start=1):
        m = anomaly_map_add[i]
        fname = os.path.basename(filenames[i])
        score = anomaly_score[i]

        plt.figure(figsize=(4, 4))
        plt.imshow(m, cmap='jet')
        plt.colorbar()
        plt.title(f'{fname}\nscore={score:.4f}')
        plt.axis('off')

        out_path = os.path.join(vis_dir, f'rank{rank:02d}_{fname}.png')
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

        print(f'保存热力图: {out_path}')


def main():
    parser = argparse.ArgumentParser(description='Export SMAP anomaly scores and heatmaps')
    parser.add_argument('--root-dir', type=str, default='.', help='项目根目录 (包含 work_dirs 的路径)')
    parser.add_argument('--top-k', type=int, default=None, help='导出到 CSV 的前 K 个样本, 默认导出全部')
    parser.add_argument('--top-n-vis', type=int, default=5, help='保存热力图的前 N 个样本')

    args = parser.parse_args()

    ckpt_dir, scores_dir, anomaly_score, anomaly_map_add, filenames, labels = load_scores(args.root_dir)
    print('ckpt_dir:', ckpt_dir)
    print('scores_dir:', scores_dir)
    print('样本数:', len(anomaly_score))

    export_csv(scores_dir, anomaly_score, filenames, labels, top_k=args.top_k)
    save_heatmaps(scores_dir, anomaly_map_add, filenames, anomaly_score, top_n=args.top_n_vis)


if __name__ == '__main__':
    main()
