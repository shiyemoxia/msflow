# 使用 msflow 进行 SMAP 时间序列异常检测的改造与实验报告

> 项目路径：`G:/research/归一化流/msflow`

---

## 1. 背景与目标

原始项目 **msflow** 是一个面向 **工业图像异常检测** 的方法，输入是 RGB 图像，输出为：

- 图像级 anomaly score（检测是否异常）
- 像素级 anomaly heatmap（定位异常区域）

本次工作的目标是：

- 在 **不大改模型结构** 的前提下，
- 把 msflow 迁移到 **时间序列异常检测** 场景，
- 使用 NASA SMAP 数据集，
- 构造合理的 **时间点（time-step）级评价指标**（AUROC / AP），用来衡量模型在时间序列上的检测效果。

简要来说，就是：

> 利用 msflow 的“图像+像素异常分数”机制，把时间序列伪装成“3 通道伪图像”送入模型，然后再把输出的 2D heatmap 还原成随时间变化的 anomaly score 曲线，从而在时间点级别进行评价。

---

## 2. 数据集：SMAP

### 2.1 数据来源与结构

- 使用的是 NASA 公布的 **SMAP/MSL 异常检测数据集**，本地解压路径（相对项目根目录）：
  - `./1/labeled_anomalies.csv`
  - `./1/data/data/train/*.npy`
  - `./1/data/data/test/*.npy`

- `labeled_anomalies.csv` 中的关键字段：
  - `chan_id`：通道 ID，对应一个 `.npy` 文件
  - `spacecraft`：选择 `SMAP`
  - `anomaly_sequences`：形如 `[[start1, end1], [start2, end2], ...]`，给出 **时间区间级别** 的异常标签
  - `num_values`：该通道的总时间点数

- `.npy` 文件结构：
  - 每个 `chan_id.npy` 对应一条时间序列；
  - 形状一般为 `(T, D)`：
    - `T`：时间长度
    - `D`：变量维度（约 25 维左右）

### 2.2 标签特点与评价难点

- 原始 msflow 的评价是“图像级 + 像素级”：
  - 每张图像一个 0/1 标签（正常/异常）；
  - 每个像素一个 0/1 标签（是否属于异常区域）。

- SMAP 的标签是 **时间区间级**：
  - 先在 `labeled_anomalies.csv` 中指出某个 `chan_id` 的哪些时间段 `[start, end]` 是异常；
  - 并不是为每个样本（通道）直接给一个 0/1 标签，更不是像素级标签。

这导致：

- 如果简单把“有异常区间的通道”标成 1、“无异常区间的通道”标成 0，再去算样本级 AUROC：
  - 很可能在当前 split 下，所有 test 样本都是“有异常”的，导致 `y_true` 只有一个类别，`roc_auc_score` 报 `UndefinedMetricWarning`。
- 因此，**原始的 msflow 图像级 AUROC 并不能反映 SMAP 时间序列的真实检测效果**。

本次工作重点就是：

- 在 msflow 的输出基础上，**构造 time-step 级别的标签与分数**，
- 并据此计算 **时间点 AUROC / AP**。

---

## 3. 总体思路

整体流程可以概括为：

1. **数据改造**：
   - 从 SMAP 的多变量时间序列中选择 3 个变量；
   - 将一条 `(T, D)` 的时间序列变换成形状为 `3×H×W` 的伪图像（H=W=512）；
   - 使用与图像类似的归一化方式（per-channel mean/std）。

2. **模型保持不变**：
   - 仍然使用 ResNet 作为特征提取 backbone（如 `wide_resnet50_2`）；
   - 仍然使用 msflow 原有的 normalizing flow 结构与后处理逻辑。

3. **训练与推理**：
   - 使用 `SMAPDataset` 作为自定义数据集；
   - 训练阶段依旧按照“样本级正常 vs 异常”的方式组织，但只是为了让模型学到分数分布；
   - 推理阶段导出：样本级 anomaly score 与 2D anomaly heatmap。

4. **评估（思路 1）**：
   - 利用 `labeled_anomalies.csv` 中的 `[start, end]` 区间，把每个时间点打 0/1 标签；
   - 将每个样本的 2D anomaly heatmap 压缩回一条 1D 时间分数曲线，并插值到原始长度 `T_len`；
   - 对所有通道的 `(y_t, s_t)` 进行拼接，计算 **time-step ROC AUC** 与 **Average Precision (AP)**。

---

## 4. 关键代码改动

### 4.1 `default.py`：配置文件

路径：`default.py`

主要改动：

1. **切换默认数据集为 SMAP**，并设置类别名：

```python
# dataset
# 原来：dataset = 'mvtec'
dataset = 'smap'
class_name = 'SMAP'
```

2. **设置输入尺寸与使用的维度**：

```python
input_size = (512, 512)

# 选取 SMAP 中用作 3 通道的维度索引
smap_used_dims = [0, 1, 2]
```

3. **根据 SMAP 训练集统计得到的 per-channel mean / std**（在 `1.py` 中辅助计算）：

```python
# img normalization (for 3 selected dims)
img_mean = [m1, m2, m3]  # 实际代码中填入具体数值
img_std  = [s1, s2, s3]
```

4. **模式与 checkpoint**（用于 test 模式评估）：

```python
# base
seed = 9826
gpu = '1'
device = torch.device("cuda")
# mode = 'train'
# eval_ckpt = ''
mode = 'test'
# 使用最新的 last.pt 做评估
eval_ckpt = r'./work_dirs/msflow_wide_resnet50_2_avgpool_pl258/smap/SMAP/last.pt'
```

5. **backbone 选择**：

实验中切换过 `resnet18` 和 `wide_resnet50_2`，目前设置为：

```python
extractor = 'wide_resnet50_2'  # [resnet18, resnet34, resnet50, resnext50_32x4d, wide_resnet50_2]
```

---

### 4.2 `main.py`：参数解析与数据集选择

路径：`main.py`

主要改动点：

1. **允许 `--dataset smap`**，并使 argparse 的默认值直接来自 `default.py` 的配置对象 `c`：

```python
parser.add_argument('--dataset', default=c.dataset, type=str, 
                    choices=['mvtec', 'visa', 'smap'], help='dataset name')

parser.add_argument('--mode', default=c.mode, type=str, help='train or test.')
parser.add_argument('--eval_ckpt', default=c.eval_ckpt, type=str, help='checkpoint path for evaluation.')
```

2. **根据数据集设置 `data_path` 和 `class_names`**：

```python
elif c.dataset == 'smap':
    setattr(c, 'data_path', './1/data/data')
    if c.class_names == ['all']:
        setattr(c, 'class_names', ['SMAP'])
```

3. **保证 `input_size` 不被覆盖**：

原始代码会根据 `class_name` 自动设置 `input_size`，我们保持逻辑，但在 `default.py` 中已经指定为 `(512, 512)`，保证与 SMAP 映射一致。

---

### 4.3 `datasets.py`：新增 `SMAPDataset`

路径：`datasets.py`

新增了一个时间序列数据集类 `SMAPDataset`，并在文件顶部和 `__all__` 中增加了相应导入，使 `from datasets import SMAPDataset` 可用。

#### 4.3.1 设计目标

- 将一条 `(T, D)` 的时间序列转成 msflow 期望的 `3×H×W` 图像张量；
- 使用配置中指定的 `smap_used_dims` 作为 3 个通道；
- 保持与图像数据相同的 `Normalize(mean=img_mean, std=img_std)` 接口；
- 保留 `mask` 张量接口（尽管在 SMAP 中没有像素级 ground truth）。

#### 4.3.2 核心逻辑

**构造函数**：

```python
class SMAPDataset(Dataset):
    def __init__(self, c, is_train=True):
        self.dataset_path = c.data_path  # './1/data/data'
        self.is_train = is_train
        self.input_size = c.input_size  # (512, 512)

        # 读取 labeled_anomalies.csv，过滤 spacecraft == 'SMAP'
        labels_path = os.path.join(os.path.dirname(os.path.dirname(self.dataset_path)), 'labeled_anomalies.csv')
        df = pd.read_csv(labels_path)
        df = df[df['spacecraft'] == 'SMAP']
        self.smap_ids = sorted(df['chan_id'].unique().tolist())

        # 构造通道级 label（是否存在异常区间）
        self.label_map = {}
        for _, row in df.iterrows():
            chan_id = row['chan_id']
            seq = str(row['anomaly_sequences'])
            has_anomaly = '[]' not in seq and seq.strip() != '[]'
            self.label_map[chan_id] = 1 if has_anomaly else 0

        split = 'train' if self.is_train else 'test'
        split_dir = os.path.join(self.dataset_path, split)

        self.x = []      # 存放所有样本的文件路径
        self.y = []      # 通道级 label（训练时全 0，测试时根据 label_map）
        self.mask = []   # 这里没有像素 mask，用占位

        for chan_id in self.smap_ids:
            fname = chan_id + '.npy'
            fpath = os.path.join(split_dir, fname)
            if not os.path.exists(fpath):
                continue
            self.x.append(fpath)
            if self.is_train:
                self.y.append(0)  # 训练集视作正常
            else:
                self.y.append(self.label_map.get(chan_id, 0))
            self.mask.append(None)

        self.normalize = T.Compose([T.Normalize(c.img_mean, c.img_std)])
        self.used_dims = getattr(c, 'smap_used_dims', [0, 1, 2])
```

**`__getitem__`：时间序列 → 3×H×W 伪图像**

```python
def __getitem__(self, idx):
    x_path = self.x[idx]
    y = self.y[idx]
    arr = np.load(x_path)  # (T,) or (T, D)

    if arr.ndim == 1:
        arr = arr[:, None]

    T_len, D = arr.shape

    # 选取指定的维度，保证索引合法
    dims = [d for d in self.used_dims if d < D]
    arr = arr[:, dims]     # (T, C)

    # 转换为 (C, T)
    arr = arr.T

    x = torch.from_numpy(arr).float()    # (C, T)

    # 扩展成 (1, C, T, 1) 方便用 2D interpolate
    x = x.unsqueeze(0).unsqueeze(2)

    # 双线性插值到 input_size=(H, W)，这里 T 方向映射到 W
    x = F.interpolate(x, size=self.input_size, mode='bilinear', align_corners=False)

    # 变回 (C, H, W)
    x = x.squeeze(0)

    # 归一化
    x = self.normalize(x)

    # SMAP 没有像素级 mask，这里返回全零占位
    mask = torch.zeros([1, *self.input_size])
    return x, y, mask
```

**`__len__`**：

```python
def __len__(self):
    return len(self.x)
```

---

### 4.4 `train.py`：集成 `SMAPDataset` 并导出 scores

路径：`train.py`

#### 4.4.1 选择数据集

在 `train(c)` 函数开始处，根据 `c.dataset` 选择不同的数据集：

```python
if c.dataset == 'mvtec':
    Dataset = MVTecDataset
elif c.dataset == 'visa':
    Dataset = VisADataset
elif c.dataset == 'smap':
    Dataset = SMAPDataset
else:
    raise ValueError(f"Unsupported dataset: {c.dataset}")

train_dataset = Dataset(c, is_train=True)
test_dataset  = Dataset(c, is_train=False)
```

其余训练过程（构造 extractor、parallel_flows、fusion_flow、优化器、scheduler 等）基本保持原状。

#### 4.4.2 在 test 模式下导出 anomaly scores 与 heatmaps

在 `c.mode == 'test'` 的分支中，增加了将模型输出保存到 `scores` 目录的逻辑：

```python
if c.mode == 'test':
    start_epoch = load_weights(parallel_flows, fusion_flow, c.eval_ckpt)
    epoch = start_epoch + 1
    gt_label_list, gt_mask_list, outputs_list, size_list = \
        inference_meta_epoch(c, epoch, test_loader, extractor, parallel_flows, fusion_flow)

    anomaly_score, anomaly_score_map_add, anomaly_score_map_mul = \
        post_process(c, size_list, outputs_list)

    # export anomaly scores and maps for offline analysis
    save_dir = os.path.join(c.ckpt_dir, 'scores')
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, 'anomaly_score.npy'), anomaly_score)
    np.save(os.path.join(save_dir, 'anomaly_score_map_add.npy'), anomaly_score_map_add)
    np.save(os.path.join(save_dir, 'anomaly_score_map_mul.npy'), anomaly_score_map_mul)
    np.save(os.path.join(save_dir, 'filenames.npy'), np.asarray(test_loader.dataset.x))
    np.save(os.path.join(save_dir, 'labels.npy'), np.asarray(gt_label_list))

    det_auroc, loc_auroc, loc_pro_auc, \
        best_det_auroc, best_loc_auroc, best_loc_pro = \
            eval_det_loc(det_auroc_obs, loc_auroc_obs, loc_pro_obs, epoch,
                         gt_label_list, anomaly_score,
                         gt_mask_list, anomaly_score_map_add, anomaly_score_map_mul, c.pro_eval)

    return
```

- `anomaly_score.npy`：每个样本一个分数（样本级）；
- `anomaly_score_map_add.npy`：每个样本一个 2D heatmap（用于后续时间点分析）；
- `filenames.npy`：样本对应的 `.npy` 路径；
- `labels.npy`：当前 msflow 框架下的样本级 0/1 标签（但对于 SMAP 来说信息有限）。

---

### 4.5 `export_smap_scores.py`：导出 CSV 和可视化 heatmap

路径：`export_smap_scores.py`

作用：

- 从 `scores` 目录读取：
  - `anomaly_score.npy`
  - `filenames.npy`
  - `labels.npy`
- 生成一个 `scores.csv`，每一行为：
  - `rank`（按 anomaly_score 从高到低排序）
  - `chan_id`
  - `anomaly_score`
  - `label` 等；
- 同时对 top-K 样本的 `anomaly_score_map_add` 画出热力图 PNG，保存到 `scores/vis/`，便于直观查看异常模式。

示例使用：

```bash
python export_smap_scores.py
```

运行后输出类似：

```text
ckpt_dir: ./work_dirs/msflow_wide_resnet50_2_avgpool_pl258/smap/SMAP
scores_dir: ./work_dirs/msflow_wide_resnet50_2_avgpool_pl258/smap/SMAP/scores
样本数: 54
CSV 导出完成: .../scores.csv
保存热力图: .../vis/rank01_T-1.npy.png
...
```

---

### 4.6 `eval_smap_timestep.py`：思路 1 — 时间点级 AUROC 评价

路径：`eval_smap_timestep.py`

这是本次工作的关键脚本，用于从 msflow 的 2D heatmap 中恢复 **time-step 级别的 anomaly score**，并结合 SMAP 的区间标签计算 ROC AUC / AP。

#### 4.6.1 主要步骤概览

1. 从 `scores` 目录读取：
   - `anomaly_score_map_add.npy`
   - `filenames.npy`
2. 从 `./1/labeled_anomalies.csv` 读取 `anomaly_sequences`，构造 `chan_id -> [intervals]` 映射；
3. 对于每个 `chan_id`：
   - 读取对应的 test `.npy` 时间序列，获取原始长度 `T_len`；
   - 按 `[start, end]` 区间构造 0/1 的时间点标签 `y_t`，长度为 `T_len`；
   - 从 2D heatmap `m`（形状约为 `H×W = 512×512`）压缩得到一条 1D 曲线：
     - 先在空间一个维度上取平均：`s_1d = m.mean(axis=0)`，得到 `(W,)`；
     - 用 `np.interp` 将 `(W,)` 线性插值到长度 `T_len`，得到 `s_t`；
   - 把每条通道的 `y_t` 和 `s_t` 累积起来。

4. 将所有通道的 `y_t` 拼接为 `y_all`，所有的 `s_t` 拼接为 `s_all`；
5. 用 `sklearn.metrics` 计算：
   - `roc_auc_score(y_all, s_all)`
   - `average_precision_score(y_all, s_all)`；
6. 可选：将 `y_all` 和 `s_all` 存成 `timestep_labels.npy`、`timestep_scores.npy`，方便后续画 ROC/PR 曲线等。

#### 4.6.2 核心函数片段

**读取 scores**：

```python
def load_scores(root_dir: str):
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
```

**解析 `labeled_anomalies.csv` 为区间字典**：

```python
def build_interval_map(base_dir: str):
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
```

**区间 → 时间点标签**：

```python
def intervals_to_labels(T_len: int, intervals):
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
```

**压缩 2D map → 1D 时间分数并插值到 T_len**：

```python
def resize_1d(vec: np.ndarray, target_len: int) -> np.ndarray:
    src_len = vec.shape[0]
    if src_len == target_len:
        return vec.astype(np.float32)
    x_src = np.linspace(0.0, 1.0, src_len)
    x_tgt = np.linspace(0.0, 1.0, target_len)
    return np.interp(x_tgt, x_src, vec).astype(np.float32)
```

**构造全局时间点标签与分数**：

```python
def build_timestep_scores(root_dir: str, anomaly_map_add, filenames, chan2intervals):
    base_1 = os.path.join(root_dir, "1")
    data_root = os.path.join(base_1, "data", "data")
    test_dir = os.path.join(data_root, "test")

    all_labels = []
    all_scores = []

    for i in range(len(filenames)):
        path = filenames[i]
        fname = os.path.basename(str(path))
        chan_id = os.path.splitext(fname)[0]

        ts_path = os.path.join(test_dir, fname)
        if not os.path.exists(ts_path):
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
```

**主函数：计算 AUROC / AP**：

```python
def main():
    parser = argparse.ArgumentParser(description="Evaluate SMAP time-step AUROC from msflow outputs")
    parser.add_argument("--root-dir", type=str, default=".", help="项目根目录 (包含 1/ 和 work_dirs/) 的路径")
    parser.add_argument("--save-npy", action="store_true", help="是否把拼接后的 y_all 和 s_all 保存为 npy")
    args = parser.parse_args()

    ckpt_dir, scores_dir, anomaly_score, anomaly_map_add, filenames, labels = load_scores(args.root_dir)

    chan2intervals = build_interval_map(os.path.join(args.root_dir, "1"))
    y_all, s_all = build_timestep_scores(args.root_dir, anomaly_map_add, filenames, chan2intervals)

    pos = int(y_all.sum())
    neg = int((y_all == 0).sum())
    print(f"总时间点数: {len(y_all)}, 正类(异常)时间点: {pos}, 负类(正常)时间点: {neg}")

    if len(np.unique(y_all)) < 2:
        print("只有一个类别，无法计算 ROC AUC。")
        return

    auc = roc_auc_score(y_all, s_all)
    ap = aaverage_precision_score(y_all, s_all)
    print(f"Time-step ROC AUC: {auc:.4f}")
    print(f"Time-step Average Precision (AP): {ap:.4f}")

    if args.save_npy:
        np.save(os.path.join(scores_dir, "timestep_labels.npy"), y_all)
        np.save(os.path.join(scores_dir, "timestep_scores.npy"), s_all)
        print("已保存 timestep_labels.npy 和 timestep_scores.npy 到:", scores_dir)
```

---

## 5. 实验与结果

### 5.1 训练过程（msflow 在 SMAP 上）

- 采用配置：
  - `dataset = 'smap'`
  - `class_name = 'SMAP'`
  - `input_size = (512, 512)`
  - `extractor = 'wide_resnet50_2'`
  - `meta_epochs = 25`（每个 meta epoch 内部有若干 sub-epochs）
- 训练日志中：
  - Loss 逐渐收敛到大负值（符合 normalizing flow 的对数似然形式）；
  - 但原始的 `Det.AUROC` / `Loc.AUROC` 一直为 `nan`，这是因为：
    - 在 msflow 原有评价逻辑里，使用的是 **样本级** 和 **像素级** 标签；
    - 当前 SMAP test 集在样本级标签上仅有单一类别，导致 `roc_auc_score` 直接报 `UndefinedMetricWarning`；
  - 这些 nan **不影响我们后续基于时间点的评价**。

### 5.2 时间点级评价：第一版（未充分训练时）

在初版模型（训练轮次较少 or 使用较弱配置）下，运行：

```bash
python eval_smap_timestep.py --root-dir . --save-npy
```

得到：

```text
总时间点数: 435826, 正类(异常)时间点: 55817, 负类(正常)时间点: 380009
Time-step ROC AUC: 0.4300
Time-step Average Precision (AP): 0.1090
```

分析：

- 正类比例 ≈ 12.8%，AP 基线约 0.128；
- ROC AUC 约 0.43 < 0.5（劣于随机排序）；
- AP 约 0.109 < 0.128（也略差于随机）；

说明在这一阶段：

> 模型几乎没有学到有效的“异常 vs 正常”时间点区分能力，甚至有一定“反向排序”的倾向。

### 5.3 时间点级评价：进一步训练与调参之后

在增加训练轮数、采用 `wide_resnet50_2` backbone 并持续训练到 `last.pt` 后，重新导出 scores 并运行：

```bash
python main.py        # 以 train 模式继续训练/保存 last.pt
python main.py        # 切换为 test 模式，导出 scores
python export_smap_scores.py
python eval_smap_timestep.py --root-dir . --save-npy
```

得到：

```text
总时间点数: 435826, 正类(异常)时间点: 55817, 负类(正常)时间点: 380009
Time-step ROC AUC: 0.6529
Time-step Average Precision (AP): 0.1867
```

对比与解释：

- 正类比例仍为约 12.8%；
- **ROC AUC 从 0.43 提升到 0.65 左右**：
  - 说明模型已经学到一定区分能力，随机抽取一个异常时间点和一个正常时间点，被模型正确排序（异常分数更高）的概率约为 65%；
- **AP 从 0.109 升到 ~0.187，高于随机基线 0.128**：
  - 在高度不平衡的场景下，这表明模型在高分区域中富集了更多的异常点，相比随机已有明显改进。

总评：

> 在保持 msflow 图像检测结构基本不变、通过“时间序列 → 伪图像”的方式迁移到 SMAP 上后，模型在 **时间点级别** 的异常检测效果已经明显优于随机基线，但与针对时间序列专门设计的 SOTA 方法相比，仍处于中等偏弱水平（ROC AUC ~0.65、AP ~0.19）。

---

## 6. 已完成工作小结

- **数据适配**：
  - 下载并整理 SMAP 数据集至 `./1/` 目录；
  - 编写脚本统计训练集 per-channel mean/std（用于 `img_mean` / `img_std`）；
  - 设计并实现 `SMAPDataset`，将 `(T, D)` 时间序列映射为 `3×512×512` 伪图像。

- **配置与主程序改造**：
  - 在 `default.py` 中新增 SMAP 相关配置（`dataset='smap'`、`class_name='SMAP'`、`smap_used_dims`、`img_mean/img_std`、`extractor` 等）；
  - 在 `main.py` 中支持 `--dataset smap`，设置 `data_path='./1/data/data'`，并将 argparse 默认值与配置对象 `c` 对齐；
  - 保持 msflow 的主训练/推理逻辑不变，仅替换数据集和 backbone。

- **结果导出与可视化**：
  - 在 `train.py` 的 test 模式下，将 `anomaly_score`、`anomaly_score_map_add/mul`、`filenames`、`labels` 保存为 `.npy`，统一到 `work_dirs/.../scores` 目录；
  - 新增 `export_smap_scores.py`，导出 `scores.csv` 与 top-K heatmap 图像，便于肉眼分析模型在不同通道上的整体分布情况。

- **时间点级评价（核心贡献）**：
  - 明确原始 msflow 在 SMAP 上的样本级 AUROC **无法直接反映真实效果** 的原因（标签单一、标签粒度不匹配）；
  - 设计并实现 `eval_smap_timestep.py`：
    - 基于 SMAP 的 `anomaly_sequences` 区间构造时间点 0/1 标签；
    - 将 msflow 输出的 2D heatmap 压缩并插值回原始时间长度，得到 per-time-step anomaly score；
    - 计算 global time-step ROC AUC 与 AP 作为评价指标；
  - 得到实验结果：
    - 早期版本：AUC ≈ 0.43, AP ≈ 0.11（劣于随机）；
    - 训练充分后：AUC ≈ 0.65, AP ≈ 0.19（明显优于随机，但仍有提升空间）。

---

## 7. 后续可选工作方向（供与导师讨论）

1. **进一步可视化分析**：
   - 基于 `timestep_labels.npy` 和 `timestep_scores.npy`，绘制 ROC 曲线与 PR 曲线；
   - 针对单条通道（如 `T-1`、`F-2` 等 top-ranked 通道），绘制：
     - 原始时间序列曲线（任选 1~3 维）；
     - 对应的时间点 anomaly score 曲线；
     - 标注异常时间区间（阴影）；
   - 通过这些图更直观地理解模型在“异常前后”的反应情况。

2. **改进时间序列到伪图像的映射方式**：
   - 当前做法是简单地将时间轴线性缩放至 512 并在空间一维上均值；
   - 可以尝试：
     - 使用更适合时间序列的 2D 布局（如 Gramian Angular Fields、Markov Transition Field 等）再输入 msflow；
     - 调整 `input_size` 与插值策略，避免在时间维度上过度压缩重要模式。

3. **考虑时间序列原生模型**：
   - 将 msflow 的“正常样本建模 + likelihood 作为 anomaly score”的思想迁移到 1D 结构：
     - 使用 1D CNN / Transformer / RNN 作为特征提取 backbone；
     - 在 feature space 上使用 flow 模型做 density estimation；
   - 或者直接与目前公开的时间序列异常检测模型（如 OmniAnomaly、USAD、TranAD 等）对比，作为基线。

4. **更细致的标签处理**：
   - 检查 SMAP 区间标签与 `num_values` 是否完全对齐；
   - 对重叠/相邻区间进行合并，避免人为的标签噪声；
   - 探索不同的“邻域扩展”（例如对每个异常区间前后扩展若干步）是否会更符合检测目标。

---

以上是目前将 msflow 迁移到 SMAP 时间序列异常检测上的主要改动、思路和实验结果总结。该报告可直接用于向导师说明：

- 为什么原始 msflow 的 AUROC 在 SMAP 上不适用；
- 我们是如何构造合理的时间点级指标；
- 当前方法在 SMAP 上的实际表现与局限；
- 以及后续可以继续优化和深入的方向。
