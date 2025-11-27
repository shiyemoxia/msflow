# MSFlow 部署指南

## 项目简介
MSFlow 是一个基于多尺度归一化流的无监督异常检测框架，用于图像异常检测和定位。

## 环境要求
- Python 3.9
- PyTorch >= 1.10
- CUDA 11.3 (推荐)

## 安装步骤

### 1. 创建虚拟环境（推荐）
```bash
# 使用 conda
conda create -n msflow python=3.9
conda activate msflow

# 或使用 venv
python -m venv msflow_env
# Windows:
msflow_env\Scripts\activate
# Linux/Mac:
source msflow_env/bin/activate
```

### 2. 安装 PyTorch
根据您的 CUDA 版本安装 PyTorch。访问 https://pytorch.org/ 获取安装命令。

例如，对于 CUDA 11.3：
```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

或者对于 CPU 版本：
```bash
pip install torch torchvision
```

### 3. 安装 FrEIA
FrEIA 是归一化流的核心库，需要单独安装：
```bash
pip install git+https://github.com/VLL-HD/FrEIA.git
```

### 4. 安装其他依赖
```bash
pip install -r requirements.txt
```

## 数据集准备

### MVTec AD 数据集
1. 从 [MVTec AD 官网](https://www.mvtec.com/company/research/datasets/mvtec-ad) 下载数据集
2. 解压到 `./data/MVTec` 目录
3. 目录结构应如下：
```
data/MVTec/
├── bottle/
│   ├── ground_truth/
│   ├── test/
│   └── train/
├── cable/
└── ...
```

### VisA 数据集
1. 从 [VisA 下载链接](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar) 下载数据集
2. 解压到 `./data/VisA_pytorch/1cls` 目录
3. 目录结构应如下：
```
data/VisA_pytorch/1cls/
├── candle/
│   ├── ground_truth/
│   ├── test/
│   └── train/
├── capsules/
└── ...
```

## 训练模型

### 训练 MVTec AD 数据集
```bash
# 训练所有类别
python main.py --mode train --dataset mvtec --class-names all

# 训练单个类别（例如 bottle）
python main.py --mode train --dataset mvtec --class-names bottle
```

### 训练 VisA 数据集
```bash
# 训练所有类别
python main.py --mode train --dataset visa --class-names all --pro-eval

# 训练单个类别
python main.py --mode train --dataset visa --class-names candle --pro-eval
```

### 使用 AMP 加速训练
```bash
python main.py --mode train --dataset mvtec --class-names all --amp_enable
```

### 使用 WandB 记录训练过程
```bash
python main.py --mode train --dataset mvtec --class-names all --wandb_enable
```

## 测试模型

```bash
# 测试单个类别
python main.py --mode test --dataset mvtec --class-names bottle --eval_ckpt ./work_dirs/msflow_wide_resnet50_2_avgpool_pl258/mvtec/bottle/best.pt
```

## 训练参数说明

- `--dataset`: 数据集名称 (mvtec 或 visa)
- `--mode`: 运行模式 (train 或 test)
- `--class-names`: 要训练的类别名称，使用 'all' 训练所有类别
- `--batch-size`: 批次大小（默认：8）
- `--lr`: 学习率（默认：1e-4）
- `--meta-epochs`: 元周期数（默认：25）
- `--sub-epochs`: 子周期数（默认：4）
- `--extractor`: 特征提取器（默认：wide_resnet50_2）
- `--amp_enable`: 启用自动混合精度训练
- `--wandb_enable`: 启用 WandB 日志记录
- `--pro-eval`: 评估 PRO 分数
- `--resume`: 恢复训练

## 结果输出

训练结果和检查点将保存在 `./work_dirs` 目录下，结构如下：
```
work_dirs/
└── msflow_wide_resnet50_2_avgpool_pl258/
    └── mvtec/
        └── bottle/
            ├── best.pt
            └── ...
```

## 性能指标

### MVTec AD 数据集
- 检测 AUROC: 99.7%
- 定位 AUROC: 98.8%
- PRO 分数: 97.1%

### VisA 数据集
- 检测 AUROC: 95.2%
- 定位 AUROC: 97.8%

## 常见问题

### 1. CUDA 内存不足
- 减小 `--batch-size` 参数
- 使用 `--amp_enable` 启用混合精度训练

### 2. FrEIA 安装失败
- 确保已安装 PyTorch
- 尝试使用 `pip install FrEIA` 或从源码安装

### 3. 数据集路径错误
- 检查 `default.py` 中的 `data_path` 设置
- 确保数据集目录结构正确

## 参考资料

- 论文: [MSFlow: Multi-Scale Flow-based Framework for Unsupervised Anomaly Detection](https://arxiv.org/pdf/2308.15300v1.pdf)
- GitHub: https://github.com/shiyemoxia/msflow
- FrEIA: https://github.com/VLL-HD/FrEIA

## 许可证

请参阅 LICENSE.md 文件了解详细信息。
