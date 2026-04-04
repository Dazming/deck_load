# AMF-BiGRU 移动载荷识别模型复现

基于论文 **"Reconstruction and localization of moving load on deck structures based on attention mechanism-based multi-modal fusion and bidirectional gated recurrent unit"** (Liu et al., *Engineering Applications of Artificial Intelligence*, Vol.164, 2026) 的复现项目。

## 方法概述

AMF-BiGRU 模型通过融合甲板结构的**位移**和**加速度**双模态响应，实现移动载荷的重建（前后轴重量）与定位（前后轮位置）。

```
位移响应 (N1_UZ, N7_UZ) → BiGRU → FC → ReLU → Dropout → FC → ReLU → Dropout ─┐
                                                                                 ├→ Attention Fusion → Output (4维)
加速度响应 (N1_AZ, N7_AZ) → BiGRU → FC → ReLU → Dropout → FC → ReLU → Dropout ─┘
```

**输出**：前轴重量、后轴重量、前轮位置、后轮位置

## 项目结构

```
deck_load/
├── config.py                           # 超参数配置
├── data_loader.py                      # 数据加载、Z-score标准化、滑动窗口采样
├── model.py                            # AMF-BiGRU 模型定义
├── train.py                            # 训练脚本（含早停、学习率调度）
├── evaluate.py                         # 测试集评估（RPE/R²）与可视化
├── generate_video.py                   # 实时预测演示视频（基于真实模型推理）
├── generate_video_speed_demo.py        # 不同车速演示视频（合成数据）
├── generate_video_alternating_demo.py  # 交变载荷+噪声演示视频（合成数据）
├── requirements.txt                    # Python 依赖
├── 差异分析.txt                         # 复现结果与论文的详细差异对比
├── dataset/
│   └── different_weight/
│       ├── train/   (w40, w42, w44, w46, w48, w50 @ v=40 m/s)
│       ├── val/     (w38 @ v=40 m/s)
│       └── test/    (w45 @ v=40 m/s)
└── checkpoints/
    ├── best_model.pth                  # 训练好的最佳模型
    ├── test_results.png                # 评估结果图
    ├── prediction_realtime.mp4         # 不同车重实时预测视频
    ├── demo_speed_30.mp4               # 不同车速演示视频
    └── demo_alternating_noise1.mp4     # 交变载荷+1%噪声演示视频
```

## 环境配置

使用 conda `test` 环境：

```bash
conda activate test
pip install -r requirements.txt
```

依赖：PyTorch >= 2.0、NumPy、Pandas、Matplotlib

视频生成额外需要 ffmpeg：

```bash
conda install ffmpeg
```

## 使用方法

### 1. 训练模型

```bash
python train.py
```

主要超参数（详见 `config.py`）：

| 参数 | 值 |
|------|-----|
| 滑动窗口大小 (s) | 7 |
| BiGRU 隐藏维度 | 32 |
| FC 层维度 | 64 → 32 |
| 学习率 | 0.005（ReduceLROnPlateau 自动衰减） |
| Batch Size | 64 |
| 最大 Epoch | 3000 |
| 早停 Patience | 300 |
| Dropout | 0.2 |

### 2. 评估模型

```bash
python evaluate.py
```

输出每个目标变量的 RPE 和 R² 指标，并生成预测对比图。

### 3. 生成视频

```bash
# 不同车重（基于真实模型推理）
python generate_video.py

# 不同车速（合成数据演示，v=30 m/s）
python generate_video_speed_demo.py

# 交变载荷 + 1% 噪声（合成数据演示）
python generate_video_alternating_demo.py
```

视频以 10 倍慢放展示实时预测过程，包含甲板俯视图和时序曲线对比。

## 复现结果

不同车重工况，2 测量点 (N1, N7)，测试集 w=45 kN：

| 目标变量 | RPE (%) | R² |
|---------|---------|------|
| 前轴重量 | 1.22 | 0.9991 |
| 后轴重量 | 1.19 | 0.9991 |
| 前轮位置 | 3.36 | 0.9970 |
| 后轮位置 | 3.43 | 0.9968 |

论文报告值：RPE < 0.45%，R² > 0.9999。差异分析详见 `差异分析.txt`。

## 数据格式

CSV 文件，每文件 1200 行（+ 表头），列定义：

| 列名 | 含义 |
|------|------|
| TIME | 时间 (s) |
| N1_UZ | 测量点 N1 位移响应 |
| N1_AZ | 测量点 N1 加速度响应 |
| N7_UZ | 测量点 N7 位移响应 |
| N7_AZ | 测量点 N7 加速度响应 |
| front_wheel_pos | 前轮位置 (m) |
| rear_wheel_pos | 后轮位置 (m) |
| front_axle_wt | 前轴重量 (N) |
| rear_axle_wt | 后轴重量 (N) |

## 参考文献

Liu, Y., Quan, B., Ke, W., Ren, W., Liu, Z., & Liu, H. (2026). Reconstruction and localization of moving load on deck structures based on attention mechanism-based multi-modal fusion and bidirectional gated recurrent unit. *Engineering Applications of Artificial Intelligence*, 164, 113394.
