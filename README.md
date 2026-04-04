# AMF-BiGRU 移动载荷识别系统

基于注意力机制的多模态融合双向门控循环单元（AMF-BiGRU）的甲板结构移动载荷重构与定位系统。

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
├── api_server.py                       # Flask API 后端（模型推理服务）
├── requirements.txt                    # Python 依赖
├── 差异分析.txt                         # 结果差异分析
├── frontend/                           # React + Vite 前端仪表板
│   ├── src/
│   │   ├── App.jsx                     # 路由 + 侧边栏布局
│   │   ├── main.jsx                    # 入口
│   │   ├── index.css                   # 全局样式（暗色主题）
│   │   └── pages/
│   │       ├── Architecture.jsx        # 模型架构页（首页）
│   │       ├── Showcase.jsx            # 结果展示页（已有工况 true vs pred）
│   │       └── Upload.jsx              # 在线预测页（上传 CSV → 纯预测）
│   ├── package.json
│   └── vite.config.js                  # 含 API 代理配置
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

### 4. 启动前端仪表板

```bash
# 安装后端依赖
pip install flask flask-cors

# 启动 Flask API 后端（端口 5000）
python api_server.py

# 新终端 — 启动前端开发服务器
cd frontend
npm install
npm run dev
```

浏览器打开 Vite 输出的地址（默认 http://localhost:5173）即可访问仪表板。

前端共 3 个页面：

| 页面 | 路由 | 功能 |
|------|------|------|
| 模型架构 | `/` | AMF-BiGRU 网络结构图、数据处理流水线、超参数说明 |
| 结果展示 | `/showcase` | 选择已有工况（车重/车速），查看真实值 vs 预测值对比图及 RPE/R² 指标 |
| 在线预测 | `/predict` | 上传仅含传感器列（N1_UZ, N7_UZ, N1_AZ, N7_AZ）的 CSV，展示纯预测结果 |

## 测试结果

不同车重工况，2 测量点 (N1, N7)，测试集 w=45 kN：

| 目标变量 | RPE (%) | R² |
|---------|---------|------|
| 前轴重量 | 1.22 | 0.9991 |
| 后轴重量 | 1.19 | 0.9991 |
| 前轮位置 | 3.36 | 0.9970 |
| 后轮位置 | 3.43 | 0.9968 |

详细差异分析见 `差异分析.txt`。

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

