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
├── shared/                             # 通用模块（跨 case 复用）
│   ├── model_arch.py                   # 模型结构
│   ├── data_pipeline.py                # 数据加载/标准化/滑窗
│   └── metrics.py                      # RPE / R²
├── cases/
│   ├── case1/                          # 不同车重
│   │   ├── config.py
│   │   ├── train_case1.py
│   │   ├── evaluate_case1.py
│   │   └── predict_video_case1.py
│   └── case2/                          # 不同车速
│       ├── config.py
│       ├── train_case2.py
│       ├── evaluate_case2.py
│       └── predict_video_case2.py
├── tools/
│   └── video_demos/                    # 演示色视频（合成数据）
│       ├── demo_video_speed.py
│       └── demo_video_alternating_noise.py
├── api_server.py                       # Flask API（当前接 case1）
├── frontend/                           # React + Vite 前端
├── dataset/
│   ├── different_weight/
│   └── different_speed/
├── checkpoints/
│   ├── case1/                          # case1 模型/评估/实时预测视频
│   └── case2/                          # case2 模型/评估/实时预测视频
├── requirements.txt
└── 差异分析.txt
```

## 快速开始

### 1) 前置条件

请先安装以下软件：

- Python 3.10+（推荐用 conda）
- Node.js 18+（建议 LTS 版本）
- npm（随 Node.js 安装）
- （可选）ffmpeg：如果你需要生成视频

### 2) 克隆项目

```bash
git clone <your-repo-url>
cd deck_load
```

### 3) 创建并安装 Python 环境（使用 conda 环境）

```bash
conda create -n test python=3.11 -y
conda activate test
pip install -r requirements.txt
pip install flask flask-cors
```

如果你需要视频生成功能，再安装 ffmpeg：

```bash
conda install -n test ffmpeg -y
```

### 4) 安装前端依赖

```bash
cd frontend
npm install
cd ..
```

### 5) 检查数据和模型文件

运行前请确认以下文件存在：

- 数据集目录：
  - `dataset/different_weight/train|val|test`（case1）
  - `dataset/different_speed/train|val|test`（case2）
- 模型权重（按 case 保存）：
  - `checkpoints/case1/best_model_case1.pth`
  - `checkpoints/case2/best_model_case2.pth`

如果 case1 权重不存在，请先训练：

```bash
conda activate test
python cases/case1/train_case1.py
```

## 运行项目

### Python case 入口（独立）

```bash
# case1: 不同车重
python cases/case1/train_case1.py
python cases/case1/evaluate_case1.py
python cases/case1/predict_video_case1.py

# case2: 不同车速
python cases/case2/train_case2.py
python cases/case2/evaluate_case2.py
python cases/case2/predict_video_case2.py

# 演示视频（合成数据）
python tools/video_demos/demo_video_speed.py
python tools/video_demos/demo_video_alternating_noise.py
```

### 1) 启动后端 API（终端 1）

```bash
conda activate test
python api_server.py
```

启动后可访问健康检查接口：

- [http://localhost:5000/api/health](http://localhost:5000/api/health)

### 2) 启动前端（终端 2）

```bash
cd frontend
npm run dev
```

打开终端输出的 Local 地址（通常是）：

- [http://localhost:5173](http://localhost:5173)

如果 5173 被占用，Vite 会自动切到 5174/5175，请以终端输出为准。

### 3) Windows 一键启动（推荐）

项目根目录提供了批处理脚本：

```bat
start_web.bat
```

它会自动：

- 启动后端（`conda run -n test python api_server.py`）
- 启动前端（`cd frontend && npm run dev`）
- 尝试打开浏览器 `http://localhost:5173`

注意：若 5173 被占用，前端会自动切换端口，请以前端终端输出的 Local 地址为准。

## 功能页面说明

前端共 3 个页面：

| 页面 | 路由 | 功能 |
|------|------|------|
| 模型架构 | `/` | AMF-BiGRU 网络结构图、数据处理流水线、超参数说明 |
| 结果展示 | `/showcase` | 选择已有工况（车重/车速），查看真实值 vs 预测值对比图及 RPE/R² 指标 |
| 在线预测 | `/predict` | 上传仅含传感器列（N1_UZ, N7_UZ, N1_AZ, N7_AZ）的 CSV，展示纯预测结果 |

## 常用脚本

```bash
# case1
python cases/case1/train_case1.py
python cases/case1/evaluate_case1.py
python cases/case1/predict_video_case1.py

# case2
python cases/case2/train_case2.py
python cases/case2/evaluate_case2.py
python cases/case2/predict_video_case2.py

# demos
python tools/video_demos/demo_video_speed.py
python tools/video_demos/demo_video_alternating_noise.py
```

## 兼容脚本映射（旧名 -> 新名）

| 旧脚本 | 新脚本 |
|---|---|
| `train.py` | `cases/case1/train_case1.py` |
| `evaluate.py` | `cases/case1/evaluate_case1.py` |
| `generate_video.py` | `cases/case1/predict_video_case1.py` |
| `generate_video_speed_demo.py` | `tools/video_demos/demo_video_speed.py` |
| `generate_video_alternating_demo.py` | `tools/video_demos/demo_video_alternating_noise.py` |

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

