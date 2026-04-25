# AMF-BiGRU 移动载荷识别系统

基于注意力机制的多模态融合双向门控循环单元（AMF-BiGRU）的甲板结构移动载荷重构与定位系统。

## 方法概述

AMF-BiGRU 模型通过融合甲板结构的**位移**和**加速度**双模态响应，实现移动载荷的重建（前后轴重量）与定位（前后轮位置）。

当前项目采用统一模型 + 双 case：
- **case1（不同重量）**：使用 2 个测点（N1、N7）
- **case2（不同速度）**：使用 7 个测点（N1~N7）

```
位移响应 (case1: N1/N7, case2: N1~N7) → BiGRU → FC → ReLU → Dropout → FC → ReLU → Dropout ─┐
                                                                                               ├→ Attention Fusion → Output (4维)
加速度响应 (case1: N1/N7, case2: N1~N7) → BiGRU → FC → ReLU → Dropout → FC → ReLU → Dropout ─┘
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
├── api_server.py                       # Flask API（支持 case1/case2 选择）
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
| 模型架构 | `/` | AMF-BiGRU 网络结构图、数据处理流水线、超参数说明（含 case1/2 输入点差异） |
| 结果展示 | `/showcase` | 先选 case，再选工况（车重/车速），查看真实值 vs 预测值对比图及 RPE/R²；页面下方展示该 case 的测试集预测视频 |
| 在线预测 | `/predict` | 先选 case，再上传 CSV：case1 需 2 点列（N1/N7），case2 需 7 点列（N1~N7），展示纯预测结果 |

## 预测后处理（异常点修复）

为减少局部尖刺并避免误伤真实物理跳变，项目在预测输出后增加了通用后处理模块：

- 位置：`shared/prediction_smoothing.py`
- 参数：`shared/prediction_postprocess_hparams.py`（case1/case2 共用）

当前后处理包含：

1. Hampel/MAD 异常点检测（支持连续异常段修复）
2. 基于轴重阈值的 on-deck / off-deck 分段处理
3. 边界保护（不在真实跳变边缘过度修复）
4. 物理约束投影：
   - 轴不在甲板上时，位置强制为 0（可选：轴重也强制为 0）
   - 轴在甲板上时，位置限制在 `[0, deck_length]` 且单调不减

### 全局开关与常用参数

在 `shared/prediction_postprocess_hparams.py`：

- `ENABLE`：是否启用异常点修复（全局开关，case1/case2 同时生效）
- `FORCE_ZERO_OFFDECK`：轴不在甲板上时是否强制轴重归零
- `AXLE_MASK_MIN_RUN`：轴在甲板上状态最短连续长度（去抖动）
- `MEDIAN_KERNEL` / `DESPIKE_NSIGMA`：异常点检测强度
- `POS_VEL_OUTLIER_NSIGMA` / `POS_FIX_MAX_PASSES`：位置局部折点修复强度
- `EVAL_USE_SMOOTH_FOR_METRICS`：是否用修复后结果计算 RPE/R²
- `EVAL_PLOT_SMOOTHED`：是否在图上显示修复后曲线

说明：后处理默认对 case1/case2 使用同一套参数；如需区分，可在各 case 的 `config.py` 做覆盖。

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

case1（不同车重）在 2 测量点 (N1, N7) 配置下，测试集 w=45 kN：

| 目标变量 | RPE (%) | R² |
|---------|---------|------|
| 前轴重量 | 1.22 | 0.9991 |
| 后轴重量 | 1.19 | 0.9991 |
| 前轮位置 | 3.36 | 0.9970 |
| 后轮位置 | 3.43 | 0.9968 |

详细差异分析见 `差异分析.txt`。

## 数据格式

CSV 文件列要求按 case 而不同（顺序不限，按列名匹配）：

- **case1（2 点）必需输入列**：
  - `N1_UZ`, `N7_UZ`, `N1_AZ`, `N7_AZ`
- **case2（7 点）必需输入列**：
  - `N1_UZ` ~ `N7_UZ`
  - `N1_AZ` ~ `N7_AZ`

离线训练/评估数据通常还包含以下目标列（在线预测可不含）：

| 列名 | 含义 |
|------|------|
| TIME | 时间 (s) |
| N*_UZ / N*_AZ | 传感器位移 / 加速度响应（* 取决于 case 的点位配置） |
| front_wheel_pos | 前轮位置 (m) |
| rear_wheel_pos | 后轮位置 (m) |
| front_axle_wt | 前轴重量 (N) |
| rear_axle_wt | 后轴重量 (N) |

