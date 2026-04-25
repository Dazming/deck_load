import { Layers, GitBranch, Target, Zap, ArrowRight, Database, BarChart2 } from 'lucide-react'

function Card({ children, className = '' }) {
  return (
    <div className={`bg-[#161b22] border border-[#30363d] rounded-xl p-5 ${className}`}>
      {children}
    </div>
  )
}

function ParamRow({ label, value }) {
  return (
    <div className="flex justify-between py-2 border-b border-[#21262d] last:border-0">
      <span className="text-[#8b949e] text-sm">{label}</span>
      <span className="text-[#e6edf3] text-sm font-mono">{value}</span>
    </div>
  )
}

function FlowStep({ icon: Icon, title, subtitle, color, isLast = false }) {
  return (
    <div className="flex items-center">
      <div className="flex flex-col items-center text-center w-32">
        <div
          className="w-14 h-14 rounded-xl flex items-center justify-center mb-2 border"
          style={{ backgroundColor: color + '15', borderColor: color + '40' }}
        >
          <Icon size={24} style={{ color }} />
        </div>
        <p className="text-sm font-medium text-[#e6edf3]">{title}</p>
        <p className="text-xs text-[#8b949e] mt-0.5">{subtitle}</p>
      </div>
      {!isLast && (
        <ArrowRight size={20} className="text-[#30363d] mx-2 shrink-0" />
      )}
    </div>
  )
}

export default function Architecture() {
  return (
    <div className="max-w-6xl mx-auto space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-[#e6edf3]">模型架构</h1>
        <p className="text-[#8b949e] mt-2">
          AMF-BiGRU: 基于注意力机制的多模态融合双向门控循环单元，用于甲板结构移动载荷重构与定位（case1: 2点，case2: 7点）
        </p>
      </div>

      {/* Architecture flow */}
      <Card>
        <h2 className="text-base font-semibold text-[#e6edf3] mb-6">数据处理流水线</h2>
        <div className="flex items-center justify-center flex-wrap gap-y-4">
          <FlowStep icon={Database} title="传感器数据" subtitle="位移 + 加速度" color="#00d4ff" />
          <FlowStep icon={BarChart2} title="Z-Score 标准化" subtitle="零均值单位方差" color="#a78bfa" />
          <FlowStep icon={Layers} title="滑动窗口" subtitle={`窗口大小 = 7`} color="#fbbf24" />
          <FlowStep icon={GitBranch} title="双模态分支" subtitle="BiGRU 特征提取" color="#00b894" />
          <FlowStep icon={Zap} title="注意力融合" subtitle="自适应权重" color="#ff6b6b" />
          <FlowStep icon={Target} title="异常点修复" subtitle="按轴在甲板上约束" color="#a78bfa" />
          <FlowStep icon={Target} title="输出层" subtitle="4维预测" color="#00d4ff" isLast />
        </div>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* AMF-BiGRU architecture diagram */}
        <Card>
          <h2 className="text-base font-semibold text-[#e6edf3] mb-4">AMF-BiGRU 网络结构</h2>
          <div className="space-y-4">
            {/* Input */}
            <div className="flex gap-4">
              <div className="flex-1 bg-[#00d4ff]/10 border border-[#00d4ff]/30 rounded-lg p-3 text-center">
                <p className="text-xs text-[#00d4ff] font-medium">位移输入</p>
                <p className="text-xs text-[#8b949e] mt-1">(batch, 7, 2/7)</p>
                <p className="text-xs text-[#484f58]">case1: N1,N7 | case2: N1~N7</p>
              </div>
              <div className="flex-1 bg-[#ff6b6b]/10 border border-[#ff6b6b]/30 rounded-lg p-3 text-center">
                <p className="text-xs text-[#ff6b6b] font-medium">加速度输入</p>
                <p className="text-xs text-[#8b949e] mt-1">(batch, 7, 2/7)</p>
                <p className="text-xs text-[#484f58]">case1: N1,N7 | case2: N1~N7</p>
              </div>
            </div>

            {/* BiGRU Branches */}
            <div className="flex gap-4">
              <div className="flex-1 bg-[#00b894]/10 border border-[#00b894]/30 rounded-lg p-3">
                <p className="text-xs text-[#00b894] font-medium text-center">位移 BiGRU 分支</p>
                <div className="mt-2 space-y-1 text-xs text-[#8b949e]">
                  <p>BiGRU(in=2/7 → 32×2)</p>
                  <p>FC(64→64) + ReLU + Dropout</p>
                  <p>FC(64→32) + ReLU + Dropout</p>
                </div>
              </div>
              <div className="flex-1 bg-[#00b894]/10 border border-[#00b894]/30 rounded-lg p-3">
                <p className="text-xs text-[#00b894] font-medium text-center">加速度 BiGRU 分支</p>
                <div className="mt-2 space-y-1 text-xs text-[#8b949e]">
                  <p>BiGRU(in=2/7 → 32×2)</p>
                  <p>FC(64→64) + ReLU + Dropout</p>
                  <p>FC(64→32) + ReLU + Dropout</p>
                </div>
              </div>
            </div>

            {/* Attention Fusion */}
            <div className="bg-[#ffeaa7]/10 border border-[#ffeaa7]/30 rounded-lg p-3 text-center">
              <p className="text-xs text-[#ffeaa7] font-medium">注意力融合层 (Attention Fusion)</p>
              <p className="text-xs text-[#8b949e] mt-1">
                W·tanh(x) → softmax → 加权求和 → (batch, 32)
              </p>
            </div>

            {/* Output */}
            <div className="bg-[#a78bfa]/10 border border-[#a78bfa]/30 rounded-lg p-3 text-center">
              <p className="text-xs text-[#a78bfa] font-medium">输出层 Linear(32→4)</p>
              <p className="text-xs text-[#8b949e] mt-1">
                [前轴重量, 后轴重量, 前轮位置, 后轮位置]
              </p>
            </div>

            <div className="bg-[#0d1117] border border-[#30363d] rounded-lg p-3">
              <p className="text-xs text-[#e6edf3] font-medium">预测后处理（case1/case2 通用）</p>
              <div className="mt-2 space-y-1 text-xs text-[#8b949e]">
                <p>• 异常点检测 + 连续异常段插值修复</p>
                <p>• 按轴是否在甲板上分段（去抖动）</p>
                <p>• off-deck 强制位置归零（可选轴重归零）</p>
                <p>• on-deck 位置约束到 [0, 40] 且单调不减</p>
                <p>• 全局开关：shared/prediction_postprocess_hparams.py → ENABLE</p>
              </div>
            </div>
          </div>
        </Card>

        {/* Parameters */}
        <div className="space-y-6">
          <Card>
            <h2 className="text-base font-semibold text-[#e6edf3] mb-3">模型参数</h2>
            <ParamRow label="BiGRU 隐藏层维度" value="32 × 2" />
            <ParamRow label="FC1 维度" value="64" />
            <ParamRow label="FC2 / 融合维度" value="32" />
            <ParamRow label="输出维度" value="4" />
            <ParamRow label="Dropout" value="0.2" />
            <ParamRow label="滑动窗口大小" value="7" />
          </Card>

          <Card>
            <h2 className="text-base font-semibold text-[#e6edf3] mb-3">训练策略</h2>
            <ParamRow label="优化器" value="Adam" />
            <ParamRow label="初始学习率" value="0.005" />
            <ParamRow label="学习率调度" value="ReduceLROnPlateau" />
            <ParamRow label="调度 Patience" value="60" />
            <ParamRow label="调度 Factor" value="0.5" />
            <ParamRow label="最小学习率" value="1e-6" />
            <ParamRow label="梯度裁剪" value="1.0" />
            <ParamRow label="Batch Size" value="64" />
            <ParamRow label="最大 Epoch" value="3000" />
            <ParamRow label="早停 Patience" value="300" />
          </Card>
        </div>
      </div>
    </div>
  )
}
