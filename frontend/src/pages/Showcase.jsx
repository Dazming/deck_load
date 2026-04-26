import { useState, useEffect, useCallback } from 'react'
import { BarChart3, Loader2, AlertCircle } from 'lucide-react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer,
} from 'recharts'
import { fetchApi, getFriendlyApiError } from '../utils/apiClient'

function MetricBadge({ label, value, unit = '' }) {
  return (
    <div className="bg-[#0d1117] border border-[#30363d] rounded-lg px-2 py-2.5 text-center min-w-0">
      <p className="text-xs text-[#8b949e] mb-1">{label}</p>
      <p className="text-sm sm:text-base font-bold font-mono text-[#ffeaa7] break-all leading-tight">
        {value}
        {unit ? <span className="text-[10px] sm:text-xs text-[#8b949e] ml-0.5">{unit}</span> : null}
      </p>
    </div>
  )
}

const CHART_TITLES = {
  front_axle_wt: '前轴重量',
  rear_axle_wt: '后轴重量',
  front_wheel_pos: '前轮位置',
  rear_wheel_pos: '后轮位置',
}

const CHART_UNITS = {
  front_axle_wt: 'N',
  rear_axle_wt: 'N',
  front_wheel_pos: 'm',
  rear_wheel_pos: 'm',
}

async function parseJsonSafely(res) {
  const text = await res.text()
  if (!text) return {}
  try {
    return JSON.parse(text)
  } catch {
    return { raw: text }
  }
}

export default function Showcase() {
  const [conditions, setConditions] = useState([])
  const [caseName, setCaseName] = useState('case1')
  const [videos, setVideos] = useState([])
  const [postprocessEnabled, setPostprocessEnabled] = useState(true)
  const [weight, setWeight] = useState(45)
  const [speed, setSpeed] = useState(40)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetchApi('/conditions')
      .then((r) => r.json())
      .then(setConditions)
      .catch(() => { })
  }, [])

  useEffect(() => {
    fetchApi(`/videos?case=${caseName}`)
      .then((r) => r.json())
      .then((rows) => {
        if (Array.isArray(rows)) setVideos(rows)
        else setVideos([])
      })
      .catch(() => setVideos([]))
  }, [caseName])

  const caseConditions = conditions.filter((c) => c.case === caseName)
  const weights = [...new Set(caseConditions.map((c) => c.weight))].sort((a, b) => a - b)
  const speeds = [...new Set(caseConditions.map((c) => c.speed))].sort((a, b) => a - b)

  useEffect(() => {
    if (weights.length > 0 && !weights.includes(weight)) {
      setWeight(weights[0])
    }
    if (speeds.length > 0 && !speeds.includes(speed)) {
      setSpeed(speeds[0])
    }
  }, [weights, speeds, weight, speed])

  const handlePredict = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetchApi('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          case: caseName,
          weight,
          speed,
          postprocess_enable: postprocessEnabled,
        }),
      })
      const body = await parseJsonSafely(res)
      if (!res.ok) {
        throw new Error(body.error || body.raw || `HTTP ${res.status}`)
      }
      setResult(body)
    } catch (e) {
      setError(getFriendlyApiError(e?.message))
      setResult(null)
    } finally {
      setLoading(false)
    }
  }, [caseName, weight, speed, postprocessEnabled])

  const chartData = result
    ? result.times.map((t, i) => {
      const row = { time: parseFloat(t.toFixed(4)) }
      for (const col of Object.keys(result.series)) {
        row[`${col}_true`] = result.series[col].true[i]
        row[`${col}_pred`] = result.series[col].pred[i]
      }
      return row
    })
    : []

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-[#e6edf3]">结果展示</h1>
        <p className="text-[#8b949e] mt-2">选择 case 与工况（不同重量 / 不同速度），查看 AMF-BiGRU 模型的预测效果与真实值对比</p>
      </div>

      {/* Controls */}
      <div className="bg-[#161b22] border border-[#30363d] rounded-xl p-5">
        <div className="flex flex-wrap items-end gap-6">
          <div className="space-y-2">
            <label className="text-sm text-[#8b949e]">Case</label>
            <select
              value={caseName}
              onChange={(e) => setCaseName(e.target.value)}
              className="block w-44 bg-[#0d1117] border border-[#30363d] text-[#e6edf3] rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-[#00d4ff]"
            >
              <option value="case1">case1（不同重量）</option>
              <option value="case2">case2（不同速度）</option>
            </select>
          </div>

          <label className="flex items-center gap-2 text-sm text-[#8b949e] select-none">
            <input
              type="checkbox"
              checked={postprocessEnabled}
              onChange={(e) => setPostprocessEnabled(e.target.checked)}
              className="h-4 w-4 accent-[#00d4ff]"
            />
            启用预测后处理（异常点修复）
          </label>

          <div className="space-y-2">
            <label className="text-sm text-[#8b949e]">车重 (kN)</label>
            <select
              value={weight}
              onChange={(e) => setWeight(Number(e.target.value))}
              className="block w-40 bg-[#0d1117] border border-[#30363d] text-[#e6edf3] rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-[#00d4ff]"
            >
              {weights.length > 0
                ? weights.map((w) => (
                  <option key={w} value={w}>{w} kN</option>
                ))
                : [38, 40, 42, 44, 45, 46, 48, 50].map((w) => (
                  <option key={w} value={w}>{w} kN</option>
                ))}
            </select>
          </div>

          <div className="space-y-2">
            <label className="text-sm text-[#8b949e]">车速 (m/s)</label>
            <select
              value={speed}
              onChange={(e) => setSpeed(Number(e.target.value))}
              className="block w-40 bg-[#0d1117] border border-[#30363d] text-[#e6edf3] rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-[#00d4ff]"
            >
              {speeds.length > 0
                ? speeds.map((v) => (
                  <option key={v} value={v}>{v} m/s</option>
                ))
                : [40].map((v) => (
                  <option key={v} value={v}>{v} m/s</option>
                ))}
            </select>
          </div>

          <button
            onClick={handlePredict}
            disabled={loading}
            className="flex items-center gap-2 bg-[#00d4ff] hover:bg-[#00b8d9] disabled:opacity-50 text-[#0d1117] font-semibold px-5 py-2 rounded-lg text-sm transition-colors"
          >
            {loading ? <Loader2 size={16} className="animate-spin" /> : <BarChart3 size={16} />}
            {loading ? '加载中...' : '查看结果'}
          </button>
        </div>
      </div>

      {error && (
        <div className="bg-[#ff6b6b]/10 border border-[#ff6b6b]/30 rounded-xl p-4 flex items-center gap-3">
          <AlertCircle size={20} className="text-[#ff6b6b] shrink-0" />
          <p className="text-sm text-[#ff6b6b]">{error}</p>
        </div>
      )}

      {result && (
        <>
          <p className="text-xs text-[#8b949e]">
            后处理：
            <span className={`ml-1 font-mono ${result.postprocess_enabled ? 'text-[#00d4ff]' : 'text-[#ff6b6b]'}`}>
              {result.postprocess_enabled ? 'ON' : 'OFF'}
            </span>
          </p>
          {/* Metrics */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Object.entries(result.metrics).map(([col, m]) => (
              <div key={col} className="bg-[#161b22] border border-[#30363d] rounded-xl p-4">
                <p className="text-xs text-[#8b949e] mb-3">{CHART_TITLES[col]}</p>
                <div className="grid grid-cols-2 gap-2">
                  <MetricBadge label="RPE" value={m.rpe.toFixed(2)} unit="%" />
                  <MetricBadge label="R²" value={m.r2.toFixed(4)} />
                </div>
              </div>
            ))}
          </div>

          {/* Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {Object.keys(result.series).map((col) => (
              <div
                key={col}
                className="bg-[#161b22] border border-[#30363d] rounded-xl p-5"
              >
                <h3 className="text-sm font-medium text-[#e6edf3] mb-4">
                  {CHART_TITLES[col]} ({CHART_UNITS[col]})
                </h3>
                <div>
                  <ResponsiveContainer width="100%" height={268}>
                    <LineChart
                      data={chartData}
                      margin={{ top: 28, right: 8, left: 4, bottom: 8 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
                      <XAxis
                        dataKey="time"
                        tick={{ fill: '#8b949e', fontSize: 10 }}
                        tickMargin={10}
                        interval="preserveStartEnd"
                        minTickGap={28}
                      />
                      <YAxis tick={{ fill: '#8b949e', fontSize: 11 }} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#161b22',
                          border: '1px solid #30363d',
                          borderRadius: 8,
                          fontSize: 12,
                        }}
                        labelStyle={{ color: '#8b949e' }}
                      />
                      <Legend
                        verticalAlign="top"
                        align="right"
                        wrapperStyle={{ fontSize: 12, paddingBottom: 4 }}
                      />
                      <Line
                        type="monotone"
                        dataKey={`${col}_true`}
                        name="真实值"
                        stroke="#00d4ff"
                        dot={false}
                        strokeWidth={1.5}
                      />
                      <Line
                        type="monotone"
                        dataKey={`${col}_pred`}
                        name="预测值"
                        stroke="#ff6b6b"
                        dot={false}
                        strokeWidth={1.5}
                        strokeDasharray="5 3"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                  <p className="text-center text-[11px] text-[#8b949e] pt-1.5 pb-0.5">Time (s)</p>
                </div>
              </div>
            ))}
          </div>
        </>
      )}

      <div className="bg-[#161b22] border border-[#30363d] rounded-xl p-5">
        <h2 className="text-base font-semibold text-[#e6edf3] mb-3">
          测试集预测视频（{caseName}）
        </h2>
        <p className="text-xs text-[#8b949e] mb-3">
          说明：此处展示的是该 case 的测试集视频文件，不会随上方车重/车速选择而变化。
        </p>
        {videos.length === 0 ? (
          <p className="text-sm text-[#8b949e]">当前 case 暂无视频文件。</p>
        ) : (
          <div className="grid grid-cols-1 gap-4">
            {videos.map((v) => (
              <div key={v.url} className="space-y-2">
                <p className="text-xs text-[#8b949e]">{v.name}</p>
                <video
                  controls
                  preload="metadata"
                  className="w-full rounded-lg border border-[#30363d] bg-black"
                  src={v.url}
                />
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
