import { useState, useRef, useCallback, useEffect } from 'react'
import { Upload as UploadIcon, Loader2, AlertCircle, FileUp, X, History, Trash2 } from 'lucide-react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer,
} from 'recharts'
import { fetchApi, getFriendlyApiError } from '../utils/apiClient'

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

const CHART_COLORS = {
  front_axle_wt: '#00d4ff',
  rear_axle_wt: '#a78bfa',
  front_wheel_pos: '#00b894',
  rear_wheel_pos: '#fbbf24',
}

const HISTORY_KEY = 'deck_load_predict_history'
const MAX_HISTORY = 10
async function parseJsonSafely(res) {
  const text = await res.text()
  if (!text) return {}
  try {
    return JSON.parse(text)
  } catch {
    return { raw: text }
  }
}

function formatTime(iso) {
  try {
    const d = new Date(iso)
    return d.toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' })
  } catch {
    return ''
  }
}

export default function Upload() {
  const [file, setFile] = useState(null)
  const [result, setResult] = useState(null)
  const [activeLabel, setActiveLabel] = useState(null)
  const [activeHistoryId, setActiveHistoryId] = useState(null)
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const inputRef = useRef(null)

  useEffect(() => {
    try {
      const raw = localStorage.getItem(HISTORY_KEY)
      if (raw) setHistory(JSON.parse(raw))
    } catch {
      /* ignore */
    }
  }, [])

  const persistHistory = useCallback((items) => {
    setHistory(items)
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(items))
    } catch {
      /* ignore */
    }
  }, [])

  const handleFile = useCallback((f) => {
    if (f && f.name.endsWith('.csv')) {
      setFile(f)
      setError(null)
    } else {
      setError('请上传 .csv 格式的文件')
    }
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setDragOver(false)
    const f = e.dataTransfer.files[0]
    handleFile(f)
  }, [handleFile])

  const handleUpload = useCallback(async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    try {
      const formData = new FormData()
      formData.append('file', file)
      const res = await fetchApi('/upload_predict', {
        method: 'POST',
        body: formData,
      })
      const body = await parseJsonSafely(res)
      if (!res.ok) {
        throw new Error(body.error || body.raw || `HTTP ${res.status}`)
      }
      const data = body
      setResult(data)
      setActiveLabel(file.name)
      const entry = {
        id: Date.now(),
        fileName: file.name,
        at: new Date().toISOString(),
        result: data,
      }
      setActiveHistoryId(entry.id)
      setHistory((prev) => {
        const next = [entry, ...prev].slice(0, MAX_HISTORY)
        try {
          localStorage.setItem(HISTORY_KEY, JSON.stringify(next))
        } catch {
          /* ignore */
        }
        return next
      })
    } catch (e) {
      setError(getFriendlyApiError(e?.message))
      setResult(null)
    } finally {
      setLoading(false)
    }
  }, [file])

  const loadFromHistory = useCallback((entry) => {
    setResult(entry.result)
    setActiveLabel(entry.fileName)
    setActiveHistoryId(entry.id)
    setError(null)
  }, [])

  const removeHistoryItem = useCallback((id, e) => {
    e.stopPropagation()
    setHistory((prev) => {
      const next = prev.filter((h) => h.id !== id)
      try {
        localStorage.setItem(HISTORY_KEY, JSON.stringify(next))
      } catch {
        /* ignore */
      }
      return next
    })
    if (activeHistoryId === id) {
      setResult(null)
      setActiveLabel(null)
      setActiveHistoryId(null)
    }
  }, [activeHistoryId])

  const clearHistory = useCallback(() => {
    persistHistory([])
  }, [persistHistory])

  const clearFileAndResult = useCallback((e) => {
    e.stopPropagation()
    setFile(null)
    setResult(null)
    setActiveLabel(null)
    setActiveHistoryId(null)
  }, [])

  const chartData = result
    ? result.times.map((t, i) => {
        const row = { time: parseFloat(t.toFixed(4)) }
        for (const col of Object.keys(result.series)) {
          row[col] = result.series[col].pred[i]
        }
        return row
      })
    : []

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-[#e6edf3]">在线预测</h1>
        <p className="text-[#8b949e] mt-2">
          上传传感器数据 CSV 文件，AMF-BiGRU 模型将输出移动载荷的重量与位置预测
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Upload area */}
        <div className="lg:col-span-2 bg-[#161b22] border border-[#30363d] rounded-xl p-5 space-y-4">
          <div
            onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            onClick={() => inputRef.current?.click()}
            className={`border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-colors ${
              dragOver
                ? 'border-[#00d4ff] bg-[#00d4ff]/5'
                : 'border-[#30363d] hover:border-[#484f58] hover:bg-[#1c2333]'
            }`}
          >
            <input
              ref={inputRef}
              type="file"
              accept=".csv"
              className="hidden"
              onChange={(e) => handleFile(e.target.files[0])}
            />
            <FileUp size={40} className="mx-auto text-[#484f58] mb-3" />
            <p className="text-sm text-[#8b949e]">
              拖拽 CSV 文件到此处，或 <span className="text-[#00d4ff]">点击选择文件</span>
            </p>
            <p className="text-xs text-[#484f58] mt-2">
              CSV 需包含列: N1_UZ, N7_UZ, N1_AZ, N7_AZ
            </p>
          </div>

          {file && (
            <div className="flex items-center justify-between bg-[#0d1117] border border-[#30363d] rounded-lg px-4 py-3">
              <div className="flex items-center gap-3 min-w-0">
                <FileUp size={18} className="text-[#00d4ff] shrink-0" />
                <div className="min-w-0">
                  <p className="text-sm text-[#e6edf3] truncate">{file.name}</p>
                  <p className="text-xs text-[#484f58]">{(file.size / 1024).toFixed(1)} KB</p>
                </div>
              </div>
              <button
                type="button"
                onClick={clearFileAndResult}
                className="text-[#484f58] hover:text-[#ff6b6b] transition-colors shrink-0"
              >
                <X size={16} />
              </button>
            </div>
          )}

          <button
            type="button"
            onClick={handleUpload}
            disabled={!file || loading}
            className="flex items-center gap-2 bg-[#00d4ff] hover:bg-[#00b8d9] disabled:opacity-50 text-[#0d1117] font-semibold px-5 py-2 rounded-lg text-sm transition-colors"
          >
            {loading ? <Loader2 size={16} className="animate-spin" /> : <UploadIcon size={16} />}
            {loading ? '预测中...' : '上传并预测'}
          </button>
        </div>

        {/* History */}
        <div className="bg-[#161b22] border border-[#30363d] rounded-xl p-5 flex flex-col max-h-[420px]">
          <div className="flex items-center justify-between gap-2 mb-3">
            <h2 className="text-sm font-semibold text-[#e6edf3] flex items-center gap-2">
              <History size={16} className="text-[#00d4ff]" />
              预测历史
            </h2>
            {history.length > 0 && (
              <button
                type="button"
                onClick={clearHistory}
                className="text-xs text-[#8b949e] hover:text-[#ff6b6b] flex items-center gap-1"
              >
                <Trash2 size={12} />
                清空
              </button>
            )}
          </div>
          {history.length === 0 ? (
            <p className="text-xs text-[#484f58] flex-1">成功预测后将自动保存，最多保留 {MAX_HISTORY} 条</p>
          ) : (
            <ul className="space-y-2 overflow-y-auto flex-1 pr-1">
              {history.map((h) => (
                <li key={h.id}>
                  <div
                    role="button"
                    tabIndex={0}
                    onClick={() => loadFromHistory(h)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault()
                        loadFromHistory(h)
                      }
                    }}
                    className={`w-full text-left rounded-lg border px-3 py-2.5 text-xs transition-colors group ${
                      activeHistoryId === h.id
                        ? 'border-[#00d4ff]/50 bg-[#00d4ff]/10'
                        : 'border-[#30363d] bg-[#0d1117] hover:border-[#484f58]'
                    }`}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="min-w-0 flex-1">
                        <p className="text-[#e6edf3] font-medium truncate">{h.fileName}</p>
                        <p className="text-[#484f58] mt-0.5">{formatTime(h.at)}</p>
                      </div>
                      <button
                        type="button"
                        onClick={(e) => removeHistoryItem(h.id, e)}
                        className="text-[#484f58] hover:text-[#ff6b6b] p-0.5 opacity-0 group-hover:opacity-100"
                        title="删除"
                      >
                        <X size={14} />
                      </button>
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      {error && (
        <div className="bg-[#ff6b6b]/10 border border-[#ff6b6b]/30 rounded-xl p-4 flex items-center gap-3">
          <AlertCircle size={20} className="text-[#ff6b6b] shrink-0" />
          <p className="text-sm text-[#ff6b6b]">{error}</p>
        </div>
      )}

      {result && (
        <div className="space-y-3">
          {activeLabel && (
            <p className="text-xs text-[#8b949e]">
              当前结果：<span className="text-[#e6edf3] font-mono">{activeLabel}</span>
            </p>
          )}
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
                      margin={{ top: 8, right: 8, left: 4, bottom: 8 }}
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
                      <Line
                        type="monotone"
                        dataKey={col}
                        name={`${CHART_TITLES[col]} 预测值`}
                        stroke={CHART_COLORS[col]}
                        dot={false}
                        strokeWidth={2}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                  <p className="text-center text-[11px] text-[#8b949e] pt-1.5 pb-0.5">Time (s)</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
