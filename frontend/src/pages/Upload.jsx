import { useState, useRef, useCallback } from 'react'
import { Upload as UploadIcon, Loader2, AlertCircle, FileUp, X } from 'lucide-react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer,
} from 'recharts'

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

export default function Upload() {
  const [file, setFile] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const inputRef = useRef(null)

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
      const res = await fetch('/api/upload_predict', {
        method: 'POST',
        body: formData,
      })
      if (!res.ok) {
        const body = await res.json()
        throw new Error(body.error || `HTTP ${res.status}`)
      }
      setResult(await res.json())
    } catch (e) {
      setError(e.message)
      setResult(null)
    } finally {
      setLoading(false)
    }
  }, [file])

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

      {/* Upload area */}
      <div className="bg-[#161b22] border border-[#30363d] rounded-xl p-5 space-y-4">
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
            <div className="flex items-center gap-3">
              <FileUp size={18} className="text-[#00d4ff]" />
              <div>
                <p className="text-sm text-[#e6edf3]">{file.name}</p>
                <p className="text-xs text-[#484f58]">{(file.size / 1024).toFixed(1)} KB</p>
              </div>
            </div>
            <button
              onClick={(e) => { e.stopPropagation(); setFile(null); setResult(null) }}
              className="text-[#484f58] hover:text-[#ff6b6b] transition-colors"
            >
              <X size={16} />
            </button>
          </div>
        )}

        <button
          onClick={handleUpload}
          disabled={!file || loading}
          className="flex items-center gap-2 bg-[#00d4ff] hover:bg-[#00b8d9] disabled:opacity-50 text-[#0d1117] font-semibold px-5 py-2 rounded-lg text-sm transition-colors"
        >
          {loading ? <Loader2 size={16} className="animate-spin" /> : <UploadIcon size={16} />}
          {loading ? '预测中...' : '上传并预测'}
        </button>
      </div>

      {error && (
        <div className="bg-[#ff6b6b]/10 border border-[#ff6b6b]/30 rounded-xl p-4 flex items-center gap-3">
          <AlertCircle size={20} className="text-[#ff6b6b] shrink-0" />
          <p className="text-sm text-[#ff6b6b]">{error}</p>
        </div>
      )}

      {result && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {Object.keys(result.series).map((col) => (
            <div
              key={col}
              className="bg-[#161b22] border border-[#30363d] rounded-xl p-5"
            >
              <h3 className="text-sm font-medium text-[#e6edf3] mb-4">
                {CHART_TITLES[col]} ({CHART_UNITS[col]})
              </h3>
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
                  <XAxis
                    dataKey="time"
                    tick={{ fill: '#8b949e', fontSize: 11 }}
                    label={{ value: 'Time (s)', position: 'insideBottom', offset: -4, fill: '#8b949e', fontSize: 11 }}
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
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
