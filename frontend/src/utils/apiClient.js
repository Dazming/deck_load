const API_FALLBACK_ORIGIN = 'http://127.0.0.1:5000'

export async function fetchApi(path, init) {
  let proxyRes = null
  try {
    proxyRes = await fetch(`/api${path}`, init)
  } catch {
    proxyRes = null
  }

  if (!proxyRes || [502, 503, 504].includes(proxyRes.status)) {
    try {
      return await fetch(`${API_FALLBACK_ORIGIN}/api${path}`, init)
    } catch (fallbackErr) {
      if (proxyRes) return proxyRes
      throw fallbackErr
    }
  }

  return proxyRes
}

export function getFriendlyApiError(errorMessage = '') {
  const msg = String(errorMessage || '')

  if (msg.includes('Failed to fetch')) {
    return '无法连接后端服务，请确认 API 已启动（http://127.0.0.1:5000）'
  }
  if (msg.includes('HTTP 400') || msg.includes('Missing columns') || msg.includes('SEQ_LEN')) {
    return `请求数据不符合接口要求：${msg}`
  }
  if (msg.includes('HTTP 404')) {
    return `未找到对应资源或工况：${msg}`
  }
  if (msg.includes('HTTP 502') || msg.includes('HTTP 503') || msg.includes('HTTP 504')) {
    return `前后端连接不稳定或服务未就绪：${msg}`
  }
  if (msg.includes('HTTP 500')) {
    return `后端内部错误，请查看后端日志：${msg}`
  }
  return msg || '请求失败，请检查后端状态与网络连接。'
}
