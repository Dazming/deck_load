import { Routes, Route, NavLink } from 'react-router-dom'
import { Cpu, BarChart3, Upload } from 'lucide-react'
import Architecture from './pages/Architecture'
import Showcase from './pages/Showcase'
import UploadPage from './pages/Upload'

const navItems = [
  { to: '/', icon: Cpu, label: '模型架构' },
  { to: '/showcase', icon: BarChart3, label: '结果展示' },
  { to: '/predict', icon: Upload, label: '在线预测' },
]

function Sidebar() {
  return (
    <aside className="fixed left-0 top-0 h-screen w-56 bg-[#161b22] border-r border-[#30363d] flex flex-col z-50">
      <div className="px-5 py-6 border-b border-[#30363d]">
        <h1 className="text-base font-bold text-[#00d4ff] tracking-wide">AMF-BiGRU</h1>
        <p className="text-xs text-[#8b949e] mt-1">移动载荷识别系统</p>
      </div>
      <nav className="flex-1 py-4 px-3 space-y-1">
        {navItems.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all duration-200 ${
                isActive
                  ? 'bg-[#00d4ff]/10 text-[#00d4ff] font-medium'
                  : 'text-[#8b949e] hover:text-[#e6edf3] hover:bg-[#1c2333]'
              }`
            }
          >
            <Icon size={18} />
            {label}
          </NavLink>
        ))}
      </nav>
      <div className="px-5 py-4 border-t border-[#30363d] text-xs text-[#484f58]">
        Deck Load Demo v1.0
      </div>
    </aside>
  )
}

export default function App() {
  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="ml-56 flex-1 p-8">
        <Routes>
          <Route path="/" element={<Architecture />} />
          <Route path="/showcase" element={<Showcase />} />
          <Route path="/predict" element={<UploadPage />} />
        </Routes>
      </main>
    </div>
  )
}
