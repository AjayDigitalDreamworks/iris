import { NavLink, Outlet, useLocation, Link } from 'react-router-dom'
import { useEffect, useState } from 'react'
import {
  LayoutDashboard,
  Radio,
  Package,
  Layers,
  Users,
  Clock,
  Bell,
  Video,
  FileBarChart2,
  Cpu,
  Settings,
  Sun,
  Moon,
  Menu,
  X,
  ChevronLeft,
  ShieldCheck,
  ExternalLink,
} from 'lucide-react'
import { eventSocket } from '../services/ws'
import { useSocketState } from '../hooks/useAsync'
import AlertBanner from '../components/AlertBanner'

interface NavItem {
  to: string
  label: string
  icon: typeof LayoutDashboard
  end?: boolean
}

const NAV: NavItem[] = [
  { to: '/overview', label: 'Overview', icon: LayoutDashboard },
  { to: '/live', label: 'Live Monitoring', icon: Radio },
  { to: '/inventory', label: 'Inventory', icon: Package },
  { to: '/shelves', label: 'Shelves', icon: Layers },
  { to: '/customers', label: 'Customers', icon: Users },
  { to: '/queues', label: 'Queues', icon: Clock },
  { to: '/alerts', label: 'Alerts', icon: Bell },
  { to: '/cameras', label: 'Cameras', icon: Video },
  { to: '/reports', label: 'Reports', icon: FileBarChart2 },
  { to: '/models', label: 'Models', icon: Cpu },
  { to: '/settings', label: 'Settings', icon: Settings },
]

const SOCKET_LABEL: Record<string, string> = {
  open: 'Online',
  connecting: 'Connecting',
  reconnecting: 'Reconnecting',
  closed: 'Offline',
}

export default function AppLayout() {
  const socketState = useSocketState()
  const location = useLocation()
  const [dark, setDark] = useState(() => (localStorage.getItem('iris-theme') ?? 'light') === 'dark')
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [collapsed, setCollapsed] = useState(true)

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
    localStorage.setItem('iris-theme', dark ? 'dark' : 'light')
  }, [dark])

  useEffect(() => {
    eventSocket.connect()
    return () => eventSocket.disconnect()
  }, [])

  useEffect(() => {
    setMobileMenuOpen(false)
  }, [location.pathname])

  const currentNav = NAV.find((n) => (n.end ? location.pathname === n.to : location.pathname.startsWith(n.to)))

  const socketStyles = {
    open: {
      dot: 'bg-emerald-500',
      ping: 'bg-emerald-400',
      text: 'text-emerald-700 dark:text-emerald-400',
      border: 'border-emerald-500/20',
      bg: 'bg-emerald-500/10',
    },
    connecting: {
      dot: 'bg-amber-500',
      ping: 'bg-amber-400',
      text: 'text-amber-700 dark:text-amber-400',
      border: 'border-amber-500/20',
      bg: 'bg-amber-500/10',
    },
    reconnecting: {
      dot: 'bg-amber-500',
      ping: 'bg-amber-400',
      text: 'text-amber-700 dark:text-amber-400',
      border: 'border-amber-500/20',
      bg: 'bg-amber-500/10',
    },
    closed: {
      dot: 'bg-rose-500',
      ping: 'bg-rose-400',
      text: 'text-rose-700 dark:text-rose-400',
      border: 'border-rose-500/20',
      bg: 'bg-rose-500/10',
    },
  }[socketState] ?? {
    dot: 'bg-zinc-400',
    ping: 'bg-zinc-300',
    text: 'text-zinc-600',
    border: 'border-zinc-300',
    bg: 'bg-zinc-100',
  }

  return (
    <div className="flex h-screen max-h-screen w-screen overflow-hidden bg-[#f0f4f9] text-slate-800 transition-colors duration-150 dark:bg-[#090d16] dark:text-[#ededed] font-sans">
      {/* Precision Dark Navigation Rail (Leftmost rail from reference) */}
      <aside
        className={`hidden h-full shrink-0 flex-col bg-[#0e1726] border-r border-[#1e293b] text-slate-300 transition-all duration-200 lg:flex z-30 ${
          collapsed ? 'w-14' : 'w-56'
        }`}
      >
        {/* Rail Top Toggle */}
        <div className="flex h-13 shrink-0 items-center justify-between border-b border-white/10 px-2.5">
          <div className="flex items-center gap-2 overflow-hidden">
            <button
              type="button"
              onClick={() => setCollapsed((v) => !v)}
              className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg text-slate-400 hover:bg-white/10 hover:text-white transition"
              title={collapsed ? 'Expand menu' : 'Collapse menu'}
            >
              <Menu className="h-4 w-4" />
            </button>
            {!collapsed && (
              <span className="text-xs font-bold tracking-wider text-slate-200 uppercase truncate">
                Navigation
              </span>
            )}
          </div>
          {!collapsed && (
            <button
              type="button"
              onClick={() => setCollapsed(true)}
              className="text-slate-400 hover:text-white p-1"
            >
              <ChevronLeft className="h-4 w-4" />
            </button>
          )}
        </div>

        {/* Nav Icons */}
        <nav className="custom-scrollbar flex-1 space-y-1 overflow-y-auto px-2 py-3">
          {NAV.map((item) => {
            const Icon = item.icon
            return (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                title={collapsed ? item.label : undefined}
                className={({ isActive }) =>
                  `group relative flex items-center gap-3 rounded-lg py-2.5 text-xs font-medium transition-all duration-150 ${
                    collapsed ? 'justify-center px-0' : 'px-3'
                  } ${
                    isActive
                      ? 'bg-blue-600 text-white shadow-md shadow-blue-600/30 font-semibold'
                      : 'text-slate-400 hover:bg-white/5 hover:text-slate-100'
                  }`
                }
              >
                {({ isActive }) => (
                  <>
                    <Icon
                      className={`h-4 w-4 shrink-0 transition-transform group-hover:scale-105 ${
                        isActive ? 'text-white' : 'text-slate-400 group-hover:text-slate-200'
                      }`}
                    />
                    {!collapsed && <span className="truncate">{item.label}</span>}
                    {collapsed && isActive && (
                      <span className="absolute -left-2 top-1/2 -translate-y-1/2 h-5 w-1 rounded-r bg-blue-400" />
                    )}
                  </>
                )}
              </NavLink>
            )
          })}
        </nav>

        {/* Vision Status Dot */}
        <div className="shrink-0 border-t border-white/10 p-2.5">
          {!collapsed ? (
            <div className="flex items-center justify-between text-[11px] px-1">
              <span className="flex items-center gap-1.5 text-slate-400">
                <ShieldCheck className="h-3.5 w-3.5 text-blue-400" />
                Pipeline Active
              </span>
              <span className="font-semibold text-emerald-400">CPU</span>
            </div>
          ) : (
            <div className="flex justify-center" title="Vision Pipeline Active">
              <span className="relative flex h-2 w-2">
                <span className="absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75 animate-ping" />
                <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-500" />
              </span>
            </div>
          )}
        </div>
      </aside>

      {/* Main Viewport Shell */}
      <div className="flex h-full min-w-0 flex-1 flex-col overflow-hidden">
        {/* Top Professional Header (Exact match to reference top bar) */}
        <header className="flex h-13 shrink-0 items-center justify-between bg-white px-4 border-b border-slate-200/80 shadow-[0_1px_3px_rgba(0,0,0,0.03)] dark:bg-[#101726] dark:border-white/10 z-20">
          {/* Left Brand & Title */}
          <div className="flex items-center gap-3">
            {/* Mobile Drawer Trigger */}
            <div className="flex items-center gap-2 lg:hidden">
              <button
                type="button"
                onClick={() => setMobileMenuOpen((v) => !v)}
                aria-label="Toggle navigation drawer"
                className="flex h-8 w-8 items-center justify-center rounded-lg border border-slate-200 text-slate-700 dark:border-white/10 dark:text-slate-300"
              >
                {mobileMenuOpen ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
              </button>
            </div>

            {/* Real Official IRIS Horizontal Logo Lockup - links to Landing Page */}
            <Link to="/" className="flex items-center gap-2 group" title="Return to IRIS Product Home">
              <img
                src={dark ? "/iris-horizontal-dark.png" : "/iris-horizontal.png"}
                alt="IRIS - Intelligent Retail Insight System"
                className="h-8.5 w-auto object-contain drop-shadow-xs transition-transform group-hover:scale-102"
              />
            </Link>
            <span className="hidden sm:inline rounded bg-blue-500/10 px-1.5 py-0.5 text-[9px] font-bold text-blue-600 dark:text-blue-400 leading-none">
              v2.4
            </span>
            <div className="hidden sm:block h-4 w-px bg-slate-200 dark:bg-slate-800 mx-1" />
            {/* Store Selector & Breadcrumb Pill */}
            <div className="hidden md:flex items-center gap-1.5 rounded-md bg-slate-100/80 px-2 py-1 text-xs font-semibold text-slate-700 border border-slate-200/60 dark:bg-slate-800/80 dark:border-white/5 dark:text-slate-300">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
              <span>Store #01</span>
              <span className="text-slate-300 dark:text-slate-600">/</span>
              <span className="text-blue-600 dark:text-blue-400 font-bold">{currentNav?.label ?? 'Overview'}</span>
            </div>
          </div>

          {/* Center Telemetry Ticker (Hardware Authority) */}
          <div className="hidden xl:flex items-center gap-4 text-[11px] font-medium text-slate-500 dark:text-slate-400 tabular-nums">
            <div className="flex items-center gap-1.5">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
              <span>Engine: <strong className="text-slate-700 dark:text-slate-200 font-semibold">YOLO11n</strong></span>
            </div>
            <span className="text-slate-300 dark:text-slate-700">•</span>
            <div>
              FPS: <strong className="text-slate-700 dark:text-slate-200 font-semibold">30.0</strong>
            </div>
            <span className="text-slate-300 dark:text-slate-700">•</span>
            <div>
              Inference: <strong className="text-slate-700 dark:text-slate-200 font-semibold">18.4ms</strong>
            </div>
          </div>

          {/* Right Utilities (Notifications, Live Stream, Theme) */}
          <div className="flex items-center gap-2.5">
            {/* Quick Search Shortcut Pill */}
            <button
              type="button"
              className="hidden sm:flex items-center gap-2 rounded-lg border border-slate-200 bg-slate-50/80 px-2.5 py-1 text-xs text-slate-500 hover:border-slate-300 transition dark:border-white/10 dark:bg-slate-900/60 dark:text-slate-400"
              title="Search telemetry and SKUs (⌘K)"
            >
              <span>Quick search...</span>
              <kbd className="rounded bg-white px-1 py-0.5 text-[9px] font-bold text-slate-500 shadow-xs border border-slate-200 dark:bg-slate-800 dark:border-slate-700 dark:text-slate-400">
                ⌘K
              </kbd>
            </button>

            {/* Live Socket Status Badge */}
            <div
              className={`hidden sm:flex items-center gap-1.5 rounded-md border px-2 py-0.5 text-[11px] font-medium transition-colors ${socketStyles.bg} ${socketStyles.border} ${socketStyles.text}`}
              title="Real-time WebSocket event connection"
            >
              <span className="relative flex h-1.5 w-1.5">
                {socketState === 'open' && (
                  <span
                    className={`absolute inline-flex h-full w-full rounded-full opacity-75 animate-ping ${socketStyles.ping}`}
                  />
                )}
                <span className={`relative inline-flex h-1.5 w-1.5 rounded-full ${socketStyles.dot}`} />
              </span>
              <span>{SOCKET_LABEL[socketState]}</span>
            </div>

            {/* Notification Bell */}
            <NavLink
              to="/alerts"
              className="relative flex h-8 w-8 items-center justify-center rounded-lg text-slate-500 hover:bg-slate-100 hover:text-slate-800 transition dark:text-slate-400 dark:hover:bg-slate-800"
              title="Alerts"
            >
              <Bell className="h-4 w-4" />
              <span className="absolute top-1.5 right-1.5 flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-rose-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-rose-500" />
              </span>
            </NavLink>

            {/* Live Video Stream Quick Access */}
            <NavLink
              to="/live"
              className="flex items-center gap-1.5 rounded-lg bg-blue-50 px-2 py-1 text-xs font-semibold text-blue-600 hover:bg-blue-100 transition dark:bg-blue-950/40 dark:text-blue-400 dark:hover:bg-blue-900/50"
              title="Live Monitoring"
            >
              <Radio className="h-3.5 w-3.5 text-blue-600 dark:text-blue-400" />
              <span className="hidden md:inline">Feeds</span>
            </NavLink>

            {/* Product Home Landing Page Shortcut */}
            <Link
              to="/"
              className="hidden sm:flex items-center gap-1 rounded-lg border border-slate-200/80 bg-slate-50 px-2 py-1 text-xs font-semibold text-slate-700 hover:bg-white hover:text-blue-600 transition dark:border-white/10 dark:bg-slate-800/80 dark:text-slate-300 dark:hover:bg-slate-700"
              title="IRIS Product Overview & Marketing Page"
            >
              <ExternalLink className="h-3.5 w-3.5 text-slate-500 dark:text-slate-400" />
              <span className="hidden lg:inline">Home</span>
            </Link>

            {/* Theme Toggle */}
            <button
              type="button"
              onClick={() => setDark((v) => !v)}
              className="flex h-8 w-8 items-center justify-center rounded-lg border border-slate-200 text-slate-600 transition hover:bg-slate-100 active:scale-95 dark:border-white/10 dark:text-slate-400 dark:hover:bg-slate-800"
              aria-label="Toggle theme"
            >
              {dark ? <Sun className="h-4 w-4 text-amber-400" /> : <Moon className="h-4 w-4 text-slate-600" />}
            </button>
          </div>
        </header>

        {/* Mobile Navigation Modal */}
        {mobileMenuOpen && (
          <div className="fixed inset-0 z-50 lg:hidden">
            <div
              className="fixed inset-0 bg-black/60 backdrop-blur-sm"
              onClick={() => setMobileMenuOpen(false)}
            />
            <div className="surface-card fixed inset-y-0 left-0 z-50 flex w-64 flex-col p-4 shadow-xl">
              <div className="mb-4 flex items-center justify-between border-b border-slate-100 pb-3 dark:border-white/10">
                <div className="flex items-center gap-2">
                  <img src="/iris-eye.png" alt="IRIS" className="h-6 w-auto object-contain" />
                  <span className="text-sm font-black text-slate-900 dark:text-white">IRIS</span>
                </div>
                <button
                  type="button"
                  onClick={() => setMobileMenuOpen(false)}
                  className="rounded-md p-1 text-zinc-500 hover:bg-zinc-100 dark:hover:bg-zinc-800"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
              <nav className="custom-scrollbar flex-1 space-y-1 overflow-y-auto">
                <Link
                  to="/"
                  onClick={() => setMobileMenuOpen(false)}
                  className="flex items-center gap-2.5 rounded-lg px-3 py-2 text-xs font-semibold text-zinc-600 hover:bg-zinc-100 dark:text-zinc-300 dark:hover:bg-zinc-800"
                >
                  <ExternalLink className="h-4 w-4 text-blue-500" />
                  <span>Product Landing</span>
                </Link>
                {NAV.map((item) => {
                  const Icon = item.icon
                  return (
                    <NavLink
                      key={item.to}
                      to={item.to}
                      end={item.end}
                      className={({ isActive }) =>
                        `flex items-center gap-2.5 rounded-lg px-3 py-2 text-xs font-semibold ${
                          isActive
                            ? 'bg-indigo-600 text-white'
                            : 'text-zinc-600 hover:bg-zinc-100 dark:text-zinc-300 dark:hover:bg-zinc-800'
                        }`
                      }
                    >
                      <Icon className="h-4 w-4" />
                      <span>{item.label}</span>
                    </NavLink>
                  )
                })}
              </nav>
            </div>
          </div>
        )}

        {/* Viewport-Fitted Main Workspace */}
        <main className="flex flex-1 flex-col gap-2 overflow-hidden p-3 sm:p-4">
          {/* Critical-alert overlay: pinned above every page, auto-clears on resolve. */}
          <AlertBanner />
          <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  )
}
