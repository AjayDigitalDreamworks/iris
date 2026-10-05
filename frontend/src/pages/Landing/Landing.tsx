import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import {
  ArrowRight,
  Radio,
  ShieldCheck,
  Zap,
  Eye,
  Layers,
  Users,
  Clock,
  Cpu,
  Server,
  ChevronRight,
  Sun,
  Moon,
  CheckCircle2,
  Lock,
  Activity,
  ArrowUpRight,
} from 'lucide-react'
import { api } from '../../services/api'
import { useApi } from '../../hooks/useAsync'

export default function Landing() {
  const [dark, setDark] = useState(() => (localStorage.getItem('iris-theme') ?? 'light') === 'dark')
  const [activeTab, setActiveTab] = useState<'overview' | 'live' | 'shelves' | 'queues'>('overview')
  const health = useApi(() => api.health(), [])

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
    localStorage.setItem('iris-theme', dark ? 'dark' : 'light')
  }, [dark])

  const activeWorkers = (health.data?.metrics as Record<string, number>)?.active_workers ?? 9

  return (
    <div className="min-h-screen w-full overflow-x-hidden bg-[#f8fafc] text-slate-900 transition-colors duration-200 dark:bg-[#090d16] dark:text-slate-100 font-sans selection:bg-blue-500/20 selection:text-blue-600">
      {/* ───────────────────────────────────────────────────────────── */}
      {/* 1. TOP FLOATING NAVIGATION BAR                                 */}
      {/* ───────────────────────────────────────────────────────────── */}
      <header className="sticky top-0 z-50 w-full border-b border-slate-200/80 bg-white/80 backdrop-blur-md dark:border-white/10 dark:bg-[#090d16]/80 transition-colors">
        <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
          {/* Brand Logo */}
          <Link to="/" className="flex items-center gap-3 group">
            <img
              src={dark ? '/iris-horizontal-dark.png' : '/iris-horizontal.png'}
              alt="IRIS - Intelligent Retail Insight System"
              className="h-8.5 w-auto object-contain transition-transform group-hover:scale-102"
            />
            <span className="rounded-full bg-blue-500/10 px-2 py-0.5 text-[10px] font-bold text-blue-600 dark:text-blue-400 border border-blue-500/20">
              v2.4
            </span>
          </Link>

          {/* Navigation Links */}
          <nav className="hidden md:flex items-center gap-7 text-xs font-semibold text-slate-600 dark:text-slate-300">
            <a href="#features" className="hover:text-blue-600 dark:hover:text-blue-400 transition">
              Capabilities
            </a>
            <a href="#architecture" className="hover:text-blue-600 dark:hover:text-blue-400 transition">
              Architecture
            </a>
            <a href="#privacy" className="hover:text-blue-600 dark:hover:text-blue-400 transition">
              Edge Privacy
            </a>
            <a href="#hardware" className="hover:text-blue-600 dark:hover:text-blue-400 transition">
              Hardware
            </a>
          </nav>

          {/* Action CTAs */}
          <div className="flex items-center gap-3">
            {/* Theme Toggle */}
            <button
              type="button"
              onClick={() => setDark((v) => !v)}
              className="flex h-9 w-9 items-center justify-center rounded-lg border border-slate-200 text-slate-600 transition hover:bg-slate-100 active:scale-95 dark:border-white/10 dark:text-slate-400 dark:hover:bg-slate-800"
              aria-label="Toggle theme"
            >
              {dark ? <Sun className="h-4 w-4 text-amber-400" /> : <Moon className="h-4 w-4 text-slate-600" />}
            </button>

            {/* Live Streams Secondary Link */}
            <Link
              to="/live"
              className="hidden sm:flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3.5 py-2 text-xs font-bold text-slate-700 shadow-2xs hover:bg-slate-50 transition active:scale-98 dark:border-white/10 dark:bg-slate-900 dark:text-slate-200 dark:hover:bg-slate-800"
            >
              <Radio className="h-3.5 w-3.5 text-blue-600 animate-pulse" />
              <span>Live Streams</span>
            </Link>

            {/* Launch Dashboard Primary Button */}
            <Link
              to="/overview"
              className="flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-xs font-bold text-white shadow-sm shadow-blue-500/25 transition hover:bg-blue-500 active:scale-98"
            >
              <span>Launch Console</span>
              <ArrowRight className="h-3.5 w-3.5" />
            </Link>
          </div>
        </div>
      </header>

      {/* ───────────────────────────────────────────────────────────── */}
      {/* 2. HERO SECTION                                                */}
      {/* ───────────────────────────────────────────────────────────── */}
      <section className="relative overflow-hidden pt-12 pb-20 sm:pt-16 sm:pb-28">
        {/* Subtle background ambient light */}
        <div className="pointer-events-none absolute -top-40 left-1/2 -translate-x-1/2 h-[500px] w-[900px] rounded-full bg-gradient-to-tr from-blue-500/10 via-sky-400/15 to-emerald-400/10 blur-3xl dark:from-blue-600/15 dark:via-cyan-500/10 dark:to-transparent" />

        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 text-center relative z-10">
          {/* Eyebrow Announcement Pill */}
          <div className="inline-flex items-center gap-2 rounded-full border border-blue-500/20 bg-blue-50/80 px-3.5 py-1 text-xs font-semibold text-blue-700 backdrop-blur-xs transition hover:bg-blue-100 dark:border-blue-500/30 dark:bg-blue-950/40 dark:text-blue-300">
            <span className="flex h-1.5 w-1.5 rounded-full bg-blue-600 dark:bg-blue-400 animate-ping" />
            <span>Autonomous Vision Intelligence for Physical Stores</span>
            <ChevronRight className="h-3.5 w-3.5 text-blue-400" />
          </div>

          {/* Main Hero Headline */}
          <h1 className="mt-6 text-4xl font-extrabold tracking-tight sm:text-6xl sm:leading-[1.1] max-w-4xl mx-auto">
            See Every Footstep.{' '}
            <span className="bg-gradient-to-r from-blue-600 via-sky-500 to-emerald-500 bg-clip-text text-transparent">
              Optimize Every Shelf.
            </span>{' '}
            Zero Cloud.
          </h1>

          {/* Subtitle */}
          <p className="mt-5 text-base sm:text-lg leading-relaxed text-slate-600 dark:text-slate-300 max-w-2xl mx-auto font-normal">
            IRIS is an edge-native retail computer vision platform powered by YOLO11 and ByteTrack.
            Transform standard RTSP camera feeds into live footfall analytics, queue wait times, and autonomous
            stock depletion alerts—100% on-premises with absolute customer privacy.
          </p>

          {/* Hero CTAs */}
          <div className="mt-8 flex flex-col sm:flex-row items-center justify-center gap-3.5">
            <Link
              to="/overview"
              className="flex w-full sm:w-auto items-center justify-center gap-2.5 rounded-xl bg-blue-600 px-6 py-3.5 text-sm font-bold text-white shadow-md shadow-blue-500/30 transition hover:bg-blue-500 active:scale-98"
            >
              <span>Open Executive Dashboard</span>
              <ArrowRight className="h-4 w-4" />
            </Link>
            <Link
              to="/live"
              className="flex w-full sm:w-auto items-center justify-center gap-2.5 rounded-xl border border-slate-200 bg-white px-6 py-3.5 text-sm font-bold text-slate-800 shadow-2xs hover:bg-slate-50 transition active:scale-98 dark:border-white/10 dark:bg-[#101726] dark:text-slate-100 dark:hover:bg-slate-800"
            >
              <Radio className="h-4 w-4 text-blue-600 dark:text-blue-400" />
              <span>Watch Live Inference Streams</span>
            </Link>
          </div>

          {/* Enterprise Metric Highlights Ticker */}
          <div className="mt-12 grid grid-cols-2 gap-3 sm:grid-cols-4 max-w-3xl mx-auto text-left">
            <div className="rounded-lg border border-slate-200/80 bg-white/70 p-3 shadow-2xs backdrop-blur-xs dark:border-white/5 dark:bg-[#101726]/60">
              <div className="flex items-center gap-1.5 text-slate-400 text-[11px] font-semibold uppercase tracking-wider">
                <Zap className="h-3 w-3 text-amber-500" />
                <span>Inference</span>
              </div>
              <div className="mt-1 text-xl font-black text-slate-900 dark:text-white tabular-nums">
                18.4ms
              </div>
              <div className="text-[10px] text-slate-500">Real-time edge speed</div>
            </div>

            <div className="rounded-lg border border-slate-200/80 bg-white/70 p-3 shadow-2xs backdrop-blur-xs dark:border-white/5 dark:bg-[#101726]/60">
              <div className="flex items-center gap-1.5 text-slate-400 text-[11px] font-semibold uppercase tracking-wider">
                <Lock className="h-3 w-3 text-emerald-500" />
                <span>Privacy</span>
              </div>
              <div className="mt-1 text-xl font-black text-slate-900 dark:text-white">
                100% Local
              </div>
              <div className="text-[10px] text-slate-500">Zero face/biometric storage</div>
            </div>

            <div className="rounded-lg border border-slate-200/80 bg-white/70 p-3 shadow-2xs backdrop-blur-xs dark:border-white/5 dark:bg-[#101726]/60">
              <div className="flex items-center gap-1.5 text-slate-400 text-[11px] font-semibold uppercase tracking-wider">
                <Eye className="h-3 w-3 text-blue-500" />
                <span>Model</span>
              </div>
              <div className="mt-1 text-xl font-black text-slate-900 dark:text-white">
                YOLO11
              </div>
              <div className="text-[10px] text-slate-500">Ultralytics object vision</div>
            </div>

            <div className="rounded-lg border border-slate-200/80 bg-white/70 p-3 shadow-2xs backdrop-blur-xs dark:border-white/5 dark:bg-[#101726]/60">
              <div className="flex items-center gap-1.5 text-slate-400 text-[11px] font-semibold uppercase tracking-wider">
                <Server className="h-3 w-3 text-sky-500" />
                <span>Pipeline</span>
              </div>
              <div className="mt-1 text-xl font-black text-slate-900 dark:text-white tabular-nums">
                {activeWorkers} Streams
              </div>
              <div className="text-[10px] text-slate-500">Decoupled queue buffers</div>
            </div>
          </div>
        </div>

        {/* ───────────────────────────────────────────────────────────── */}
        {/* 3. INTERACTIVE HERO APP DEMO SHOWCASE                         */}
        {/* ───────────────────────────────────────────────────────────── */}
        <div className="mx-auto mt-14 max-w-6xl px-4 sm:px-6 lg:px-8">
          <div className="rounded-2xl border border-slate-200/90 bg-white shadow-xl overflow-hidden dark:border-white/10 dark:bg-[#101726]">
            {/* Mock Window Controls Top Bar */}
            <div className="flex items-center justify-between border-b border-slate-200/80 bg-slate-50/90 px-4 py-3 dark:border-white/10 dark:bg-slate-900/80">
              <div className="flex items-center gap-2">
                <span className="h-3 w-3 rounded-full bg-rose-400" />
                <span className="h-3 w-3 rounded-full bg-amber-400" />
                <span className="h-3 w-3 rounded-full bg-emerald-400" />
                <span className="ml-2 text-xs font-bold text-slate-500 dark:text-slate-400 font-mono">
                  iris-console://store-01.local
                </span>
              </div>

              {/* View Switcher Pills */}
              <div className="flex items-center gap-1 rounded-lg bg-slate-200/60 p-0.5 dark:bg-slate-800">
                <button
                  type="button"
                  onClick={() => setActiveTab('overview')}
                  className={`rounded-md px-2.5 py-1 text-[11px] font-bold transition ${
                    activeTab === 'overview'
                      ? 'bg-white text-blue-600 shadow-2xs dark:bg-slate-700 dark:text-white'
                      : 'text-slate-600 dark:text-slate-400 hover:text-slate-900'
                  }`}
                >
                  Overview Matrix
                </button>
                <button
                  type="button"
                  onClick={() => setActiveTab('live')}
                  className={`rounded-md px-2.5 py-1 text-[11px] font-bold transition ${
                    activeTab === 'live'
                      ? 'bg-white text-blue-600 shadow-2xs dark:bg-slate-700 dark:text-white'
                      : 'text-slate-600 dark:text-slate-400 hover:text-slate-900'
                  }`}
                >
                  Live Vision Feed
                </button>
                <button
                  type="button"
                  onClick={() => setActiveTab('shelves')}
                  className={`rounded-md px-2.5 py-1 text-[11px] font-bold transition ${
                    activeTab === 'shelves'
                      ? 'bg-white text-blue-600 shadow-2xs dark:bg-slate-700 dark:text-white'
                      : 'text-slate-600 dark:text-slate-400 hover:text-slate-900'
                  }`}
                >
                  Floor Heatmap
                </button>
              </div>

              <Link
                to="/overview"
                className="flex items-center gap-1 text-xs font-semibold text-blue-600 hover:text-blue-500 dark:text-blue-400"
              >
                <span>Full Console</span>
                <ArrowUpRight className="h-3.5 w-3.5" />
              </Link>
            </div>

            {/* Mock Screen Content Preview */}
            <div className="relative p-4 sm:p-6 bg-slate-100/50 dark:bg-[#090d16]/50 min-h-[380px] flex flex-col justify-center">
              {activeTab === 'overview' && (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {/* Preview Card 1 */}
                  <div className="rounded-xl border border-slate-200/80 bg-white p-4 shadow-2xs dark:border-white/10 dark:bg-[#101726]">
                    <div className="flex items-center justify-between text-xs text-slate-500">
                      <span className="font-bold text-slate-700 dark:text-slate-200">Customer Flow</span>
                      <span className="text-[10px] text-emerald-600 font-bold">Live Stream</span>
                    </div>
                    <div className="mt-2 text-2xl font-black text-slate-900 dark:text-white tabular-nums">
                      51 Entries
                    </div>
                    <div className="mt-3 h-2 w-full rounded-full bg-slate-100 dark:bg-slate-800 overflow-hidden">
                      <div className="h-full bg-blue-600 rounded-full w-[51%]" />
                    </div>
                    <div className="mt-2 flex justify-between text-[10px] text-slate-400">
                      <span>Daily Target</span>
                      <span>51% achieved</span>
                    </div>
                  </div>

                  {/* Preview Card 2 */}
                  <div className="rounded-xl border border-slate-200/80 bg-white p-4 shadow-2xs dark:border-white/10 dark:bg-[#101726]">
                    <div className="flex items-center justify-between text-xs text-slate-500">
                      <span className="font-bold text-slate-700 dark:text-slate-200">Stock Depletion</span>
                      <span className="text-[10px] text-blue-600 font-bold">YOLO Detection</span>
                    </div>
                    <div className="mt-2 text-2xl font-black text-slate-900 dark:text-white tabular-nums">
                      9 Configured SKUs
                    </div>
                    <div className="mt-3 flex h-2 w-full rounded-sm overflow-hidden bg-slate-100 dark:bg-slate-800">
                      <div className="bg-blue-600 w-1/3" />
                      <div className="bg-sky-400 w-1/4" />
                      <div className="bg-rose-500 w-1/6" />
                      <div className="bg-slate-300 dark:bg-slate-700 w-1/4" />
                    </div>
                    <div className="mt-2 flex justify-between text-[10px] text-slate-400">
                      <span>Optimal Stock Ratio</span>
                      <span className="font-semibold text-emerald-600">Healthy</span>
                    </div>
                  </div>

                  {/* Preview Card 3 */}
                  <div className="rounded-xl border border-slate-200/80 bg-white p-4 shadow-2xs dark:border-white/10 dark:bg-[#101726]">
                    <div className="flex items-center justify-between text-xs text-slate-500">
                      <span className="font-bold text-slate-700 dark:text-slate-200">Queue Clearance</span>
                      <span className="text-[10px] text-emerald-600 font-bold">M/M/c Model</span>
                    </div>
                    <div className="mt-2 text-2xl font-black text-slate-900 dark:text-white tabular-nums">
                      0.2 min avg
                    </div>
                    <div className="mt-3 h-2 w-full rounded-full bg-slate-100 dark:bg-slate-800 overflow-hidden">
                      <div className="h-full bg-sky-500 rounded-full w-[100%]" />
                    </div>
                    <div className="mt-2 flex justify-between text-[10px] text-slate-400">
                      <span>Clearance Efficiency</span>
                      <span className="font-semibold text-sky-600">100% Normal</span>
                    </div>
                  </div>
                </div>
              )}

              {activeTab === 'live' && (
                <div className="flex flex-col items-center justify-center p-8 text-center">
                  <div className="relative flex h-16 w-16 items-center justify-center rounded-2xl bg-blue-500/10 text-blue-600 dark:text-blue-400 mb-4">
                    <Radio className="h-8 w-8 animate-pulse" />
                  </div>
                  <h3 className="text-base font-bold text-slate-900 dark:text-white">
                    Multi-Camera Real-Time Streaming Available
                  </h3>
                  <p className="mt-1 text-xs text-slate-500 dark:text-slate-400 max-w-md">
                    Low-latency MJPEG feeds with bounding box detection overlays, person tracks, line-crossing
                    indicators, and shelf zone compliance meters.
                  </p>
                  <Link
                    to="/live"
                    className="mt-4 inline-flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-xs font-bold text-white shadow-sm hover:bg-blue-500"
                  >
                    <span>Launch Live Video Feeds</span>
                    <ArrowRight className="h-3.5 w-3.5" />
                  </Link>
                </div>
              )}

              {activeTab === 'shelves' && (
                <div className="flex flex-col items-center justify-center p-8 text-center">
                  <div className="relative flex h-16 w-16 items-center justify-center rounded-2xl bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 mb-4">
                    <Activity className="h-8 w-8" />
                  </div>
                  <h3 className="text-base font-bold text-slate-900 dark:text-white">
                    Architectural CAD Floorplan & Thermal Flow
                  </h3>
                  <p className="mt-1 text-xs text-slate-500 dark:text-slate-400 max-w-md">
                    1:100 scale blueprint mapping customer dwell time, entrance-to-checkout trajectories,
                    and interactive shelf bay inventory inspection.
                  </p>
                  <Link
                    to="/overview"
                    className="mt-4 inline-flex items-center gap-2 rounded-lg bg-emerald-600 px-4 py-2 text-xs font-bold text-white shadow-sm hover:bg-emerald-500"
                  >
                    <span>Explore CAD Floorplan</span>
                    <ArrowRight className="h-3.5 w-3.5" />
                  </Link>
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* ───────────────────────────────────────────────────────────── */}
      {/* 4. FOUR PILLARS OF IRIS SYSTEM                                 */}
      {/* ───────────────────────────────────────────────────────────── */}
      <section id="features" className="py-20 border-t border-slate-200/80 dark:border-white/5">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-2xl mx-auto">
            <span className="text-xs font-bold uppercase tracking-wider text-blue-600 dark:text-blue-400">
              Enterprise Vision Stack
            </span>
            <h2 className="mt-2 text-3xl font-extrabold tracking-tight sm:text-4xl text-slate-900 dark:text-white">
              Built for Real-World Retail Complexity
            </h2>
            <p className="mt-3 text-sm text-slate-600 dark:text-slate-400">
              Four unified modules engineered to run simultaneously on cost-effective edge workstations.
            </p>
          </div>

          <div className="mt-14 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {/* Feature 1: Customer Analytics */}
            <div className="rounded-xl border border-slate-200/80 bg-white p-5 shadow-2xs hover:border-blue-500/40 transition duration-200 dark:border-white/10 dark:bg-[#101726]">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-blue-50 text-blue-600 dark:bg-blue-950/50 dark:text-blue-400">
                <Users className="h-5 w-5" />
              </div>
              <h3 className="mt-4 text-base font-bold text-slate-900 dark:text-white">
                Customer Footfall & Dwell
              </h3>
              <p className="mt-2 text-xs leading-relaxed text-slate-600 dark:text-slate-400">
                Multi-camera line crossing counts entries and exits with ByteTrack person re-identification.
                Measures customer dwell time per promotional zone without face identification.
              </p>
              <Link
                to="/customers"
                className="mt-4 inline-flex items-center gap-1 text-xs font-bold text-blue-600 hover:text-blue-500 dark:text-blue-400"
              >
                <span>Customer Metrics</span>
                <ChevronRight className="h-3 w-3" />
              </Link>
            </div>

            {/* Feature 2: Shelf Vision */}
            <div className="rounded-xl border border-slate-200/80 bg-white p-5 shadow-2xs hover:border-blue-500/40 transition duration-200 dark:border-white/10 dark:bg-[#101726]">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-sky-50 text-sky-600 dark:bg-sky-950/50 dark:text-sky-400">
                <Layers className="h-5 w-5" />
              </div>
              <h3 className="mt-4 text-base font-bold text-slate-900 dark:text-white">
                Autonomous Shelf Vision
              </h3>
              <p className="mt-2 text-xs leading-relaxed text-slate-600 dark:text-slate-400">
                Optical item counting, void/gap detection, and planogram mismatch auditing.
                Automatically flags low-stock conditions before customers encounter empty shelves.
              </p>
              <Link
                to="/shelves"
                className="mt-4 inline-flex items-center gap-1 text-xs font-bold text-sky-600 hover:text-sky-500 dark:text-sky-400"
              >
                <span>Shelf Telemetry</span>
                <ChevronRight className="h-3 w-3" />
              </Link>
            </div>

            {/* Feature 3: Queue Operations */}
            <div className="rounded-xl border border-slate-200/80 bg-white p-5 shadow-2xs hover:border-blue-500/40 transition duration-200 dark:border-white/10 dark:bg-[#101726]">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-indigo-50 text-indigo-600 dark:bg-indigo-950/50 dark:text-indigo-400">
                <Clock className="h-5 w-5" />
              </div>
              <h3 className="mt-4 text-base font-bold text-slate-900 dark:text-white">
                Queue & Checkout Velocity
              </h3>
              <p className="mt-2 text-xs leading-relaxed text-slate-600 dark:text-slate-400">
                Live queue depth counting and M/M/c waiting time calculations.
                Triggers cashier redeployment alerts before checkout bottlenecks affect satisfaction.
              </p>
              <Link
                to="/queues"
                className="mt-4 inline-flex items-center gap-1 text-xs font-bold text-indigo-600 hover:text-indigo-500 dark:text-indigo-400"
              >
                <span>Queue Metrics</span>
                <ChevronRight className="h-3 w-3" />
              </Link>
            </div>

            {/* Feature 4: Loss Prevention & Anomaly */}
            <div className="rounded-xl border border-slate-200/80 bg-white p-5 shadow-2xs hover:border-blue-500/40 transition duration-200 dark:border-white/10 dark:bg-[#101726]">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-emerald-50 text-emerald-600 dark:bg-emerald-950/50 dark:text-emerald-400">
                <ShieldCheck className="h-5 w-5" />
              </div>
              <h3 className="mt-4 text-base font-bold text-slate-900 dark:text-white">
                Loss & Anomaly Detection
              </h3>
              <p className="mt-2 text-xs leading-relaxed text-slate-600 dark:text-slate-400">
                Audits rapid shelf clearance events and off-hours zone intrusions with evidence frame snapshots
                and automated severity escalation.
              </p>
              <Link
                to="/alerts"
                className="mt-4 inline-flex items-center gap-1 text-xs font-bold text-emerald-600 hover:text-emerald-500 dark:text-emerald-400"
              >
                <span>Alert Center</span>
                <ChevronRight className="h-3 w-3" />
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* ───────────────────────────────────────────────────────────── */}
      {/* 5. ARCHITECTURE & EDGE PRIVACY                                 */}
      {/* ───────────────────────────────────────────────────────────── */}
      <section id="architecture" className="py-20 bg-slate-100/50 dark:bg-[#0c1220]/50 border-t border-slate-200/80 dark:border-white/5">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-10 items-center">
            {/* Left Column: Privacy & Architecture Narrative */}
            <div className="lg:col-span-6 space-y-4">
              <span className="text-xs font-bold uppercase tracking-wider text-emerald-600 dark:text-emerald-400">
                Privacy-First Engineering
              </span>
              <h2 className="text-3xl font-extrabold tracking-tight sm:text-4xl text-slate-900 dark:text-white">
                Zero Cloud Egress. 100% On-Premise Privacy.
              </h2>
              <p className="text-sm leading-relaxed text-slate-600 dark:text-slate-300">
                Traditional retail camera solutions stream sensitive video over the public internet to third-party
                cloud vendors, racking up high bandwidth costs and regulatory liabilities.
              </p>
              <p className="text-sm leading-relaxed text-slate-600 dark:text-slate-300">
                IRIS executes all AI inference on the store&apos;s local hardware appliance. Video frames never leave
                the local network, no biometric profiles are created, and tracks expire automatically.
              </p>

              <div className="pt-2 space-y-2.5">
                <div className="flex items-center gap-2 text-xs font-semibold text-slate-700 dark:text-slate-200">
                  <CheckCircle2 className="h-4 w-4 text-emerald-500 shrink-0" />
                  <span>No facial recognition or customer biometric identity storage</span>
                </div>
                <div className="flex items-center gap-2 text-xs font-semibold text-slate-700 dark:text-slate-200">
                  <CheckCircle2 className="h-4 w-4 text-emerald-500 shrink-0" />
                  <span>Completely air-gapped deployment capability</span>
                </div>
                <div className="flex items-center gap-2 text-xs font-semibold text-slate-700 dark:text-slate-200">
                  <CheckCircle2 className="h-4 w-4 text-emerald-500 shrink-0" />
                  <span>Zero cloud API subscription fees or bandwidth egress charges</span>
                </div>
              </div>
            </div>

            {/* Right Column: Visual Pipeline Flow Diagram */}
            <div className="lg:col-span-6 rounded-2xl border border-slate-200/90 bg-white p-6 shadow-sm dark:border-white/10 dark:bg-[#101726]">
              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-4">
                Local Real-Time Pipeline Workflow
              </h4>

              <div className="space-y-3 font-mono text-xs">
                {/* Step 1 */}
                <div className="flex items-center gap-3 rounded-lg border border-slate-200 bg-slate-50 p-3 dark:border-white/5 dark:bg-slate-900/60">
                  <div className="flex h-7 w-7 items-center justify-center rounded bg-blue-600 text-white font-sans text-xs font-bold">
                    1
                  </div>
                  <div>
                    <div className="font-bold text-slate-900 dark:text-slate-100">
                      RTSP Stream Capture
                    </div>
                    <div className="text-[10px] text-slate-500 font-sans">
                      Non-blocking capture thread buffers newest frames
                    </div>
                  </div>
                </div>

                {/* Step 2 */}
                <div className="flex items-center gap-3 rounded-lg border border-slate-200 bg-slate-50 p-3 dark:border-white/5 dark:bg-slate-900/60">
                  <div className="flex h-7 w-7 items-center justify-center rounded bg-sky-600 text-white font-sans text-xs font-bold">
                    2
                  </div>
                  <div>
                    <div className="font-bold text-slate-900 dark:text-slate-100">
                      YOLO11 Object Inference
                    </div>
                    <div className="text-[10px] text-slate-500 font-sans">
                      Detects person bounding boxes and product bounding regions
                    </div>
                  </div>
                </div>

                {/* Step 3 */}
                <div className="flex items-center gap-3 rounded-lg border border-slate-200 bg-slate-50 p-3 dark:border-white/5 dark:bg-slate-900/60">
                  <div className="flex h-7 w-7 items-center justify-center rounded bg-indigo-600 text-white font-sans text-xs font-bold">
                    3
                  </div>
                  <div>
                    <div className="font-bold text-slate-900 dark:text-slate-100">
                      ByteTrack State Estimation
                    </div>
                    <div className="text-[10px] text-slate-500 font-sans">
                      Tracks trajectory across virtual tripwires and queue polygon zones
                    </div>
                  </div>
                </div>

                {/* Step 4 */}
                <div className="flex items-center gap-3 rounded-lg border border-slate-200 bg-slate-50 p-3 dark:border-white/5 dark:bg-slate-900/60">
                  <div className="flex h-7 w-7 items-center justify-center rounded bg-emerald-600 text-white font-sans text-xs font-bold">
                    4
                  </div>
                  <div>
                    <div className="font-bold text-slate-900 dark:text-slate-100">
                      WebSocket Event Broadcast
                    </div>
                    <div className="text-[10px] text-slate-500 font-sans">
                      Pushes live footfall metrics directly to executive dashboard
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ───────────────────────────────────────────────────────────── */}
      {/* 6. HARDWARE & ONNX AGNOSTIC COMPATIBILITY                     */}
      {/* ───────────────────────────────────────────────────────────── */}
      <section id="hardware" className="py-20 border-t border-slate-200/80 dark:border-white/5">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-2xl mx-auto">
            <span className="text-xs font-bold uppercase tracking-wider text-blue-600 dark:text-blue-400">
              Hardware Agnostic
            </span>
            <h2 className="mt-2 text-3xl font-extrabold tracking-tight sm:text-4xl text-slate-900 dark:text-white">
              Runs on Modern Edge Workstations
            </h2>
            <p className="mt-3 text-sm text-slate-600 dark:text-slate-400">
              Leverage ONNX Runtime, CUDA, or CPU fallback without proprietary hardware lock-in.
            </p>
          </div>

          <div className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="rounded-xl border border-slate-200/80 bg-white p-6 shadow-2xs dark:border-white/10 dark:bg-[#101726]">
              <Cpu className="h-6 w-6 text-blue-600 mb-3" />
              <h4 className="text-base font-bold text-slate-900 dark:text-white">CPU Execution</h4>
              <p className="mt-2 text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
                Optimized intra-op thread allocation ensures stable 30 FPS across multi-core Intel Core i7/i9 or
                AMD Ryzen processors without requiring expensive graphics cards.
              </p>
            </div>

            <div className="rounded-xl border border-slate-200/80 bg-white p-6 shadow-2xs dark:border-white/10 dark:bg-[#101726]">
              <Zap className="h-6 w-6 text-emerald-500 mb-3" />
              <h4 className="text-base font-bold text-slate-900 dark:text-white">NVIDIA CUDA Acceleration</h4>
              <p className="mt-2 text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
                Hardware acceleration via TensorRT and CUDA Execution Providers reduces per-frame inference latency
                below 10ms for multi-camera retail enterprise scaling.
              </p>
            </div>

            <div className="rounded-xl border border-slate-200/80 bg-white p-6 shadow-2xs dark:border-white/10 dark:bg-[#101726]">
              <Server className="h-6 w-6 text-sky-500 mb-3" />
              <h4 className="text-base font-bold text-slate-900 dark:text-white">Flexible Video Inputs</h4>
              <p className="mt-2 text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
                Supports IP cameras via RTSP/RTSPS, USB webcams, pre-recorded video files, and built-in synthetic test
                generators for seamless validation.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ───────────────────────────────────────────────────────────── */}
      {/* 7. BOTTOM CALL TO ACTION                                      */}
      {/* ───────────────────────────────────────────────────────────── */}
      <section className="py-20 border-t border-slate-200/80 bg-gradient-to-b from-transparent to-blue-50/50 dark:to-blue-950/20 dark:border-white/5">
        <div className="mx-auto max-w-4xl px-4 sm:px-6 lg:px-8 text-center">
          <div className="flex justify-center mb-4">
            <img src="/iris-eye.png" alt="IRIS" className="h-10 w-auto object-contain" />
          </div>
          <h2 className="text-3xl font-black tracking-tight sm:text-4xl text-slate-900 dark:text-white">
            Ready to Experience Retail Intelligence?
          </h2>
          <p className="mt-3 text-sm text-slate-600 dark:text-slate-300 max-w-xl mx-auto">
            Launch the live executive dashboard or watch real-time vision analytics directly in your browser.
          </p>
          <div className="mt-8 flex flex-col sm:flex-row items-center justify-center gap-3">
            <Link
              to="/overview"
              className="flex w-full sm:w-auto items-center justify-center gap-2 rounded-xl bg-blue-600 px-6 py-3.5 text-sm font-bold text-white shadow-md shadow-blue-500/30 hover:bg-blue-500 active:scale-98 transition"
            >
              <span>Launch Dashboard Console</span>
              <ArrowRight className="h-4 w-4" />
            </Link>
            <Link
              to="/live"
              className="flex w-full sm:w-auto items-center justify-center gap-2 rounded-xl border border-slate-200 bg-white px-6 py-3.5 text-sm font-bold text-slate-700 hover:bg-slate-50 dark:border-white/10 dark:bg-[#101726] dark:text-slate-200 dark:hover:bg-slate-800 transition"
            >
              <span>Watch Live Streams</span>
            </Link>
          </div>
        </div>
      </section>

      {/* ───────────────────────────────────────────────────────────── */}
      {/* 8. FOOTER                                                      */}
      {/* ───────────────────────────────────────────────────────────── */}
      <footer className="border-t border-slate-200/80 bg-white py-12 dark:border-white/5 dark:bg-[#080d16]">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <img
              src={dark ? '/iris-horizontal-dark.png' : '/iris-horizontal.png'}
              alt="IRIS"
              className="h-6 w-auto object-contain"
            />
          </div>
          <div className="flex items-center gap-6 text-xs font-semibold text-slate-500 dark:text-slate-400">
            <Link to="/overview" className="hover:text-blue-600 dark:hover:text-blue-400">Dashboard</Link>
            <Link to="/live" className="hover:text-blue-600 dark:hover:text-blue-400">Live Vision</Link>
            <Link to="/inventory" className="hover:text-blue-600 dark:hover:text-blue-400">Inventory</Link>
            <Link to="/cameras" className="hover:text-blue-600 dark:hover:text-blue-400">Cameras</Link>
            <Link to="/settings" className="hover:text-blue-600 dark:hover:text-blue-400">Settings</Link>
          </div>
          <div className="text-xs text-slate-400">
            IRIS · Intelligent Retail Insight System
          </div>
        </div>
      </footer>
    </div>
  )
}
