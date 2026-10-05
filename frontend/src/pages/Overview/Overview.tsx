import { useState, useMemo } from 'react'
import { Link } from 'react-router-dom'
import {
  Info,
  MoreHorizontal,
  ChevronRight,
  ChevronDown,
  ChevronUp,
  Radio,
  Video,
  Package,
} from 'lucide-react'
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  Tooltip,
} from 'recharts'
import { api } from '../../services/api'
import { useApi, useLiveEvents, usePolling } from '../../hooks/useAsync'
import { RadialGauge, fmtNumber } from '../../components/ui'
import type { DashboardData, FootfallResponse, QueueMeasurement, Shelf } from '../../types/api'

// Custom High-Craft HUD Tooltip for charts
interface TooltipPayloadItem {
  color?: string
  name?: string
  value?: number | string
}

function HudChartTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean
  payload?: TooltipPayloadItem[]
  label?: string
}) {
  if (!active || !payload || !payload.length) return null
  return (
    <div className="rounded-lg border border-slate-200/90 bg-white/95 px-3 py-2 shadow-lg backdrop-blur-md dark:border-white/10 dark:bg-slate-900/95 dark:text-white font-sans text-xs">
      <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400 mb-1">
        {label}
      </div>
      <div className="space-y-1">
        {payload.map((entry, idx) => (
          <div key={idx} className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-1.5 text-slate-600 dark:text-slate-300">
              <span
                className="h-2 w-2 rounded-xs shrink-0"
                style={{ backgroundColor: entry.color || '#2563eb' }}
              />
              <span className="capitalize">{entry.name || 'Value'}:</span>
            </div>
            <span className="font-extrabold text-slate-900 dark:text-white tabular-nums">
              {entry.value !== undefined ? fmtNumber(Number(entry.value), 0) : '0'}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function Overview() {
  const dashboard = useApi<DashboardData>(() => api.dashboard(), [])
  const footfall7d = useApi<FootfallResponse>(() => api.footfall({ range: '7d' }), [])
  const queues = useApi<QueueMeasurement[]>(() => api.queues(), [])
  const cameras = useApi(() => api.cameras({ limit: 100 }), [])
  const shelves = useApi<{ items: Shelf[] }>(() => api.shelves({ limit: 10 }), [])

  // Dynamic live event reload
  useLiveEvents(() => {
    dashboard.reload()
    footfall7d.reload()
    queues.reload()
    cameras.reload()
    shelves.reload()
  })

  // Dynamic polling every 10 seconds
  usePolling(() => {
    dashboard.reload()
    queues.reload()
  }, 10000)

  const [occupancyExpanded, setOccupancyExpanded] = useState(true)
  const [selectedShelfId, setSelectedShelfId] = useState<string | null>(null)

  // Camera telemetry
  const cameraItems = cameras.data?.items ?? []
  const onlineCount = cameraItems.filter((c) => c.status === 'online').length

  // Customer Footfall & Occupancy metrics
  const customerData = dashboard.data?.customers
  const totalFootfall = customerData?.total_entries ?? 0
  const currentOccupancy = customerData?.current_occupancy ?? customerData?.active_tracks ?? 0
  const peakOccupancy = customerData?.peak_occupancy ?? 0
  const maxCapacity = Math.max(100, Math.ceil((peakOccupancy > 0 ? peakOccupancy : 60) * 1.5))
  const occupancyDensityPct = Math.min(100, Math.round((currentOccupancy / maxCapacity) * 100))
  const footfallTarget = Math.max(100, Math.ceil(totalFootfall * 1.25))
  const footfallProgress = Math.min(100, Math.max(4, Math.round((totalFootfall / footfallTarget) * 100)))
  const hourlyRate = (totalFootfall / Math.max(1, Math.min(24, (customerData?.points?.length ?? 1)))).toFixed(1)

  // 1. Store Occupancy Mini Trend Data (Left Panel)
  const occupancyMiniData = useMemo(() => {
    const pts = dashboard.data?.customers?.points ?? []
    if (pts.length === 0) {
      return [
        { label: '08:00', value: 0 },
        { label: '12:00', value: 0 },
        { label: '16:00', value: 0 },
        { label: '20:00', value: 0 },
      ]
    }
    return pts.map((p) => ({
      label: new Date(p.bucket).toLocaleTimeString([], { hour: 'numeric', hour12: false }),
      value: p.entries,
    }))
  }, [dashboard.data])

  // 2. Weekly Traffic Double Spline Area Chart
  const weeklyTrafficData = useMemo(() => {
    const pts7d = footfall7d.data?.points ?? []
    if (pts7d.length > 0) {
      const dayMap = new Map<string, { upper: number; lower: number }>()
      for (const p of pts7d) {
        const d = new Date(p.bucket).toLocaleDateString([], { weekday: 'short' })
        const cur = dayMap.get(d) ?? { upper: 0, lower: 0 }
        dayMap.set(d, {
          upper: cur.upper + (p.entries || 0),
          lower: cur.lower + (p.exits || 0),
        })
      }
      return Array.from(dayMap.entries()).map(([day, val]) => ({
        day,
        upper: val.upper,
        lower: val.lower,
      }))
    }

    const dashPts = dashboard.data?.customers?.points ?? []
    if (dashPts.length > 0) {
      return dashPts.map((p) => ({
        day: new Date(p.bucket).toLocaleTimeString([], { hour: 'numeric' }),
        upper: p.entries,
        lower: p.exits,
      }))
    }

    return [
      { day: 'Mon', upper: 0, lower: 0 },
      { day: 'Tue', upper: 0, lower: 0 },
      { day: 'Wed', upper: 0, lower: 0 },
      { day: 'Thu', upper: 0, lower: 0 },
      { day: 'Fri', upper: 0, lower: 0 },
      { day: 'Sat', upper: 0, lower: 0 },
      { day: 'Sun', upper: 0, lower: 0 },
    ]
  }, [footfall7d.data, dashboard.data])

  // 3. Weekly Traffic & Peak Hours Bar/Line Chart
  const trafficPeakData = useMemo(() => {
    const pts = dashboard.data?.customers?.points ?? []
    if (pts.length === 0) {
      return [
        { label: '09:00', bar1: 0, bar2: 0, line: 0 },
        { label: '11:00', bar1: 0, bar2: 0, line: 0 },
        { label: '13:00', bar1: 0, bar2: 0, line: 0 },
        { label: '15:00', bar1: 0, bar2: 0, line: 0 },
        { label: '17:00', bar1: 0, bar2: 0, line: 0 },
      ]
    }
    return pts.map((p) => ({
      label: new Date(p.bucket).toLocaleTimeString([], { hour: 'numeric' }),
      bar1: p.entries,
      bar2: p.exits,
      line: Math.max(0, p.net > 0 ? p.net : Math.round((p.entries + p.exits) / 2)),
    }))
  }, [dashboard.data])

  // 4. Peak Hours Distribution Chart
  const peakHoursData = useMemo(() => {
    const pts = dashboard.data?.customers?.points ?? []
    if (pts.length === 0) {
      return [
        { hour: '8am', visitors: 0, trend: 0 },
        { hour: '12pm', visitors: 0, trend: 0 },
        { hour: '4pm', visitors: 0, trend: 0 },
        { hour: '8pm', visitors: 0, trend: 0 },
      ]
    }
    return pts.map((p) => ({
      hour: new Date(p.bucket).toLocaleTimeString([], { hour: 'numeric', hour12: true }),
      visitors: p.entries,
      trend: Math.max(0, p.entries - 1),
    }))
  }, [dashboard.data])

  const maxPeakVisitors = Math.max(...peakHoursData.map((d) => d.visitors), 0)
  const peakHourItem = peakHoursData.find((d) => d.visitors === maxPeakVisitors)
  const peakSharePct = totalFootfall > 0 ? Math.round((maxPeakVisitors / totalFootfall) * 100) : 0

  // 5. Inventory Status Percentages
  const inv = dashboard.data?.inventory
  const totalSkus = inv?.total_skus ?? 0
  const healthyCount = inv?.healthy_count ?? 0
  const lowStockCount = inv?.low_stock_count ?? 0
  const outOfStockCount = inv?.out_of_stock_count ?? 0
  const unknownCount = inv?.unknown_count ?? 0
  const totalCount =
    totalSkus > 0 ? totalSkus : Math.max(1, healthyCount + lowStockCount + outOfStockCount + unknownCount)

  const normalPct = Math.round((healthyCount / totalCount) * 100)
  const lowPct = Math.round((lowStockCount / totalCount) * 100)
  const oosPct = Math.round((outOfStockCount / totalCount) * 100)
  const unobservedPct = Math.round((unknownCount / totalCount) * 100)

  // 6. Queue Telemetry
  const queueItems = queues.data ?? dashboard.data?.queues ?? []
  const waitSecondsList = queueItems.map((q) => q.estimated_wait_seconds || 0)
  const avgWaitSec =
    waitSecondsList.length > 0
      ? waitSecondsList.reduce((acc, v) => acc + v, 0) / waitSecondsList.length
      : 0
  const avgWaitMins = (avgWaitSec / 60).toFixed(1)
  const waitGaugeVal = Math.min(100, Math.round((avgWaitSec / 300) * 100))

  const totalQueueZones = queueItems.length
  const normalQueueZones = queueItems.filter((q) => q.status === 'normal').length
  const clearancePct =
    totalQueueZones > 0 ? Math.round((normalQueueZones / totalQueueZones) * 100) : 100

  // Shelf data
  const shelfList = shelves.data?.items ?? []
  const selectedShelf = shelfList.find((s) => s.id === selectedShelfId) ?? shelfList[0]

  return (
    <div className="flex h-full w-full gap-3 overflow-hidden p-1 font-sans">
      {/* ───────────────────────────────────────────────────────────── */}
      {/* LEFT COLUMN: Industrial Hardware Console                       */}
      {/* ───────────────────────────────────────────────────────────── */}
      <div className="retail-card flex w-68 shrink-0 flex-col justify-between overflow-hidden bg-white p-3.5 shadow-xs border border-slate-200/80 dark:bg-[#0f172a] dark:border-white/10">
        <div className="space-y-4">
          {/* Hardware Header with Serial Badge & Logo */}
          <div className="flex items-center justify-between border-b border-slate-100 pb-2.5 dark:border-white/5">
            <div className="flex items-center gap-2">
              <img src="/iris-eye.png" alt="IRIS" className="h-5 w-auto object-contain" />
              <span className="text-xs font-black tracking-tight text-slate-800 dark:text-slate-100 font-sans uppercase">
                Console #01
              </span>
            </div>
            <span className="rounded bg-slate-100 px-1.5 py-0.5 text-[9px] font-bold text-slate-500 uppercase tracking-wider dark:bg-slate-800 dark:text-slate-400">
              Live Edge
            </span>
          </div>

          {/* Metric 1: Customer Footfall (Industrial Pacer) */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400">
              <span className="font-semibold text-slate-700 dark:text-slate-200">
                Customer Footfall
              </span>
              <span className="text-[10px] font-semibold text-blue-600 dark:text-blue-400 tabular-nums">
                ~{hourlyRate}/hr
              </span>
            </div>
            <div className="text-3xl font-black tracking-tight text-slate-900 dark:text-white tabular-nums">
              {fmtNumber(totalFootfall, 0)}
            </div>

            {/* Segmented Milestone Pacer */}
            <div className="pt-1 space-y-1">
              <div className="flex justify-between text-[10px] text-slate-400 font-medium">
                <span>Daily Pacing</span>
                <span className="font-semibold text-slate-600 dark:text-slate-300 tabular-nums">
                  {footfallProgress}% of {fmtNumber(footfallTarget, 0)}
                </span>
              </div>
              {/* High-craft track with tick notches */}
              <div className="relative h-2 w-full rounded-sm bg-slate-100 dark:bg-slate-800 overflow-hidden">
                <div
                  className="h-full bg-blue-600 transition-all duration-500 rounded-sm"
                  style={{ width: `${footfallProgress}%` }}
                />
                {/* 25%, 50%, 75% tick marks */}
                <div className="absolute top-0 bottom-0 left-1/4 w-px bg-white/60 dark:bg-black/40 pointer-events-none" />
                <div className="absolute top-0 bottom-0 left-2/4 w-px bg-white/60 dark:bg-black/40 pointer-events-none" />
                <div className="absolute top-0 bottom-0 left-3/4 w-px bg-white/60 dark:bg-black/40 pointer-events-none" />
              </div>
            </div>
          </div>

          {/* Metric 2: Store Occupancy (Live Density Status) */}
          <div className="border-t border-slate-100 pt-3 dark:border-white/5 space-y-1.5">
            <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400">
              <span className="font-semibold text-slate-700 dark:text-slate-200">
                Store Occupancy
              </span>
              <span className={`inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wider ${
                occupancyDensityPct > 80
                  ? 'bg-rose-50 text-rose-600 dark:bg-rose-950/40 dark:text-rose-400'
                  : occupancyDensityPct > 50
                  ? 'bg-amber-50 text-amber-600 dark:bg-amber-950/40 dark:text-amber-400'
                  : 'bg-emerald-50 text-emerald-600 dark:bg-emerald-950/40 dark:text-emerald-400'
              }`}>
                <span className={`h-1 w-1 rounded-full ${
                  occupancyDensityPct > 80 ? 'bg-rose-500' : occupancyDensityPct > 50 ? 'bg-amber-500' : 'bg-emerald-500'
                }`} />
                {occupancyDensityPct > 80 ? 'Surge' : occupancyDensityPct > 50 ? 'Moderate' : 'Optimal'}
              </span>
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-3xl font-black tracking-tight text-slate-900 dark:text-white tabular-nums">
                {fmtNumber(currentOccupancy, 0)}
              </span>
              <span className="text-xs text-slate-400 font-medium tabular-nums">
                / {maxCapacity} cap
              </span>
            </div>
          </div>

          {/* Live Feeds Summary */}
          <div className="border-t border-slate-100 pt-3 dark:border-white/5">
            <Link
              to="/cameras"
              className="flex w-full items-center justify-between text-xs font-semibold text-slate-700 hover:text-blue-600 dark:text-slate-300 transition"
            >
              <span className="flex items-center gap-1.5">
                <Video className="h-3.5 w-3.5 text-slate-400" />
                <span>Camera Feeds</span>
              </span>
              <div className="flex items-center gap-1">
                <span className="text-[10px] text-emerald-600 dark:text-emerald-400 font-bold tabular-nums">
                  {cameraItems.length > 0 ? `${onlineCount}/${cameraItems.length} Active` : '0 Feeds'}
                </span>
                <ChevronRight className="h-3 w-3 text-slate-400" />
              </div>
            </Link>
          </div>

          {/* Collapsible Occupancy Trend Wave */}
          <div className="border-t border-slate-100 pt-2.5 dark:border-white/5">
            <button
              type="button"
              onClick={() => setOccupancyExpanded((v) => !v)}
              className="flex w-full items-center justify-between text-xs font-semibold text-slate-700 dark:text-slate-300 cursor-pointer"
            >
              <span>Occupancy Curve</span>
              {occupancyExpanded ? (
                <ChevronUp className="h-3.5 w-3.5 text-slate-400" />
              ) : (
                <ChevronDown className="h-3.5 w-3.5 text-slate-400" />
              )}
            </button>

            {occupancyExpanded && (
              <div className="mt-2 h-24 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart
                    data={occupancyMiniData}
                    margin={{ top: 5, right: 5, left: -25, bottom: 0 }}
                  >
                    <defs>
                      <linearGradient id="occupancyGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#2563eb" stopOpacity={0.4} />
                        <stop offset="95%" stopColor="#2563eb" stopOpacity={0.0} />
                      </linearGradient>
                    </defs>
                    <XAxis
                      dataKey="label"
                      tickLine={false}
                      axisLine={false}
                      tick={{ fontSize: 9, fill: '#94a3b8' }}
                    />
                    <YAxis
                      domain={[0, 'auto']}
                      tickLine={false}
                      axisLine={false}
                      tick={{ fontSize: 8, fill: '#94a3b8' }}
                    />
                    <Tooltip content={<HudChartTooltip />} />
                    <Area
                      type="natural"
                      dataKey="value"
                      name="Occupants"
                      stroke="#2563eb"
                      strokeWidth={2}
                      fillOpacity={1}
                      fill="url(#occupancyGrad)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>
        </div>

        {/* Primary Action Button (Tactile Hardware Style) */}
        <div className="pt-2">
          <Link
            to="/live"
            className="flex w-full items-center justify-center gap-2 rounded-lg bg-blue-600 px-3.5 py-2.5 text-xs font-bold text-white shadow-xs transition hover:bg-blue-500 active:scale-98"
          >
            <Radio className="h-3.5 w-3.5 animate-pulse" />
            <span>Live Stream Console</span>
          </Link>
        </div>
      </div>

      {/* ───────────────────────────────────────────────────────────── */}
      {/* MAIN RIGHT MATRIX: Top 3 Charts + Bottom 3 Analytics Sections  */}
      {/* ───────────────────────────────────────────────────────────── */}
      <div className="flex flex-1 flex-col gap-3 overflow-hidden">
        {/* TOP ROW: 3 Visual Analysis Charts */}
        <div className="grid h-[42%] grid-cols-12 gap-3 min-h-0">
          {/* Card 1: Weekly Traffic (Double Spline Area) */}
          <div className="retail-card retail-card-bracket col-span-4 flex flex-col p-3.5 bg-white border border-slate-200/80 dark:bg-[#0f172a] dark:border-white/10">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-1.5">
                <h3 className="text-xs font-bold text-slate-800 dark:text-slate-100">
                  Weekly Footfall
                </h3>
                <span title="Aggregate customer entries and exits over 7 days">
                  <Info className="h-3 w-3 text-slate-400 cursor-pointer" />
                </span>
              </div>
              <div className="flex items-center gap-1">
                <span className="rounded bg-slate-100 px-1.5 py-0.5 text-[9px] font-semibold text-slate-600 dark:bg-slate-800 dark:text-slate-300">
                  7D Aggregate
                </span>
              </div>
            </div>

            <div className="flex-1 min-h-0 mt-1">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart
                  data={weeklyTrafficData}
                  margin={{ top: 10, right: 10, left: -22, bottom: 0 }}
                >
                  <defs>
                    <linearGradient id="trafficGradUpper" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#2563eb" stopOpacity={0.35} />
                      <stop offset="95%" stopColor="#2563eb" stopOpacity={0.0} />
                    </linearGradient>
                    <linearGradient id="trafficGradLower" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#38bdf8" stopOpacity={0.25} />
                      <stop offset="95%" stopColor="#38bdf8" stopOpacity={0.0} />
                    </linearGradient>
                  </defs>
                  <XAxis
                    dataKey="day"
                    tickLine={false}
                    axisLine={false}
                    tick={{ fontSize: 10, fill: '#94a3b8' }}
                  />
                  <YAxis
                    domain={[0, 'auto']}
                    tickLine={false}
                    axisLine={false}
                    tick={{ fontSize: 9, fill: '#94a3b8' }}
                  />
                  <Tooltip content={<HudChartTooltip />} />
                  <Area
                    type="monotone"
                    dataKey="upper"
                    name="Entries"
                    stroke="#2563eb"
                    strokeWidth={2}
                    fillOpacity={1}
                    fill="url(#trafficGradUpper)"
                  />
                  <Area
                    type="monotone"
                    dataKey="lower"
                    name="Exits"
                    stroke="#38bdf8"
                    strokeWidth={1.5}
                    fillOpacity={1}
                    fill="url(#trafficGradLower)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Card 2: Traffic & Flow (Grouped Bars + Net Flow Spline) */}
          <div className="retail-card retail-card-bracket col-span-5 flex flex-col p-3.5 bg-white border border-slate-200/80 dark:bg-[#0f172a] dark:border-white/10">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3 text-[10px] font-semibold text-slate-500">
                <span className="flex items-center gap-1">
                  <span className="h-2 w-2 rounded-xs bg-blue-600" /> Entries
                </span>
                <span className="flex items-center gap-1">
                  <span className="h-2 w-2 rounded-xs bg-sky-200 dark:bg-sky-800" /> Exits
                </span>
                <span className="flex items-center gap-1">
                  <span className="h-0.5 w-3 bg-blue-500" /> Net Flow
                </span>
              </div>
              <span className="text-[10px] font-medium text-slate-400 tabular-nums">
                Hourly Buckets
              </span>
            </div>

            <div className="flex-1 min-h-0 mt-1">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={trafficPeakData}
                  margin={{ top: 10, right: 10, left: -22, bottom: 0 }}
                  barGap={3}
                >
                  <XAxis
                    dataKey="label"
                    tickLine={false}
                    axisLine={false}
                    tick={{ fontSize: 10, fill: '#94a3b8' }}
                  />
                  <YAxis
                    domain={[0, 'auto']}
                    tickLine={false}
                    axisLine={false}
                    tick={{ fontSize: 9, fill: '#94a3b8' }}
                  />
                  <Tooltip content={<HudChartTooltip />} />
                  <Bar
                    dataKey="bar1"
                    name="Entries"
                    fill="#2563eb"
                    radius={[2, 2, 0, 0]}
                    maxBarSize={14}
                  />
                  <Bar
                    dataKey="bar2"
                    name="Exits"
                    fill="#bae6fd"
                    radius={[2, 2, 0, 0]}
                    maxBarSize={14}
                    className="dark:fill-sky-900/60"
                  />
                  <Line
                    type="monotone"
                    dataKey="line"
                    name="Net Flow"
                    stroke="#0284c7"
                    strokeWidth={2}
                    dot={{ r: 2.5, fill: '#0284c7' }}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Card 3: Peak Hours (Load Distribution) */}
          <div className="retail-card retail-card-bracket col-span-3 flex flex-col p-3.5 bg-white border border-slate-200/80 dark:bg-[#0f172a] dark:border-white/10">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <h3 className="text-xs font-bold text-slate-800 dark:text-slate-100">Peak Load</h3>
                {peakHourItem && maxPeakVisitors > 0 && (
                  <span className="rounded bg-blue-50 px-1.5 py-0.5 text-[9px] font-bold text-blue-600 dark:bg-blue-900/30 dark:text-blue-300">
                    {peakHourItem.hour} ({peakSharePct}%)
                  </span>
                )}
              </div>
              <button
                type="button"
                className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-200"
              >
                <MoreHorizontal className="h-4 w-4" />
              </button>
            </div>

            <div className="flex-1 min-h-0 mt-1">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={peakHoursData}
                  margin={{ top: 10, right: 5, left: -28, bottom: 0 }}
                >
                  <XAxis
                    dataKey="hour"
                    tickLine={false}
                    axisLine={false}
                    tick={{ fontSize: 9, fill: '#94a3b8' }}
                  />
                  <YAxis
                    domain={[0, 'auto']}
                    tickLine={false}
                    axisLine={false}
                    tick={{ fontSize: 9, fill: '#94a3b8' }}
                  />
                  <Tooltip content={<HudChartTooltip />} />
                  <Bar
                    dataKey="visitors"
                    name="Footfall"
                    fill="#38bdf8"
                    radius={[2, 2, 0, 0]}
                    maxBarSize={10}
                  />
                  <Line
                    type="natural"
                    dataKey="trend"
                    name="Moving Avg"
                    stroke="#2563eb"
                    strokeWidth={1.5}
                    dot={{ r: 2, fill: '#2563eb' }}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* BOTTOM ROW: Inventory Status, Product Navigation Store Heatmap, Queue Gauges */}
        <div className="grid h-[58%] grid-cols-12 gap-3 min-h-0">
          {/* Card 4: Inventory Health Matrix (Dynamic Stacked & Queue Regions) */}
          <div className="retail-card retail-card-bracket col-span-3 flex flex-col justify-between p-3.5 bg-white border border-slate-200/80 dark:bg-[#0f172a] dark:border-white/10 overflow-hidden">
            <div>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-1.5">
                  <Package className="h-3.5 w-3.5 text-blue-600" />
                  <h3 className="text-xs font-bold text-slate-800 dark:text-slate-100">
                    Inventory Health
                  </h3>
                </div>
                <span className="text-[10px] font-semibold text-slate-400 tabular-nums">
                  {totalSkus} SKUs
                </span>
              </div>

              {/* Segmented Stacked Stock Bar */}
              <div className="mt-2.5 space-y-1">
                <div className="flex h-2 w-full rounded-sm overflow-hidden bg-slate-100 dark:bg-slate-800">
                  <div
                    className="bg-blue-600 transition-all duration-500"
                    style={{ width: `${normalPct}%` }}
                    title={`Healthy: ${normalPct}%`}
                  />
                  <div
                    className="bg-sky-400 transition-all duration-500"
                    style={{ width: `${lowPct}%` }}
                    title={`Low Stock: ${lowPct}%`}
                  />
                  <div
                    className="bg-rose-500 transition-all duration-500"
                    style={{ width: `${oosPct}%` }}
                    title={`Out of Stock: ${oosPct}%`}
                  />
                  <div
                    className="bg-slate-300 dark:bg-slate-700 transition-all duration-500"
                    style={{ width: `${unobservedPct}%` }}
                    title={`Pending Scan: ${unobservedPct}%`}
                  />
                </div>
              </div>

              {/* Detailed Breakdown List */}
              <div className="mt-2.5 space-y-2">
                <div className="flex items-center justify-between text-[11px]">
                  <div className="flex items-center gap-1.5">
                    <span className="h-2 w-2 rounded-xs bg-blue-600" />
                    <span className="text-slate-600 dark:text-slate-300">Healthy Stock</span>
                  </div>
                  <span className="font-bold text-slate-900 dark:text-white tabular-nums">
                    {healthyCount} <span className="text-[10px] text-slate-400 font-normal">({normalPct}%)</span>
                  </span>
                </div>

                <div className="flex items-center justify-between text-[11px]">
                  <div className="flex items-center gap-1.5">
                    <span className="h-2 w-2 rounded-xs bg-sky-400" />
                    <span className="text-slate-600 dark:text-slate-300">Low Stock Warning</span>
                  </div>
                  <span className="font-bold text-slate-900 dark:text-white tabular-nums">
                    {lowStockCount} <span className="text-[10px] text-slate-400 font-normal">({lowPct}%)</span>
                  </span>
                </div>

                <div className="flex items-center justify-between text-[11px]">
                  <div className="flex items-center gap-1.5">
                    <span className="h-2 w-2 rounded-xs bg-rose-500" />
                    <span className="text-slate-600 dark:text-slate-300">Out of Stock</span>
                  </div>
                  <span className="font-bold text-slate-900 dark:text-white tabular-nums">
                    {outOfStockCount} <span className="text-[10px] text-slate-400 font-normal">({oosPct}%)</span>
                  </span>
                </div>

                <div className="flex items-center justify-between text-[11px]">
                  <div className="flex items-center gap-1.5">
                    <span className="h-2 w-2 rounded-xs bg-slate-300 dark:bg-slate-700" />
                    <span className="text-slate-600 dark:text-slate-300">Pending Vision Scan</span>
                  </div>
                  <span className="font-bold text-slate-900 dark:text-white tabular-nums">
                    {unknownCount} <span className="text-[10px] text-slate-400 font-normal">({unobservedPct}%)</span>
                  </span>
                </div>
              </div>
            </div>

            {/* Sub-section: Live Queue Telemetry */}
            <div className="border-t border-slate-100 pt-2.5 mt-2 dark:border-white/5 space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-slate-800 dark:text-slate-100">
                  Active Queues
                </span>
                <span className="text-[9px] font-bold text-emerald-600 uppercase tracking-wider">
                  M/M/c Live
                </span>
              </div>
              {queueItems.length > 0 ? (
                queueItems.slice(0, 3).map((q) => {
                  return (
                    <div key={q.zone_id} className="flex justify-between items-center text-[10px] text-slate-500">
                      <span className="truncate max-w-[130px] font-medium" title={q.zone_name}>
                        {q.zone_name}
                      </span>
                      <span className="font-semibold text-slate-800 dark:text-slate-200 tabular-nums">
                        {q.current_queue_length > 0 ? `${q.current_queue_length} waiting` : 'Clear'}
                        <span className="ml-1 text-[9px] text-slate-400 font-normal">
                          ({q.status})
                        </span>
                      </span>
                    </div>
                  )
                })
              ) : (
                <div className="text-[10px] text-slate-400 italic">No queue regions active</div>
              )}
            </div>
          </div>

          {/* Card 5: Architectural CAD Blueprint & Surveillance Heatmap (THE SHOWCASE) */}
          <div className="retail-card retail-card-bracket col-span-6 flex flex-col p-3.5 bg-white border border-slate-200/80 dark:bg-[#0f172a] dark:border-white/10 relative overflow-hidden">
            <div className="flex items-center justify-between z-10">
              <div className="flex items-center gap-2">
                <div className="flex items-center gap-1.5">
                  <span className="h-2 w-2 rounded-full bg-emerald-500 shadow-[0_0_6px_rgba(16,185,129,0.5)]" />
                  <h3 className="text-xs font-bold text-slate-800 dark:text-slate-100">
                    CAD Architectural Floorplan
                  </h3>
                </div>
                <span className="text-[10px] font-mono font-medium text-slate-400 border border-slate-200 px-1.5 py-0.2 rounded dark:border-white/10">
                  GRID 1:100
                </span>
                {selectedShelf && (
                  <span className="text-[10px] font-bold text-blue-600 bg-blue-50 px-2 py-0.5 rounded dark:bg-blue-900/30 dark:text-blue-300 truncate max-w-[140px]">
                    {selectedShelf.name}
                  </span>
                )}
              </div>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => setSelectedShelfId((v) => (v ? null : shelfList[0]?.id ?? null))}
                  className="rounded border border-slate-200 px-2 py-0.5 text-[11px] font-semibold text-slate-600 hover:bg-slate-100 transition dark:border-white/10 dark:text-slate-300 dark:hover:bg-slate-800 cursor-pointer"
                >
                  {selectedShelfId ? 'Deselect' : 'Inspect Bays'}
                </button>
                <MoreHorizontal className="h-4 w-4 text-slate-400" />
              </div>
            </div>

            {/* Architectural CAD Blueprint & Surveillance SVG Canvas */}
            <div className="relative flex-1 min-h-0 w-full mt-2 rounded-lg bg-slate-50 border border-slate-200/70 p-2 overflow-hidden dark:bg-[#070b14] dark:border-slate-800 cad-grid">
              <svg className="w-full h-full" viewBox="0 0 460 260">
                <defs>
                  {/* Heatmap Radial Gradients */}
                  <radialGradient id="heatHot" cx="50%" cy="50%" r="50%">
                    <stop offset="0%" stopColor="#ef4444" stopOpacity="0.8" />
                    <stop offset="35%" stopColor="#f97316" stopOpacity="0.55" />
                    <stop offset="70%" stopColor="#eab308" stopOpacity="0.25" />
                    <stop offset="100%" stopColor="#3b82f6" stopOpacity="0" />
                  </radialGradient>
                  <radialGradient id="heatWarm" cx="50%" cy="50%" r="50%">
                    <stop offset="0%" stopColor="#f59e0b" stopOpacity="0.65" />
                    <stop offset="45%" stopColor="#10b981" stopOpacity="0.35" />
                    <stop offset="85%" stopColor="#06b6d4" stopOpacity="0.15" />
                    <stop offset="100%" stopColor="#3b82f6" stopOpacity="0" />
                  </radialGradient>
                  <radialGradient id="heatCool" cx="50%" cy="50%" r="50%">
                    <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.5" />
                    <stop offset="60%" stopColor="#3b82f6" stopOpacity="0.2" />
                    <stop offset="100%" stopColor="#3b82f6" stopOpacity="0" />
                  </radialGradient>

                  {/* Camera FOV Radar Cones */}
                  <linearGradient id="camFov1" x1="0" y1="0" x2="0.8" y2="0.8">
                    <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.2" />
                    <stop offset="100%" stopColor="#3b82f6" stopOpacity="0.0" />
                  </linearGradient>
                  <linearGradient id="camFov2" x1="0.5" y1="0" x2="0.5" y2="1">
                    <stop offset="0%" stopColor="#10b981" stopOpacity="0.2" />
                    <stop offset="100%" stopColor="#10b981" stopOpacity="0.0" />
                  </linearGradient>
                </defs>

                {/* 1. Camera Surveillance Coverage Cones (FOV Sectors) */}
                <g className="pointer-events-none">
                  {/* CAM-01 Entrance FOV Sector */}
                  <path
                    d="M 50,240 L 20,160 L 140,160 Z"
                    fill="url(#camFov1)"
                    stroke="#3b82f6"
                    strokeWidth="1"
                    strokeDasharray="3 3"
                    strokeOpacity="0.4"
                  />
                  {/* CAM-02 Grocery Central Bay FOV Sector */}
                  <path
                    d="M 230,20 L 150,110 L 310,110 Z"
                    fill="url(#camFov2)"
                    stroke="#10b981"
                    strokeWidth="1"
                    strokeDasharray="3 3"
                    strokeOpacity="0.4"
                  />
                  {/* CAM-03 Checkout FOV Sector */}
                  <path
                    d="M 430,200 L 250,180 L 350,245 Z"
                    fill="url(#camFov1)"
                    stroke="#3b82f6"
                    strokeWidth="1"
                    strokeDasharray="3 3"
                    strokeOpacity="0.4"
                  />
                </g>

                {/* 2. Store Perimeter Double Walls */}
                <rect
                  x="20"
                  y="15"
                  width="420"
                  height="230"
                  rx="6"
                  fill="none"
                  stroke="#cbd5e1"
                  strokeWidth="3"
                  className="dark:stroke-slate-700"
                />
                <rect
                  x="23"
                  y="18"
                  width="414"
                  height="224"
                  rx="4"
                  fill="none"
                  stroke="#94a3b8"
                  strokeWidth="0.8"
                  strokeDasharray="4 2"
                  className="dark:stroke-slate-600"
                />

                {/* Store Entrance / Exit Double Gate Arc */}
                <path
                  d="M 50,245 A 35,35 0 0,0 85,210"
                  fill="none"
                  stroke="#94a3b8"
                  strokeWidth="1"
                  strokeDasharray="2 2"
                />
                <rect x="50" y="243" width="70" height="6" fill="#f8fafc" className="dark:fill-[#070b14]" />
                <text x="85" y="240" fontSize="8" fill="#64748b" textAnchor="middle" fontWeight="bold" letterSpacing="1">
                  MAIN ENTRANCE
                </text>

                {/* 3. Interior Structural Shelves & Gondola Bays */}
                <g stroke="#94a3b8" strokeWidth="1.5" fill="#ffffff" className="dark:stroke-slate-600 dark:fill-slate-800">
                  {/* Shelf Aisle 1 (Interactive) */}
                  <rect
                    x="50"
                    y="40"
                    width="80"
                    height="24"
                    rx="3"
                    className={`cursor-pointer transition-all ${
                      selectedShelfId === (shelfList[0]?.id ?? 's1')
                        ? 'stroke-blue-600 stroke-2 fill-blue-50 dark:fill-blue-950/60'
                        : 'hover:stroke-blue-500'
                    }`}
                    onClick={() => setSelectedShelfId(shelfList[0]?.id ?? 's1')}
                  />
                  {/* Shelf Aisle 2 (Interactive) */}
                  <rect
                    x="170"
                    y="40"
                    width="120"
                    height="24"
                    rx="3"
                    className={`cursor-pointer transition-all ${
                      selectedShelfId === (shelfList[1]?.id ?? 's2')
                        ? 'stroke-blue-600 stroke-2 fill-blue-50 dark:fill-blue-950/60'
                        : 'hover:stroke-blue-500'
                    }`}
                    onClick={() => setSelectedShelfId(shelfList[1]?.id ?? 's2')}
                  />
                  {/* Shelf Aisle 3 (Interactive) */}
                  <rect
                    x="330"
                    y="40"
                    width="80"
                    height="24"
                    rx="3"
                    className={`cursor-pointer transition-all ${
                      selectedShelfId === (shelfList[2]?.id ?? 's3')
                        ? 'stroke-blue-600 stroke-2 fill-blue-50 dark:fill-blue-950/60'
                        : 'hover:stroke-blue-500'
                    }`}
                    onClick={() => setSelectedShelfId(shelfList[2]?.id ?? 's3')}
                  />

                  {/* Central Display Gondola Island */}
                  <rect x="180" y="105" width="100" height="50" rx="4" stroke="#64748b" strokeWidth="1.5" />

                  {/* Left Endcap Bay */}
                  <rect x="50" y="95" width="30" height="70" rx="2" />

                  {/* Right Endcap Bay */}
                  <rect x="380" y="95" width="30" height="70" rx="2" />

                  {/* Checkout Registers Area */}
                  <rect x="230" y="195" width="55" height="18" rx="2" />
                  <rect x="310" y="195" width="55" height="18" rx="2" />
                </g>

                {/* Aisle & Zone Text Labels */}
                <g fontSize="8" fill="#475569" fontWeight="600" className="dark:fill-slate-300">
                  <text x="90" y="55" textAnchor="middle">
                    {shelfList[0]?.name ? shelfList[0].name.slice(0, 14) : 'Aisle 1 (Dry)'}
                  </text>
                  <text x="230" y="55" textAnchor="middle">
                    {shelfList[1]?.name ? shelfList[1].name.slice(0, 16) : 'Beverages & Dairy'}
                  </text>
                  <text x="370" y="55" textAnchor="middle">
                    {shelfList[2]?.name ? shelfList[2].name.slice(0, 14) : 'Snacks & Bakery'}
                  </text>
                  <text x="230" y="132" textAnchor="middle" fill="#0284c7" fontWeight="bold">
                    Promotional Island
                  </text>
                  <text x="257" y="207" textAnchor="middle">Lane 1</text>
                  <text x="337" y="207" textAnchor="middle">Lane 2</text>
                </g>

                {/* 4. Thermal Dwell Heatmap Blobs */}
                <g className="heatmap-blur pointer-events-none opacity-85">
                  <circle cx="95" cy="115" r="44" fill="url(#heatHot)" />
                  <circle cx="230" cy="130" r="50" fill="url(#heatWarm)" />
                  <circle cx="350" cy="110" r="46" fill="url(#heatHot)" />
                  <circle cx="310" cy="195" r="40" fill="url(#heatWarm)" />
                  <circle cx="85" cy="210" r="36" fill="url(#heatCool)" />
                </g>

                {/* 5. Camera Mounting Physical Pins */}
                <g fill="#2563eb" stroke="#ffffff" strokeWidth="1.5">
                  <circle cx="50" cy="240" r="4">
                    <title>CAM-01 Entrance</title>
                  </circle>
                  <circle cx="230" cy="20" r="4">
                    <title>CAM-02 Central Bay</title>
                  </circle>
                  <circle cx="430" cy="200" r="4">
                    <title>CAM-03 Checkout</title>
                  </circle>
                </g>

                {/* 6. Shopper Flow Trajectory Spline */}
                <path
                  d="M 85,235 C 80,185 85,120 110,85 C 135,50 310,50 350,90 C 385,125 360,175 320,195"
                  fill="none"
                  stroke="#2563eb"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeDasharray="5 5"
                />

                {/* Waypoint Tracking Nodes */}
                <g fill="#ffffff" stroke="#2563eb" strokeWidth="2">
                  <circle cx="85" cy="235" r="3" />
                  <circle cx="105" cy="95" r="4" fill="#3b82f6" />
                  <circle cx="230" cy="52" r="3.5" />
                  <circle cx="345" cy="75" r="4" fill="#ef4444" />
                  <circle cx="320" cy="195" r="4" fill="#10b981" />
                </g>
              </svg>
            </div>
          </div>

          {/* Card 6: Queue & Occupancy Precision Meters (Dynamic) */}
          <div className="retail-card retail-card-bracket col-span-3 flex flex-col justify-between p-3.5 bg-white border border-slate-200/80 dark:bg-[#0f172a] dark:border-white/10 overflow-hidden">
            {/* Upper Section: Queue Wait Meters */}
            <div>
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-bold text-slate-800 dark:text-slate-100">
                  Queue Telemetry
                </h3>
                <span className="text-[10px] font-semibold text-slate-400 tabular-nums">
                  {totalQueueZones} Zones
                </span>
              </div>

              {/* 2 Circular Precision Radial Gauges */}
              <div className="grid grid-cols-2 gap-2 mt-2">
                <RadialGauge
                  value={waitGaugeVal}
                  displayValue={avgWaitMins}
                  sublabel="Mins"
                  label="Wait Time"
                  color="#2563eb"
                  size={84}
                  strokeWidth={7}
                />
                <RadialGauge
                  value={clearancePct}
                  displayValue={`${clearancePct}%`}
                  sublabel="Cleared"
                  label="Clearance"
                  color="#0284c7"
                  size={84}
                  strokeWidth={7}
                />
              </div>
            </div>

            {/* Lower Section: Store Flow & Live Occupancy Density */}
            <div className="border-t border-slate-100 pt-2.5 mt-2 dark:border-white/5">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-slate-800 dark:text-slate-100">
                  Store Capacity Load
                </span>
                <span className="text-[10px] text-slate-500 font-medium tabular-nums">
                  {maxCapacity} Max
                </span>
              </div>

              {/* Capacity Bar with Milestone Notches */}
              <div className="relative mt-1.5 h-1.5 w-full rounded-sm bg-slate-100 dark:bg-slate-800 overflow-hidden">
                <div
                  className="h-full bg-blue-600 transition-all duration-500 rounded-sm"
                  style={{ width: `${occupancyDensityPct}%` }}
                />
                <div className="absolute top-0 bottom-0 left-1/2 w-px bg-white/60 dark:bg-black/40 pointer-events-none" />
                <div className="absolute top-0 bottom-0 left-[80%] w-px bg-rose-400/80 pointer-events-none" />
              </div>

              {/* 2 Lower Radial Gauges */}
              <div className="grid grid-cols-2 gap-2 mt-2">
                <RadialGauge
                  value={occupancyDensityPct}
                  displayValue={String(currentOccupancy)}
                  sublabel="Active"
                  label="In Store"
                  color="#2563eb"
                  size={84}
                  strokeWidth={7}
                />
                <RadialGauge
                  value={occupancyDensityPct}
                  displayValue={`${occupancyDensityPct}%`}
                  sublabel="Load"
                  label="Density"
                  color="#3b82f6"
                  size={84}
                  strokeWidth={7}
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
