import { useMemo, useState } from 'react'
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import {
  Users,
  LogIn,
  LogOut,
  Clock,
  Activity,
  Shield,
} from 'lucide-react'
import { api } from '../../services/api'
import { useApi, useLiveEvents, usePolling } from '../../hooks/useAsync'
import {
  AsyncContent,
  SurfaceCard,
  MetricTile,
  EmptyState,
  fmtNumber,
  fmtSeconds,
  fmtTime,
} from '../../components/ui'

interface FootfallPoint {
  bucket: string
  entries: number
  exits: number
  occupancy: number | null
}

interface Footfall {
  camera_id: string | null
  camera_name: string | null
  granularity: string
  total_entries: number
  total_exits: number
  current_occupancy: number
  peak_occupancy: number
  mean_dwell_seconds: number | null
  max_dwell_seconds: number | null
  range_start: string
  range_end: string
  points: FootfallPoint[]
  note?: string
}

const RANGES = [
  { label: '6h', value: '6h' },
  { label: '24h', value: '24h' },
  { label: '7d', value: '7d' },
]

export default function Customers() {
  const [range, setRange] = useState('24h')
  const footfall = useApi<Footfall>(
    () => api.footfall({ range }) as unknown as Promise<Footfall>,
    [range],
  )
  const occupancy = useApi<Record<string, unknown>>(() => api.occupancy(), [])
  useLiveEvents(() => footfall.reload())
  usePolling(() => footfall.reload(), 20000)

  const chart = useMemo(
    () =>
      (footfall.data?.points ?? []).map((point) => ({
        bucket: point.bucket,
        Entries: point.entries,
        Exits: point.exits,
      })),
    [footfall.data],
  )

  const activeTracks = (occupancy.data?.active_tracks as number) ?? 0

  return (
    <div className="flex h-full flex-col gap-3 overflow-hidden">
      {/* Top Header */}
      <div className="flex shrink-0 items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-base font-bold tracking-tight text-zinc-900 dark:text-zinc-100">
              Customer Analytics
            </h1>
            <span className="inline-flex items-center gap-1 rounded bg-indigo-500/10 px-1.5 py-0.5 text-[10px] font-bold text-indigo-600 dark:text-indigo-400">
              <Shield className="h-3 w-3" /> Anonymous Vectors
            </span>
          </div>
          <p className="truncate text-[11px] text-zinc-500 dark:text-zinc-400">
            Anonymous traffic counts from temporary tracking IDs without identity markers
          </p>
        </div>

        {/* Range Selector */}
        <div className="surface-card flex items-center rounded-lg p-0.5">
          {RANGES.map((option) => {
            const active = range === option.value
            return (
              <button
                key={option.value}
                type="button"
                onClick={() => setRange(option.value)}
                className={`rounded-md px-2.5 py-1 text-xs font-semibold transition-all ${
                  active
                    ? 'bg-indigo-600 text-white shadow-sm'
                    : 'text-zinc-600 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100'
                }`}
              >
                {option.label}
              </button>
            )
          })}
        </div>
      </div>

      <AsyncContent state={footfall} empty="No footfall data for this range" rows={4}>
        {(data) => (
          <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-hidden">
            {/* 4 Compact KPI Metrics (Shrink-0) */}
            <div className="grid shrink-0 grid-cols-2 gap-2.5 lg:grid-cols-4">
              <MetricTile
                label="Customer Entries"
                value={data.total_entries}
                hint={`${range} window count`}
                icon={<LogIn className="h-4 w-4 text-indigo-500" />}
              />
              <MetricTile
                label="Customer Exits"
                value={data.total_exits}
                hint="Verified crossing events"
                icon={<LogOut className="h-4 w-4 text-zinc-400" />}
              />
              <MetricTile
                label="Current Occupancy"
                value={data.current_occupancy}
                tone="good"
                hint="Store-wide active count"
                icon={<Users className="h-4 w-4 text-emerald-500" />}
              />
              <MetricTile
                label="Mean Dwell Time"
                value={fmtSeconds(data.mean_dwell_seconds)}
                hint={data.mean_dwell_seconds === null ? 'No completed tracks yet' : 'Average customer visit'}
                icon={<Clock className="h-4 w-4 text-amber-500" />}
              />
            </div>

            {/* 2-Zone Workspace (Fills Remaining Viewport) */}
            <div className="grid min-h-0 flex-1 grid-cols-1 gap-3 overflow-hidden lg:grid-cols-12">
              {/* Primary Chart (8 cols) */}
              <SurfaceCard
                title="Entries vs Exits Velocity"
                subtitle={`${data.granularity} buckets · ${fmtTime(data.range_start)} to ${fmtTime(data.range_end)}`}
                icon={<Activity className="h-4 w-4 text-indigo-500" />}
                actions={
                  <div className="flex items-center gap-3 text-[11px] font-semibold">
                    <div className="flex items-center gap-1.5">
                      <span className="h-2 w-2 rounded-full bg-indigo-500" />
                      <span className="text-zinc-600 dark:text-zinc-300">Entries</span>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <span className="h-2 w-2 rounded-full bg-amber-500" />
                      <span className="text-zinc-600 dark:text-zinc-300">Exits</span>
                    </div>
                  </div>
                }
                className="flex-1 min-h-0 overflow-hidden lg:col-span-8"
                bodyClassName="flex flex-col p-2"
              >
                {chart.length === 0 ? (
                  <EmptyState
                    title="No line crossings registered in this range"
                    hint="Crossings appear when a tracked person crosses a designated entrance polygon."
                  />
                ) : (
                  <div className="h-full w-full min-h-[160px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={chart} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="currentColor" className="text-zinc-200 dark:text-zinc-800" opacity={0.6} />
                        <XAxis
                          dataKey="bucket"
                          tick={{ fontSize: 10, fill: 'currentColor' }}
                          className="text-zinc-400"
                          stroke="currentColor"
                          opacity={0.3}
                        />
                        <YAxis
                          allowDecimals={false}
                          tick={{ fontSize: 10, fill: 'currentColor' }}
                          className="text-zinc-400"
                          stroke="currentColor"
                          opacity={0.3}
                        />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: 'rgba(18, 18, 22, 0.95)',
                            borderRadius: '8px',
                            border: '1px solid rgba(255, 255, 255, 0.1)',
                            color: '#ededed',
                            fontSize: '11px',
                          }}
                        />
                        <Line
                          type="monotone"
                          dataKey="Entries"
                          stroke="#6366f1"
                          strokeWidth={2}
                          dot={false}
                        />
                        <Line
                          type="monotone"
                          dataKey="Exits"
                          stroke="#f59e0b"
                          strokeWidth={2}
                          dot={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </SurfaceCard>

              {/* Right Context Meta Panel (4 cols) */}
              <div className="flex h-full min-h-0 flex-col gap-3 overflow-hidden lg:col-span-4">
                <SurfaceCard
                  title="Live Store Occupancy"
                  subtitle="Active anonymous tracks"
                  icon={<Users className="h-4 w-4 text-emerald-500" />}
                  className="flex-1 min-h-0"
                >
                  <AsyncContent state={occupancy} empty="Occupancy unavailable" rows={2}>
                    {(occ) => (
                      <div className="flex flex-col justify-between h-full space-y-3">
                        <div className="flex items-center gap-3">
                          <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-emerald-500/10 text-emerald-600 dark:text-emerald-400">
                            <Activity className="h-6 w-6 animate-pulse" />
                          </div>
                          <div>
                            <span className="text-[10px] font-semibold uppercase text-zinc-400">
                              Active Tracks Now
                            </span>
                            <p className="text-3xl font-extrabold text-zinc-900 dark:text-zinc-100">
                              {activeTracks}
                            </p>
                          </div>
                        </div>

                        <div className="rounded-lg border border-zinc-200/60 bg-zinc-50/50 p-2.5 text-xs text-zinc-500 dark:border-white/5 dark:bg-zinc-900/40 dark:text-zinc-400">
                          Peak range occupancy: <strong className="text-zinc-800 dark:text-zinc-200">{fmtNumber(data.peak_occupancy, 0)}</strong> visitors.
                          {typeof occ.note === 'string' && <p className="mt-1">{occ.note}</p>}
                        </div>
                      </div>
                    )}
                  </AsyncContent>
                </SurfaceCard>
              </div>
            </div>
          </div>
        )}
      </AsyncContent>
    </div>
  )
}
