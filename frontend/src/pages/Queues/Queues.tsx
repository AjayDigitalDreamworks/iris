import { useEffect, useMemo, useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import {
  Clock,
  Users,
  Timer,
  TrendingUp,
  Activity,
  Layers,
  Video,
  ScanFace,
  WifiOff,
} from 'lucide-react'
import { api, API_BASE } from '../../services/api'
import { useApi, useLiveEvents, usePolling } from '../../hooks/useAsync'
import {
  AsyncContent,
  Badge,
  SurfaceCard,
  MetricTile,
  EmptyState,
  fmtSeconds,
} from '../../components/ui'
import type { Camera, LiveDetections, QueueMeasurement } from '../../types/api'

const RANGES = [
  { label: '6h', value: '6h' },
  { label: '24h', value: '24h' },
  { label: '7d', value: '7d' },
]

const SNAPSHOT_MS = 1000
const DETECTIONS_MS = 2500

function streamUrl(cameraId: string): string {
  return `${API_BASE}/api/v1/streams/${cameraId}/mjpeg?fps=15&annotated=true`
}
function snapshotUrl(cameraId: string, nonce: number): string {
  return `${API_BASE}/api/v1/streams/${cameraId}/snapshot.jpg?n=${nonce}`
}

/** One queue camera tile: live stream + person-count overlay + queue status. */
function QueueCameraFeed({
  camera,
  measurement,
}: {
  camera: Camera
  measurement: QueueMeasurement | undefined
}) {
  const [mode, setMode] = useState<'mjpeg' | 'snapshot'>('mjpeg')
  const [nonce, setNonce] = useState(() => Date.now())
  const [painted, setPainted] = useState(false)
  const [persons, setPersons] = useState<number | null>(null)

  useEffect(() => {
    setMode('mjpeg')
    setPainted(false)
    setNonce(Date.now())
  }, [camera.id])

  // If the stream paints nothing within 8 s, fall back to still polling so a
  // tile can never stay permanently blank.
  useEffect(() => {
    if (mode !== 'mjpeg' || painted) return
    const timer = window.setTimeout(() => setMode('snapshot'), 8000)
    return () => window.clearTimeout(timer)
  }, [mode, painted])

  useEffect(() => {
    if (mode !== 'snapshot') return
    const timer = window.setInterval(() => setNonce(Date.now()), SNAPSHOT_MS)
    return () => window.clearInterval(timer)
  }, [mode])

  // Person detections for the CURRENT frame (memory, not history).
  useEffect(() => {
    if (camera.status === 'offline' || camera.status === 'disabled') return
    let cancelled = false
    const tick = () => {
      api
        .liveDetections(camera.id)
        .then((payload: LiveDetections) => {
          if (cancelled) return
          const count = (payload.detections ?? []).filter((d) => d.is_person).length
          setPersons(count)
        })
        .catch(() => {
          if (!cancelled) setPersons(null)
        })
    }
    tick()
    const timer = window.setInterval(tick, DETECTIONS_MS)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [camera.id, camera.status])

  const feedable = camera.status !== 'offline' && camera.status !== 'disabled' && camera.status !== 'error'

  return (
    <div className="w-[280px] shrink-0 overflow-hidden rounded-lg border border-zinc-200/80 bg-white/50 dark:border-white/5 dark:bg-zinc-900/50">
      <div className="relative h-[150px] bg-zinc-950">
        {feedable ? (
          <img
            key={`${mode}-${camera.id}`}
            src={mode === 'mjpeg' ? streamUrl(camera.id) : snapshotUrl(camera.id, nonce)}
            alt={`${camera.name} live feed`}
            className="h-full w-full object-cover"
            onLoad={() => setPainted(true)}
            onError={() => {
              setPainted(false)
              if (mode === 'mjpeg') setMode('snapshot')
            }}
          />
        ) : (
          <div className="flex h-full flex-col items-center justify-center gap-1 text-zinc-500">
            <WifiOff className="h-5 w-5" />
            <span className="text-[10px] font-semibold uppercase">{camera.status}</span>
          </div>
        )}
        <span className="absolute top-1.5 left-1.5 rounded bg-black/70 px-1.5 py-0.5 text-[9px] font-mono font-semibold text-white backdrop-blur">
          {mode === 'mjpeg' && feedable ? 'LIVE' : 'STILL'}
        </span>
        {persons !== null && feedable && (
          <span className="absolute top-1.5 right-1.5 flex items-center gap-1 rounded bg-black/70 px-1.5 py-0.5 text-[9px] font-bold text-white backdrop-blur">
            <ScanFace className="h-3 w-3 text-indigo-400" /> {persons} in frame
          </span>
        )}
        {measurement && (
          <span className="absolute bottom-1.5 left-1.5 rounded bg-black/70 px-1.5 py-0.5 text-[9px] font-bold text-white backdrop-blur">
            Queue: {measurement.current_queue_length} · {measurement.status.toUpperCase()}
          </span>
        )}
      </div>
      <div className="flex items-center justify-between gap-2 px-2.5 py-1.5">
        <span className="truncate text-[11px] font-bold text-zinc-800 dark:text-zinc-100" title={camera.name}>
          {camera.name}
        </span>
        {measurement ? (
          <Badge value={measurement.status} />
        ) : (
          <span className="text-[9px] uppercase text-zinc-400">no readings</span>
        )}
      </div>
    </div>
  )
}

export default function Queues() {
  const [range, setRange] = useState('24h')
  const queues = useApi<QueueMeasurement[]>(() => api.queues({ range }), [range])
  useLiveEvents((event) => {
    if (event.event_type.includes('queue')) queues.reload()
  })
  usePolling(() => queues.reload(), 15000)

  // Every camera that participates in queue monitoring: an explicit queue-role
  // camera, a checkout camera, or any camera that has an active queue zone.
  const cameras = useApi(() => api.cameras({ limit: 100 }), [])
  const [queueZoneNames, setQueueZoneNames] = useState<Record<string, string[]>>({})
  const [zonesLoaded, setZonesLoaded] = useState(false)

  useEffect(() => {
    const items = cameras.data?.items ?? []
    if (!cameras.data) return
    let cancelled = false
    Promise.all(
      items.map((camera) =>
        api
          .zones(camera.id)
          .then((zones) => [
            camera.id,
            zones.filter((z) => z.zone_type === 'queue' && z.is_active).map((z) => z.name),
          ] as const),
      ),
    )
      .then((pairs) => {
        if (cancelled) return
        setQueueZoneNames(Object.fromEntries(pairs))
        setZonesLoaded(true)
      })
      .catch(() => {
        if (!cancelled) setZonesLoaded(true)
      })
    return () => {
      cancelled = true
    }
  }, [cameras.data])

  const queueCameras = useMemo(() => {
    const items = cameras.data?.items ?? []
    return items.filter(
      (camera) =>
        camera.role === 'queue' ||
        camera.role === 'checkout' ||
        (queueZoneNames[camera.id]?.length ?? 0) > 0,
    )
  }, [cameras.data, queueZoneNames])

  // Match a camera to its latest queue reading by the camera's queue-zone names.
  const measurementFor = (cameraId: string): QueueMeasurement | undefined => {
    const names = queueZoneNames[cameraId] ?? []
    return (queues.data ?? []).find((z) => names.includes(z.zone_name))
  }

  const chart = useMemo(() => {
    const perZone = new Map<string, { bucket: string; queue_length: number }>()
    for (const zone of queues.data ?? []) {
      for (const point of zone.points ?? []) {
        perZone.set(point.bucket, { bucket: point.bucket, queue_length: point.queue_length })
      }
    }
    return [...perZone.values()]
  }, [queues.data])

  const busiest = [...(queues.data ?? [])].sort(
    (a, b) => b.current_queue_length - a.current_queue_length,
  )

  return (
    <div className="flex h-full flex-col gap-3 overflow-hidden">
      {/* Top Header */}
      <div className="flex shrink-0 items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-base font-bold tracking-tight text-zinc-900 dark:text-zinc-100">
              Queue Intelligence
            </h1>
            <span className="inline-flex items-center gap-1 rounded bg-indigo-500/10 px-1.5 py-0.5 text-[10px] font-bold text-indigo-600 dark:text-indigo-400">
              <Clock className="h-3 w-3" /> Service Latency
            </span>
          </div>
          <p className="truncate text-[11px] text-zinc-500 dark:text-zinc-400">
            Live queue cameras, wait-time estimates and congestion tracking across checkout zones
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

      {/* Live queue-camera strip — every queue camera with its stream + detections */}
      <div className="shrink-0">
        <div className="mb-1.5 flex items-center gap-2">
          <Video className="h-3.5 w-3.5 text-indigo-500" />
          <h2 className="text-[11px] font-bold uppercase tracking-wide text-zinc-600 dark:text-zinc-300">
            Queue Cameras
          </h2>
          <span className="rounded bg-indigo-500/10 px-1.5 py-0.5 text-[10px] font-bold text-indigo-600 dark:text-indigo-400">
            {queueCameras.length}
          </span>
          {queueCameras.length === 0 && zonesLoaded && (
            <span className="text-[10px] text-zinc-400">
              Add a camera with the QUEUE role (Cameras page) to see its live stream here
            </span>
          )}
        </div>
        {queueCameras.length === 0 ? (
          <div className="rounded-lg border border-dashed border-zinc-300/70 px-3 py-3 text-center text-[11px] text-zinc-400 dark:border-white/10">
            No queue cameras yet — register one with role QUEUE or CHECKOUT
          </div>
        ) : (
          <div className="custom-scrollbar flex gap-2.5 overflow-x-auto pb-1">
            {queueCameras.map((camera) => (
              <QueueCameraFeed
                key={camera.id}
                camera={camera}
                measurement={measurementFor(camera.id)}
              />
            ))}
          </div>
        )}
      </div>

      <AsyncContent
        state={queues}
        empty="No queue zones configured"
        isEmpty={(data) => data.length === 0}
        rows={3}
      >
        {(data) => (
          <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-hidden">
            {/* 4 Compact Metric Tiles (Shrink-0) */}
            <div className="grid shrink-0 grid-cols-2 gap-2.5 lg:grid-cols-4">
              <MetricTile
                label="Monitored Zones"
                value={data.length}
                hint={`${data.filter((z) => z.status !== 'normal').length} congested zones`}
                icon={<Layers className="h-4 w-4 text-indigo-500" />}
              />
              <MetricTile
                label="Longest Queue Now"
                value={busiest[0]?.current_queue_length ?? 0}
                hint={busiest[0]?.zone_name ?? 'All zones clear'}
                tone={(busiest[0]?.current_queue_length ?? 0) > 4 ? 'warn' : 'default'}
                icon={<Users className="h-4 w-4 text-amber-500" />}
              />
              <MetricTile
                label="Max Wait Estimate"
                value={fmtSeconds(Math.max(0, ...data.map((z) => z.estimated_wait_seconds ?? 0)))}
                hint="Expected service duration"
                icon={<Timer className="h-4 w-4 text-indigo-500" />}
              />
              <MetricTile
                label="Peak in Range"
                value={Math.max(0, ...data.map((z) => z.peak_queue_length))}
                hint="Maximum recorded surge"
                icon={<TrendingUp className="h-4 w-4 text-zinc-400" />}
              />
            </div>

            {/* 2-Zone Workspace (Fills Remaining Viewport) */}
            <div className="grid min-h-0 flex-1 grid-cols-1 gap-3 overflow-hidden lg:grid-cols-12">
              {/* Left Zone: Current Readings List (7 cols) */}
              <SurfaceCard
                title="Current Zone Readings"
                subtitle="Live person counts and service station occupancies"
                icon={<Clock className="h-4 w-4 text-indigo-500" />}
                scrollable
                className="flex-1 min-h-0 overflow-hidden lg:col-span-7"
              >
                <div className="space-y-2">
                  {busiest.map((zone) => (
                    <div
                      key={zone.zone_id}
                      className="rounded-lg border border-zinc-200/70 bg-white/40 p-2.5 transition-colors dark:border-white/5 dark:bg-zinc-900/40"
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-xs font-bold text-zinc-800 dark:text-zinc-200">
                          {zone.zone_name}
                        </span>
                        <Badge value={zone.status} />
                      </div>

                      <div className="mt-2 grid grid-cols-4 gap-1.5 text-center text-[10px]">
                        <div className="rounded border border-zinc-200/50 bg-zinc-50/50 p-1.5 dark:border-white/5 dark:bg-zinc-800/30">
                          <span className="text-zinc-400">In Queue</span>
                          <p className="mt-0.5 font-mono text-sm font-bold text-zinc-900 dark:text-zinc-100">
                            {zone.current_queue_length}
                          </p>
                        </div>
                        <div className="rounded border border-zinc-200/50 bg-zinc-50/50 p-1.5 dark:border-white/5 dark:bg-zinc-800/30">
                          <span className="text-zinc-400">At Counter</span>
                          <p className="mt-0.5 font-mono text-sm font-bold text-zinc-900 dark:text-zinc-100">
                            {zone.service_zone_occupancy}
                          </p>
                        </div>
                        <div className="rounded border border-zinc-200/50 bg-zinc-50/50 p-1.5 dark:border-white/5 dark:bg-zinc-800/30">
                          <span className="text-zinc-400">Stations</span>
                          <p className="mt-0.5 font-mono text-sm font-bold text-zinc-900 dark:text-zinc-100">
                            {zone.service_stations}
                          </p>
                        </div>
                        <div className="rounded border border-zinc-200/50 bg-zinc-50/50 p-1.5 dark:border-white/5 dark:bg-zinc-800/30">
                          <span className="text-zinc-400">Wait (Est.)</span>
                          <p className="mt-0.5 font-mono text-sm font-bold text-indigo-600 dark:text-indigo-400">
                            {fmtSeconds(zone.estimated_wait_seconds)}
                          </p>
                        </div>
                      </div>

                      <p className="mt-2 text-[10px] text-zinc-400">
                        Thresholds: busy ≥ {zone.busy_threshold}, critical ≥ {zone.critical_threshold}. {zone.note}
                      </p>
                    </div>
                  ))}
                </div>
              </SurfaceCard>

              {/* Right Zone: Queue Bar Chart (5 cols) */}
              <SurfaceCard
                title="Queue Snapshots Over Time"
                subtitle="Measured queue counts"
                icon={<Activity className="h-4 w-4 text-zinc-400" />}
                className="flex-1 min-h-0 overflow-hidden lg:col-span-5"
                bodyClassName="flex flex-col p-2"
              >
                {chart.length === 0 ? (
                  <EmptyState
                    title="No queue timeline snapshots recorded in this range"
                    hint="Snapshots are captured automatically when queues are observed."
                  />
                ) : (
                  <div className="h-full w-full min-h-[160px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={chart} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
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
                        <Bar
                          dataKey="queue_length"
                          fill="#6366f1"
                          radius={[4, 4, 0, 0]}
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </SurfaceCard>
            </div>
          </div>
        )}
      </AsyncContent>
    </div>
  )
}
