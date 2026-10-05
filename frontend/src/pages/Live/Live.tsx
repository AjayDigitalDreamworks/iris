import { useEffect, useMemo, useState } from 'react'
import {
  Video,
  RefreshCw,
  Eye,
  Activity,
  WifiOff,
  Radio,
  Play,
  Pause,
  Sliders,
  Package,
  Layers,
  Zap,
  Gauge,
} from 'lucide-react'
import { api, API_BASE } from '../../services/api'
import { useApi, usePolling } from '../../hooks/useAsync'
import {
  AsyncContent,
  Badge,
  SurfaceCard,
  MetricTile,
  ActionButton,
} from '../../components/ui'

import type { Camera, CameraHealth, LiveDetections, StreamTelemetry } from '../../types/api'

interface Box {
  id: string
  x1: number
  y1: number
  x2: number
  y2: number
  label: string
  confidence: number
  track: number | null
  kind: 'person' | 'product' | 'multipack' | 'other'
  color: string
}

type OverlayFilter = 'all' | 'products' | 'persons'

const SNAPSHOT_FPS_MS = 400

function streamUrl(cameraId: string, annotated: boolean = true, fps: number = 25): string {
  return `${API_BASE}/api/v1/streams/${cameraId}/mjpeg?fps=${fps}&annotated=${annotated}`
}

function snapshotUrl(cameraId: string, nonce: number): string {
  return `${API_BASE}/api/v1/streams/${cameraId}/snapshot.jpg?n=${nonce}`
}

function hasFeedSource(status: Camera['status']): boolean {
  return status !== 'disabled' && status !== 'error'
}

function CameraFeed({
  camera,
  isPaused,
  annotated = true,
  className,
}: {
  camera: Camera
  isPaused: boolean
  annotated?: boolean
  className?: string
}) {
  // 'mjpeg' streams continuously; 'snapshot' polls the JPEG endpoint. Start
  // with MJPEG and fall back to stills if no frame paints, so a camera can
  // never leave a permanently blank tile.
  const [mode, setMode] = useState<'mjpeg' | 'snapshot'>('mjpeg')
  const [nonce, setNonce] = useState(() => Date.now())
  const [painted, setPainted] = useState(false)

  // Re-probe whenever the selected camera changes.
  useEffect(() => {
    setMode('mjpeg')
    setPainted(false)
    setNonce(Date.now())
  }, [camera.id])

  // Watchdog: if MJPEG paints nothing within 8 s, switch to snapshot polling.
  useEffect(() => {
    if (isPaused || mode !== 'mjpeg' || painted) return
    const timer = window.setTimeout(() => setMode('snapshot'), 8000)
    return () => window.clearTimeout(timer)
  }, [isPaused, mode, painted, camera.id])

  // Refresh the still frame while paused or in snapshot mode.
  useEffect(() => {
    if (!isPaused && mode !== 'snapshot') return
    const timer = window.setInterval(() => setNonce(Date.now()), SNAPSHOT_FPS_MS)
    return () => window.clearInterval(timer)
  }, [isPaused, mode])

  const usingSnapshot = isPaused || mode === 'snapshot'

  if (!hasFeedSource(camera.status)) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-1.5 p-4 text-center text-xs text-zinc-400">
        <WifiOff className="h-6 w-6 text-zinc-500" />
        <span>{camera.status === 'disabled' ? 'Camera disabled' : `Status: ${camera.status}`}</span>
      </div>
    )
  }

  if (camera.status === 'offline') {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-1.5 p-4 text-center text-xs text-zinc-400">
        <WifiOff className="h-6 w-6 text-rose-400" />
        <span className="font-semibold text-zinc-300">Camera offline</span>
        <span className="text-[10px] text-zinc-500">Start the optical stream on the Cameras page</span>
        {camera.last_error && (
          <span className="mt-1 max-w-xs truncate text-[9px] text-rose-400">
            {camera.last_error}
          </span>
        )}
      </div>
    )
  }

  return (
    <>
      <img
        key={`${usingSnapshot ? 'snapshot' : 'mjpeg'}-${annotated}`}
        src={usingSnapshot ? snapshotUrl(camera.id, nonce) : streamUrl(camera.id, annotated, 25)}
        alt={`${camera.name} live feed`}
        className={className ?? 'feed-frame h-full w-full object-contain'}
        onLoad={() => setPainted(true)}
        onError={() => {
          setPainted(false)
          if (!usingSnapshot) setMode('snapshot')
        }}
      />
      <span className="absolute top-2 left-2 rounded border border-white/10 bg-black/70 px-1.5 py-0.5 text-[9px] font-mono font-semibold tracking-wide text-white backdrop-blur-md">
        {usingSnapshot ? 'STILL · 2.5fps' : 'LIVE · 25fps'}
      </span>
    </>
  )
}

function SVGOverlay({
  boxes,
  filter,
  minConfidence,
}: {
  boxes: Box[]
  filter: OverlayFilter
  minConfidence: number
}) {
  const visible = boxes.filter((b) => {
    if (b.confidence < minConfidence) return false
    if (filter === 'persons') return b.kind === 'person'
    if (filter === 'products') return b.kind === 'product' || b.kind === 'multipack'
    return true
  })

  if (visible.length === 0) return null

  return (
    <svg className="pointer-events-none absolute inset-0 h-full w-full">
      <defs>
        <filter id="box-glow" x="-10%" y="-10%" width="120%" height="120%">
          <feDropShadow dx="0" dy="1" stdDeviation="2" floodColor="#000" floodOpacity="0.8" />
        </filter>
      </defs>
      {visible.map((box) => {
        const width = Math.max(0, box.x2 - box.x1)
        const height = Math.max(0, box.y2 - box.y1)
        const strokeColor = box.color || '#34D399'

        return (
          <g key={box.id}>
            {/* Specular Bounding Box */}
            <rect
              x={`${box.x1}%`}
              y={`${box.y1}%`}
              width={`${width}%`}
              height={`${height}%`}
              fill="none"
              stroke={strokeColor}
              strokeWidth="2"
              strokeDasharray={box.kind === 'multipack' ? '4 2' : 'none'}
              className="transition-all duration-150"
              style={{ filter: 'drop-shadow(0 0 4px rgba(0, 0, 0, 0.6))' }}
            />

            {/* Label Chip */}
            <foreignObject
              x={`${box.x1}%`}
              y={`${Math.max(0, box.y1 - 6)}%`}
              width="240"
              height="28"
              className="overflow-visible"
            >
              <div
                className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[9px] font-bold tracking-wide uppercase text-white shadow-lg backdrop-blur-md"
                style={{
                  backgroundColor: `${strokeColor}dd`,
                  border: '1px solid rgba(255, 255, 255, 0.25)',
                }}
              >
                <span>{box.label}</span>
                <span className="opacity-90">{Math.round(box.confidence * 100)}%</span>
                {box.track !== null && (
                  <span className="opacity-75">#{box.track}</span>
                )}
                {box.kind === 'multipack' && (
                  <span className="rounded bg-black/40 px-1 py-0.2 text-[8px] text-amber-300">
                    PACK
                  </span>
                )}
              </div>
            </foreignObject>
          </g>
        )
      })}
    </svg>
  )
}

export default function Live() {
  const cameras = useApi(() => api.cameras({ limit: 100 }), [])
  const [selected, setSelected] = useState<string | null>(null)
  const [isPaused, setIsPaused] = useState(false)
  const [filter, setFilter] = useState<OverlayFilter>('all')
  const [confidenceSlider, setConfidenceSlider] = useState(25)
  const [overlayMode, setOverlayMode] = useState<'sync' | 'vector'>('sync')

  const items = cameras.data?.items ?? []
  // Default to a camera that is actually producing frames. Picking the first
  // row blindly selected a disabled template camera and left the Live page
  // showing "Camera disabled" on load.
  const defaultCamera = useMemo(
    () =>
      items.find((c) => c.status === 'online') ??
      items.find((c) => c.is_enabled && c.status !== 'disabled') ??
      items[0] ??
      null,
    [items],
  )
  const active = selected ?? defaultCamera?.id ?? null
  const activeCamera = useMemo(() => items.find((c) => c.id === active) ?? null, [items, active])

  // Follow the RESOLVED camera (active), not the raw selection: on first load
  // `selected` is null, which left the overlay and telemetry permanently empty.
  //
  // Boxes come from the CURRENT frame (served from worker memory), never the
  // stored detection history — replaying history draws many frames' boxes at
  // once, which is why the model appeared to detect far more people than were
  // on screen.
  const live = useApi<LiveDetections>(
    () =>
      active
        ? api.liveDetections(active)
        : Promise.resolve({ camera_id: '', detections: [], frame: {} }),
    [active],
  )

  const telemetry = useApi<StreamTelemetry>(
    () =>
      active
        ? api.streamTelemetry(active)
        : Promise.reject(new Error('no camera')),
    [active],
  )

  usePolling(() => {
    cameras.reload()
    telemetry.reload()
  }, 3000)
  // Detections refresh 5x a second so telemetry and cards track live analysis closely.
  usePolling(() => live.reload(), 200)

  const frameSize = useFrameSize(activeCamera)

  // Map the live boxes straight to overlay percentages (already pixel coords).
  const boxes: Box[] = useMemo(() => {
    const width = live.data?.frame.width ?? frameSize?.width ?? activeCamera?.width ?? 0
    const height = live.data?.frame.height ?? frameSize?.height ?? activeCamera?.height ?? 0
    if (!width || !height) return []
    return (live.data?.detections ?? [])
      .map((row, idx): Box => {
        const isPerson = row.is_person
        // Reference look: shoppers are drawn YELLOW, retail products GREEN.
        // Multipacks stay green but render with a dashed outline (SVGOverlay).
        const color = isPerson ? '#FACC15' : '#34D399'
        return {
          id: `live_${idx}`,
          x1: (row.x1 / width) * 100,
          y1: (row.y1 / height) * 100,
          x2: (row.x2 / width) * 100,
          y2: (row.y2 / height) * 100,
          label: row.sku ?? row.class_name,
          confidence: row.confidence,
          track: row.track_id,
          kind: isPerson ? 'person' : row.is_multipack ? 'multipack' : 'product',
          color,
        }
      })
      .filter((box) => box.x2 > box.x1 && box.y2 > box.y1)
  }, [live.data, frameSize, activeCamera])

  // Count breakdowns
  const productCount = boxes.filter((b) => b.kind === 'product' || b.kind === 'multipack').length
  const personCount = boxes.filter((b) => b.kind === 'person').length
  const avgConfidence =
    boxes.length > 0 ? boxes.reduce((acc, b) => acc + b.confidence, 0) / boxes.length : 0

  const classBreakdown = useMemo(() => {
    const counts: Record<string, number> = {}
    for (const b of boxes) {
      counts[b.label] = (counts[b.label] ?? 0) + 1
    }
    return counts
  }, [boxes])

  const effectiveFps = telemetry.data?.inference_fps ?? activeCamera?.capture_fps ?? 5.0
  const latencyMs = telemetry.data?.inference_latency_ms ?? 24.5
  const hwDevice = telemetry.data?.hardware.device ?? 'CPU'

  return (
    <div className="flex h-full flex-col gap-2.5 overflow-hidden">
      {/* Top Header Bar */}
      <div className="flex shrink-0 items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-base font-bold tracking-tight text-zinc-900 dark:text-zinc-100">
              YOLO11 Live Retail Vision
            </h1>
            <span className="inline-flex items-center gap-1 rounded border border-indigo-500/20 bg-indigo-500/10 px-2 py-0.5 text-[10px] font-bold text-indigo-600 dark:text-indigo-400">
              <Radio className="h-3 w-3 animate-pulse text-indigo-500" />
              <span>LIVE · {effectiveFps.toFixed(1)} FPS (YOLO11n · {hwDevice.toUpperCase()})</span>
            </span>
          </div>
          <p className="truncate text-[11px] text-zinc-500 dark:text-zinc-400">
            Real-time retail product recognition with multipack precedence and spatial telemetry
          </p>
        </div>

        <div className="flex items-center gap-2">
          <ActionButton
            variant="secondary"
            size="xs"
            onClick={() => {
              cameras.reload()
              live.reload()
              telemetry.reload()
            }}
            icon={<RefreshCw className="h-3 w-3" />}
          >
            Sync Pipeline
          </ActionButton>
        </div>
      </div>

      {/* Control Strip & Filter Bar */}
      <div className="flex shrink-0 flex-wrap items-center justify-between gap-3 rounded-lg border border-zinc-200/60 bg-white/60 p-2 backdrop-blur-md dark:border-white/5 dark:bg-zinc-900/60">
        <div className="flex items-center gap-2">
          {/* Play/Pause Button */}
          <button
            type="button"
            onClick={() => setIsPaused((prev) => !prev)}
            className={`flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-semibold transition-all ${
              isPaused
                ? 'bg-amber-500/20 text-amber-600 hover:bg-amber-500/30 dark:text-amber-400'
                : 'bg-zinc-800 text-white hover:bg-zinc-700 dark:bg-zinc-700 dark:hover:bg-zinc-600'
            }`}
          >
            {isPaused ? <Play className="h-3 w-3" /> : <Pause className="h-3 w-3" />}
            <span>{isPaused ? 'Resume' : 'Pause'}</span>
          </button>

          {/* Stream Overlay Mode: Synchronized 25 FPS AI vs Vector Overlay */}
          <div className="flex items-center rounded-lg border border-white/10 bg-zinc-800/80 p-0.5 text-xs">
            <button
              type="button"
              onClick={() => setOverlayMode('sync')}
              className={`rounded px-2 py-0.5 text-[11px] font-medium transition ${
                overlayMode === 'sync'
                  ? 'bg-emerald-600 text-white shadow-sm'
                  : 'text-zinc-400 hover:text-white'
              }`}
              title="Synchronized AI stream (Burned-in boxes, 25 FPS, zero lag)"
            >
              Sync 25FPS
            </button>
            <button
              type="button"
              onClick={() => setOverlayMode('vector')}
              className={`rounded px-2 py-0.5 text-[11px] font-medium transition ${
                overlayMode === 'vector'
                  ? 'bg-indigo-600 text-white shadow-sm'
                  : 'text-zinc-400 hover:text-white'
              }`}
              title="Raw video stream with client SVG vector overlay"
            >
              Vector SVG
            </button>
          </div>

          {/* Filter Pills */}
          <div className="flex items-center gap-1">
            {(
              [
                { id: 'all', label: 'All' },
                { id: 'products', label: 'Products (green)' },
                { id: 'persons', label: 'Persons (yellow)' },
              ] as const
            ).map((opt) => (
              <button
                key={opt.id}
                type="button"
                onClick={() => setFilter(opt.id)}
                className={`rounded px-2.5 py-1 text-[11px] font-medium transition ${
                  filter === opt.id
                    ? 'bg-indigo-600 text-white shadow-sm'
                    : 'text-zinc-600 hover:bg-zinc-100 dark:text-zinc-400 dark:hover:bg-zinc-800/60'
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>

        {/* Confidence Threshold Slider */}
        <div className="flex items-center gap-2 text-xs">
          <Sliders className="h-3.5 w-3.5 text-zinc-400" />
          <span className="text-[11px] text-zinc-500 dark:text-zinc-400">Min Conf:</span>
          <input
            type="range"
            min="10"
            max="90"
            step="5"
            value={confidenceSlider}
            onChange={(e) => setConfidenceSlider(Number(e.target.value))}
            className="h-1.5 w-24 cursor-pointer appearance-none rounded-lg bg-zinc-200 accent-indigo-600 dark:bg-zinc-700"
          />
          <span className="font-mono text-[11px] font-bold text-zinc-700 dark:text-zinc-300">
            {confidenceSlider}%
          </span>
        </div>
      </div>

      {/* Main 2-Zone Bento Grid (Viewport-Locked) */}
      <div className="grid min-h-0 flex-1 grid-cols-1 gap-3 overflow-hidden lg:grid-cols-12">
        {/* Left Primary Live Viewport (7 of 12 columns) */}
        <div className="flex h-full min-h-0 flex-col overflow-hidden lg:col-span-7">
          <SurfaceCard
            title={activeCamera ? activeCamera.name : 'Optical Feed'}
            subtitle={
              frameSize
                ? `Resolution ${frameSize.width}×${frameSize.height} · ${boxes.length} items parsed`
                : 'Calibrating video raster'
            }
            icon={<Eye className="h-4 w-4 text-indigo-500" />}
            actions={
              activeCamera && (
                <div className="flex items-center gap-1.5">
                  <span className="rounded bg-zinc-100 px-1.5 py-0.5 font-mono text-[10px] text-zinc-600 dark:bg-zinc-800 dark:text-zinc-300">
                    {activeCamera.source_type}
                  </span>
                  <Badge value={activeCamera.status} />
                </div>
              )
            }
            className="h-full flex-1"
            bodyClassName="p-2 flex flex-col justify-center bg-black/80 relative overflow-hidden"
          >
            {activeCamera ? (
              <div className="relative aspect-video w-full max-h-full overflow-hidden rounded-lg bg-zinc-950 shadow-inner">
                <CameraFeed
                  camera={activeCamera}
                  isPaused={isPaused}
                  annotated={overlayMode === 'sync'}
                  className="h-full w-full object-contain"
                />
                {overlayMode === 'vector' && (
                  <SVGOverlay
                    boxes={boxes}
                    filter={filter}
                    minConfidence={confidenceSlider / 100}
                  />
                )}

                {/* In-feed status telemetry overlay */}
                <div className="absolute bottom-2 left-2 flex items-center gap-2 rounded border border-white/10 bg-black/75 px-2.5 py-1 text-[10px] text-white shadow-lg backdrop-blur-md">
                  <Activity className="h-3 w-3 text-emerald-400" />
                  <span className="font-mono" style={{fontSize: 30}}>{productCount} products</span>
                  <span className="text-zinc-500">·</span>
                  <span className="font-mono text-indigo-300" style={{fontSize: 30}}>{personCount} shoppers</span>
                  {activeCamera.dropped_frames > 0 && (
                    <span className="text-amber-400">
                      ({activeCamera.dropped_frames} dropped)
                    </span>
                  )}
                </div>

                <div className="absolute top-2 right-2 rounded border border-white/10 bg-black/75 px-2 py-0.5 text-[9px] font-mono font-semibold text-emerald-400 backdrop-blur-md">
                  YOLO11 · REAL-TIME
                </div>
              </div>
            ) : (
              <div className="flex h-full items-center justify-center text-xs text-zinc-400">
                Select an optical sensor to view stream
              </div>
            )}
          </SurfaceCard>
        </div>

        {/* Right Retail Telemetry & Feed Panel (5 of 12 columns) */}
        <div className="flex h-full min-h-0 flex-col gap-2.5 overflow-hidden lg:col-span-5">
          {/* Performance KPI Tiles */}
          <div className="grid shrink-0 grid-cols-2 gap-2 sm:grid-cols-4">
            <MetricTile
              label="Latency"
              value={`${latencyMs.toFixed(1)} ms`}
              hint={hwDevice}
              icon={<Zap className="h-3.5 w-3.5 text-amber-500" />}
            />
            <MetricTile
              label="Pipeline FPS"
              value={`${effectiveFps.toFixed(1)}`}
              hint="Real-time"
              tone={effectiveFps >= 10 ? 'good' : 'default'}
              icon={<Gauge className="h-3.5 w-3.5 text-indigo-500" />}
            />
            <MetricTile
              label="Products"
              value={productCount}
              hint="Detected"
              tone="good"
              icon={<Package className="h-3.5 w-3.5 text-emerald-500" />}
            />
            <MetricTile
              label="Avg Confidence"
              value={`${Math.round(avgConfidence * 100)}%`}
              hint="Score"
              icon={<Activity className="h-3.5 w-3.5 text-cyan-500" />}
            />
          </div>

          {/* Class Breakdown Chips */}
          {Object.keys(classBreakdown).length > 0 && (
            <div className="flex shrink-0 flex-wrap items-center gap-1.5 rounded-lg border border-zinc-200/60 bg-white/40 p-2 dark:border-white/5 dark:bg-zinc-900/40">
              <span className="text-[10px] font-bold uppercase tracking-wider text-zinc-400">
                Classes:
              </span>
              {Object.entries(classBreakdown).map(([clsName, count]) => (
                <span
                  key={clsName}
                  className="inline-flex items-center gap-1 rounded bg-zinc-100 px-2 py-0.5 text-[10px] font-medium text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300"
                >
                  <span className="capitalize">{clsName}</span>
                  <span className="rounded-full bg-zinc-300 px-1 text-[9px] font-bold dark:bg-zinc-700">
                    {count}
                  </span>
                </span>
              ))}
            </div>
          )}

          {/* Real-Time Detection Feed (Contained Scrolling) */}
          <SurfaceCard
            title="Real-Time Detection Stream"
            subtitle={`${boxes.length} items parsed this cycle`}
            icon={<Layers className="h-4 w-4 text-emerald-500" />}
            scrollable
            className="flex-1 min-h-0"
          >
            {boxes.length === 0 ? (
              <div className="flex h-28 items-center justify-center text-xs text-zinc-400">
                No active detections above {confidenceSlider}% confidence
              </div>
            ) : (
              <div className="space-y-1.5">
                {boxes
                  .filter((b) => b.confidence >= confidenceSlider / 100)
                  .map((box) => (
                    <div
                      key={box.id}
                      className="flex items-center justify-between rounded-lg border border-zinc-200/50 bg-white/40 p-2 text-xs transition-colors hover:bg-zinc-50 dark:border-white/5 dark:bg-zinc-900/40 dark:hover:bg-zinc-800/40"
                    >
                      <div className="flex items-center gap-2 min-w-0">
                        <div
                          className="h-2 w-2 shrink-0 rounded-full"
                          style={{ backgroundColor: box.color }}
                        />
                        <div className="min-w-0">
                          <p className="truncate font-semibold text-zinc-800 dark:text-zinc-200">
                            {box.label}
                          </p>
                          <p className="text-[10px] font-mono text-zinc-400">
                            bbox [{box.x1.toFixed(0)}%, {box.y1.toFixed(0)}%, {box.x2.toFixed(0)}%, {box.y2.toFixed(0)}%]
                          </p>
                        </div>
                      </div>

                      <div className="flex shrink-0 items-center gap-1.5">
                        {box.kind === 'multipack' && (
                          <span className="rounded bg-amber-500/10 px-1.5 py-0.5 text-[9px] font-bold text-amber-600 dark:text-amber-400">
                            MULTIPACK
                          </span>
                        )}
                        <span className="font-mono font-bold text-zinc-700 dark:text-zinc-300">
                          {(box.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  ))}
              </div>
            )}
          </SurfaceCard>

          {/* Optical Sensors Mini Switcher */}
          <SurfaceCard
            title="Optical Camera Devices"
            subtitle={`${items.length} online sensors`}
            icon={<Video className="h-4 w-4 text-zinc-400" />}
            scrollable
            className="h-36 shrink-0"
          >
            <AsyncContent state={cameras} empty="No cameras configured" isEmpty={(p) => p.items.length === 0}>
              {(page) => (
                <div className="space-y-1">
                  {page.items.map((camera) => {
                    const isSelected = camera.id === active
                    return (
                      <button
                        key={camera.id}
                        type="button"
                        onClick={() => setSelected(camera.id)}
                        className={`flex w-full items-center justify-between rounded-md border p-1.5 text-left text-xs transition-all ${
                          isSelected
                            ? 'border-indigo-500/50 bg-indigo-500/10 shadow-sm'
                            : 'border-zinc-200/50 bg-white/40 hover:bg-zinc-100 dark:border-white/5 dark:bg-zinc-900/40 dark:hover:bg-zinc-800/60'
                        }`}
                      >
                        <div className="min-w-0 pr-2">
                          <p className="truncate text-xs font-semibold text-zinc-800 dark:text-zinc-200">
                            {camera.name}
                          </p>
                          <p className="truncate text-[10px] text-zinc-400 uppercase font-mono">
                            {camera.role} · {camera.source_type}
                          </p>
                        </div>
                        <Badge value={camera.status} />
                      </button>
                    )
                  })}
                </div>
              )}
            </AsyncContent>
          </SurfaceCard>
        </div>
      </div>
    </div>
  )
}

function useFrameSize(camera: Camera | null): { width: number; height: number } | null {
  const [size, setSize] = useState<{ width: number; height: number } | null>(null)
  const health = useApi<CameraHealth>(
    () => (camera ? api.cameraHealth(camera.id) : Promise.reject(new Error('no camera'))),
    [camera?.id],
  )
  // The frame raster can arrive a moment after the stream opens, so poll the
  // live health until the size is known — otherwise the overlay never renders.
  usePolling(() => health.reload(), 3000, Boolean(camera))
  const width = health.data?.width ?? camera?.width ?? 0
  const height = health.data?.height ?? camera?.height ?? 0
  useEffect(() => {
    if (width > 0 && height > 0) setSize({ width, height })
  }, [width, height])
  return size
}