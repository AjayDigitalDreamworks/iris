import { useState } from 'react'
import {
  Video,
  Plus,
  Play,
  Square,
  RotateCw,
  Trash2,
  AlertTriangle,
  X,
} from 'lucide-react'
import { api } from '../../services/api'
import { useApi, usePolling } from '../../hooks/useAsync'
import {
  AsyncContent,
  Badge,
  SurfaceCard,
  ActionButton,
  fmtNumber,
  fmtTime,
} from '../../components/ui'
import type { Camera } from '../../types/api'

const ROLE_HINTS: Record<string, string> = {
  entrance: 'Counts people crossing the entrance line (footfall)',
  shelf: 'Watches shelves for stock levels & planogram compliance',
  checkout: 'Watches the checkout counter queue (queue length + wait)',
  queue: 'Queue-only camera: raises OPEN ANOTHER COUNTER alerts when the queue is too long — appears on the Queues page with a live stream',
  other: 'General-purpose camera (occupancy only)',
}

const BLANK = {
  name: '',
  role: 'shelf',
  source_type: 'ip_webcam',
  capture_fps: 5,
  is_enabled: true,
  location: '',
  stream_url: '',
  file_path: '',
  image_path: '',
}

export default function Cameras() {
  const cameras = useApi(() => api.cameras({ limit: 100 }), [])
  // A store_id is required to register a camera, and with zero cameras the
  // list cannot provide one — ask the backend for the stores directly.
  const stores = useApi(() => api.stores(), [])
  const [form, setForm] = useState({ ...BLANK })
  const [showForm, setShowForm] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const storeId = stores.data?.[0]?.id ?? cameras.data?.items?.[0]?.store_id ?? ''

  usePolling(() => cameras.reload(), 8000)

  async function run(key: string, action: () => Promise<unknown>) {
    setBusy(key)
    setError(null)
    try {
      await action()
      cameras.reload()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(null)
    }
  }

  async function submit(event: React.FormEvent) {
    event.preventDefault()
    if (!storeId) {
      setError('No store found. Run scripts/seed.py to create one, then reload.')
      return
    }
    const body: Record<string, unknown> = {
      store_id: storeId,
      name: form.name,
      role: form.role,
      source_type: form.source_type,
      capture_fps: Number(form.capture_fps),
      is_enabled: form.is_enabled,
      location: form.location || null,
    }
    if (form.source_type === 'rtsp') body.stream_url = form.stream_url
    if (form.source_type === 'url') body.stream_url = form.stream_url
    if (form.source_type === 'ip_webcam') body.stream_url = form.stream_url
    if (form.source_type === 'video_file') body.file_path = form.file_path
    if (form.source_type === 'webcam' && form.file_path.trim()) {
      // A webcam stores its DEVICE INDEX in file_path ("0", "1", "webcam:2").
      body.file_path = form.file_path.trim()
    }
    if (form.source_type === 'image') body.image_path = form.image_path

    await run('create', async () => {
      await api.createCamera(body)
      setForm({ ...BLANK })
      setShowForm(false)
    })
  }

  return (
    <div className="flex h-full flex-col gap-3 overflow-hidden">
      {/* Top Header */}
      <div className="flex shrink-0 items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-base font-bold tracking-tight text-zinc-900 dark:text-zinc-100">
              Camera Devices
            </h1>
            <span className="inline-flex items-center gap-1 rounded bg-indigo-500/10 px-1.5 py-0.5 text-[10px] font-bold text-indigo-600 dark:text-indigo-400">
              <Video className="h-3 w-3" /> Sensors
            </span>
          </div>
          <p className="truncate text-[11px] text-zinc-500 dark:text-zinc-400">
            Hardware capture devices, RTSP streams, and inferencing worker status
          </p>
        </div>

        <ActionButton
          variant={showForm ? 'secondary' : 'primary'}
          size="sm"
          onClick={() => setShowForm((v) => !v)}
          icon={showForm ? <X className="h-3.5 w-3.5" /> : <Plus className="h-3.5 w-3.5" />}
        >
          {showForm ? 'Close Form' : 'Add Camera'}
        </ActionButton>
      </div>

      {error && (
        <div
          role="alert"
          className="flex shrink-0 items-center gap-2 rounded-lg border border-rose-500/30 bg-rose-500/10 p-2.5 text-xs text-rose-700 dark:text-rose-300"
        >
          <AlertTriangle className="h-4 w-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Registration Card (Collapsible) */}
      {showForm && (
        <SurfaceCard
          title="Register Camera Sensor"
          subtitle="Configure video source parameters"
          icon={<Video className="h-4 w-4 text-indigo-500" />}
          className="shrink-0"
        >
          <form onSubmit={submit} className="grid gap-2.5 sm:grid-cols-4">
            <div>
              <label className="mb-1 block text-[10px] font-semibold uppercase text-zinc-500">
                Camera Name
              </label>
              <input
                required
                value={form.name}
                onChange={(e) => setForm({ ...form, name: e.target.value })}
                placeholder="Front Entrance"
                className="precision-input w-full rounded-md px-2.5 py-1.5 text-xs"
              />
            </div>

            <div>
              <label className="mb-1 block text-[10px] font-semibold uppercase text-zinc-500">
                Role
              </label>
              <select
                value={form.role}
                onChange={(e) => setForm({ ...form, role: e.target.value })}
                className="precision-input w-full rounded-md px-2.5 py-1.5 text-xs"
              >
                {[
                  ['entrance', 'ENTRANCE'],
                  ['shelf', 'SHELF'],
                  ['checkout', 'CHECKOUT'],
                  ['queue', 'QUEUE'],
                  ['other', 'OTHER'],
                ].map(([value, label]) => (
                  <option key={value} value={value}>
                    {label}
                  </option>
                ))}
              </select>
              <p className="mt-1 text-[9px] leading-tight text-zinc-500 dark:text-zinc-400">
                {ROLE_HINTS[form.role] ?? 'General-purpose camera'}
              </p>
            </div>

            <div>
              <label className="mb-1 block text-[10px] font-semibold uppercase text-zinc-500">
                Source Type
              </label>
              <select
                value={form.source_type}
                onChange={(e) => setForm({ ...form, source_type: e.target.value })}
                className="precision-input w-full rounded-md px-2.5 py-1.5 text-xs"
              >
                {['ip_webcam', 'rtsp', 'webcam', 'url', 'video_file', 'image'].map((r) => (
                  <option key={r} value={r}>
                    {r.toUpperCase()}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="mb-1 block text-[10px] font-semibold uppercase text-zinc-500">
                Capture FPS
              </label>
              <input
                type="number"
                step="0.5"
                min="0.5"
                max="30"
                value={form.capture_fps}
                onChange={(e) => setForm({ ...form, capture_fps: Number(e.target.value) })}
                className="precision-input w-full rounded-md px-2.5 py-1.5 text-xs"
              />
            </div>

            {form.source_type === 'ip_webcam' && (
              <div className="sm:col-span-4">
                <label className="mb-1 block text-[10px] font-semibold uppercase text-zinc-500">
                  Phone IP Webcam Address
                </label>
                <input
                  required
                  value={form.stream_url}
                  onChange={(e) => setForm({ ...form, stream_url: e.target.value })}
                  placeholder="192.168.1.42:8080  or  http://192.168.1.42:8080/video"
                  className="precision-input w-full rounded-md px-2.5 py-1.5 text-xs font-mono"
                />
                <p className="mt-1 text-[10px] text-zinc-500 dark:text-zinc-400">
                  Install the free <span className="font-semibold">IP Webcam</span> app on the
                  Android phone, tap <span className="font-semibold">Start server</span>, then
                  enter the address it shows (Wi-Fi IP, e.g. 192.168.1.42:8080). Phone and this
                  machine must be on the same network.
                </p>
              </div>
            )}

            {form.source_type === 'rtsp' && (
              <div className="sm:col-span-4">
                <label className="mb-1 block text-[10px] font-semibold uppercase text-zinc-500">
                  RTSP Stream URL (encrypted)
                </label>
                <input
                  required
                  type="password"
                  value={form.stream_url}
                  onChange={(e) => setForm({ ...form, stream_url: e.target.value })}
                  placeholder="rtsp://user:password@host:554/live/ch0"
                  className="precision-input w-full rounded-md px-2.5 py-1.5 text-xs font-mono"
                />
              </div>
            )}

            {form.source_type === 'url' && (
              <div className="sm:col-span-4">
                <label className="mb-1 block text-[10px] font-semibold uppercase text-zinc-500">
                  Video Link (YouTube, Vimeo, MP4)
                </label>
                <input
                  required
                  value={form.stream_url}
                  onChange={(e) => setForm({ ...form, stream_url: e.target.value })}
                  placeholder="https://www.youtube.com/watch?v=..."
                  className="precision-input w-full rounded-md px-2.5 py-1.5 text-xs font-mono"
                />
              </div>
            )}

            {form.source_type === 'video_file' && (
              <div className="sm:col-span-4">
                <label className="mb-1 block text-[10px] font-semibold uppercase text-zinc-500">
                  Video File Path
                </label>
                <input
                  required
                  value={form.file_path}
                  onChange={(e) => setForm({ ...form, file_path: e.target.value })}
                  placeholder="C:/media/sample.mp4  or  /var/media/sample.mp4"
                  className="precision-input w-full rounded-md px-2.5 py-1.5 text-xs font-mono"
                />
                <p className="mt-1 text-[10px] text-zinc-500 dark:text-zinc-400">
                  A prerecorded clip plays on loop at wall-clock speed.
                </p>
              </div>
            )}

            {form.source_type === 'webcam' && (
              <div className="sm:col-span-4">
                <label className="mb-1 block text-[10px] font-semibold uppercase text-zinc-500">
                  Device Index (optional)
                </label>
                <input
                  value={form.file_path}
                  onChange={(e) => setForm({ ...form, file_path: e.target.value })}
                  placeholder="0"
                  className="precision-input w-full rounded-md px-2.5 py-1.5 text-xs font-mono"
                />
                <p className="mt-1 text-[10px] text-zinc-500 dark:text-zinc-400">
                  0 = default camera, 1 = second camera, and so on. Leave empty to use
                  the default camera.
                </p>
              </div>
            )}

            {form.source_type === 'image' && (
              <div className="sm:col-span-4">
                <label className="mb-1 block text-[10px] font-semibold uppercase text-zinc-500">
                  Image Path
                </label>
                <input
                  required
                  value={form.image_path}
                  onChange={(e) => setForm({ ...form, image_path: e.target.value })}
                  placeholder="/var/media/sample.jpg"
                  className="precision-input w-full rounded-md px-2.5 py-1.5 text-xs font-mono"
                />
              </div>
            )}

            <div className="flex items-center gap-2 sm:col-span-4 pt-1">
              <ActionButton type="submit" size="xs" disabled={busy === 'create'}>
                {busy === 'create' ? 'Saving…' : 'Register Sensor'}
              </ActionButton>
              <ActionButton
                type="button"
                variant="ghost"
                size="xs"
                onClick={() => setShowForm(false)}
              >
                Cancel
              </ActionButton>
            </div>
          </form>
        </SurfaceCard>
      )}

      {/* Main Contained Scrollable Cameras Grid */}
      <div className="custom-scrollbar flex-1 min-h-0 overflow-y-auto">
        <AsyncContent state={cameras} empty="No cameras configured" isEmpty={(p) => p.items.length === 0} rows={3}>
          {(page) => (
            <div className="grid gap-3 sm:grid-cols-2">
              {page.items.map((camera: Camera) => (
                <CameraCard
                  key={camera.id}
                  camera={camera}
                  busy={busy}
                  onAction={(action) =>
                    run(`${camera.id}:${action}`, () =>
                      action === 'start'
                        ? api.startCamera(camera.id)
                        : action === 'stop'
                          ? api.stopCamera(camera.id)
                          : action === 'restart'
                            ? api.restartCamera(camera.id)
                            : action === 'zones'
                              ? api.provisionZones(camera.id)
                              : api.deleteCamera(camera.id),
                    )
                  }
                />
              ))}
            </div>
          )}
        </AsyncContent>
      </div>

      {stores.error && <p className="shrink-0 text-[10px] text-zinc-400">{stores.error}</p>}
    </div>
  )
}

function CameraCard({
  camera,
  busy,
  onAction,
}: {
  camera: Camera
  busy: string | null
  onAction: (action: 'start' | 'stop' | 'restart' | 'zones' | 'delete') => Promise<unknown>
}) {
  const running = camera.status === 'online' || camera.status === 'connecting'
  return (
    <SurfaceCard className="h-full justify-between">
      <div className="space-y-2.5">
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <p className="truncate text-xs font-bold text-zinc-900 dark:text-zinc-100">
              {camera.name}
            </p>
            <p className="truncate text-[10px] text-zinc-500 dark:text-zinc-400">
              <span className="capitalize">{camera.role}</span> · {camera.source_type}
              {camera.location ? ` · ${camera.location}` : ''}
            </p>
          </div>
          <Badge value={camera.status} />
        </div>

        {/* 6-Cell Telemetry Matrix */}
        <div className="grid grid-cols-3 gap-1.5 text-center text-[10px]">
          <div className="rounded border border-zinc-200/50 bg-zinc-50/50 p-1 dark:border-white/5 dark:bg-zinc-800/30">
            <span className="text-zinc-400">Target</span>
            <p className="mt-0.5 font-mono font-semibold text-zinc-700 dark:text-zinc-300">
              {camera.capture_fps} FPS
            </p>
          </div>
          <div className="rounded border border-zinc-200/50 bg-zinc-50/50 p-1 dark:border-white/5 dark:bg-zinc-800/30">
            <span className="text-zinc-400">Resolution</span>
            <p className="mt-0.5 font-mono font-semibold text-zinc-700 dark:text-zinc-300">
              {camera.width ?? '—'}×{camera.height ?? '—'}
            </p>
          </div>
          <div className="rounded border border-zinc-200/50 bg-zinc-50/50 p-1 dark:border-white/5 dark:bg-zinc-800/30">
            <span className="text-zinc-400">Detections</span>
            <p className="mt-0.5 font-mono font-semibold text-indigo-600 dark:text-indigo-400">
              {camera.total_frames}
            </p>
          </div>
          <div className="rounded border border-zinc-200/50 bg-zinc-50/50 p-1 dark:border-white/5 dark:bg-zinc-800/30">
            <span className="text-zinc-400">Dropped</span>
            <p className="mt-0.5 font-mono font-semibold text-zinc-700 dark:text-zinc-300">
              {camera.dropped_frames}
            </p>
          </div>
          <div className="rounded border border-zinc-200/50 bg-zinc-50/50 p-1 dark:border-white/5 dark:bg-zinc-800/30">
            <span className="text-zinc-400">Reconnects</span>
            <p className="mt-0.5 font-mono font-semibold text-zinc-700 dark:text-zinc-300">
              {camera.reconnect_count}
            </p>
          </div>
          <div className="rounded border border-zinc-200/50 bg-zinc-50/50 p-1 dark:border-white/5 dark:bg-zinc-800/30">
            <span className="text-zinc-400">Native</span>
            <p className="mt-0.5 font-mono font-semibold text-zinc-700 dark:text-zinc-300">
              {fmtNumber(camera.native_fps, 1)} FPS
            </p>
          </div>
        </div>

        {camera.stream_url_preview && (
          <p className="truncate rounded bg-zinc-100 px-2 py-0.5 font-mono text-[10px] text-zinc-500 dark:bg-zinc-900 dark:text-zinc-400">
            {camera.stream_url_preview}
          </p>
        )}

        {camera.last_error && (
          <p className="rounded bg-rose-500/10 p-1.5 text-[10px] text-rose-600 dark:text-rose-400">
            {camera.last_error}
          </p>
        )}

        <div className="flex items-center justify-between pt-1 text-[10px] text-zinc-400 border-t border-zinc-200/60 dark:border-white/5">
          <span>Frame: {fmtTime(camera.last_frame_at)}</span>
          <div className="flex items-center gap-1.5">
            {running ? (
              <ActionButton
                variant="secondary"
                size="xs"
                disabled={busy !== null}
                onClick={() => onAction('stop')}
                icon={<Square className="h-3 w-3 fill-current" />}
              >
                Stop
              </ActionButton>
            ) : (
              <ActionButton
                variant="primary"
                size="xs"
                disabled={busy !== null || !camera.is_enabled}
                onClick={() => onAction('start')}
                icon={<Play className="h-3 w-3 fill-current" />}
              >
                Start
              </ActionButton>
            )}
            <ActionButton
              variant="secondary"
              size="xs"
              disabled={busy !== null}
              onClick={() => onAction('restart')}
              icon={<RotateCw className="h-3 w-3" />}
            >
              Restart
            </ActionButton>
            <ActionButton
              variant="secondary"
              size="xs"
              disabled={busy !== null}
              onClick={() => onAction('zones')}
            >
              Auto-zones
            </ActionButton>
            <ActionButton
              variant="danger"
              size="xs"
              disabled={busy !== null}
              onClick={() => onAction('delete')}
              icon={<Trash2 className="h-3 w-3" />}
            >
              Delete
            </ActionButton>
          </div>
        </div>
      </div>
    </SurfaceCard>
  )
}
