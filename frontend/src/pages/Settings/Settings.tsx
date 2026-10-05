import { useEffect, useState } from 'react'
import {
  Sliders,
  CheckCircle2,
  Save,
  Server,
  Loader2,
} from 'lucide-react'
import { api } from '../../services/api'
import { useApi } from '../../hooks/useAsync'
import {
  AsyncContent,
  SurfaceCard,
  ActionButton,
} from '../../components/ui'

export default function SettingsPage() {
  const config = useApi(() => api.config(), [])
  const thresholds = useApi(() => api.queueThresholds(), [])
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState<string | null>(null)
  const [form, setForm] = useState({ busy: 4, critical: 7, service_time_seconds: 150 })

  useEffect(() => {
    const data = thresholds.data
    if (!data) return
    setForm({
      busy: Number((data as Record<string, unknown>).busy ?? 4),
      critical: Number((data as Record<string, unknown>).critical ?? 7),
      service_time_seconds: Number((data as Record<string, unknown>).service_time_seconds ?? 150),
    })
  }, [thresholds.data])

  async function save() {
    setBusy(true)
    setMessage(null)
    try {
      await api.updateQueueThresholds({
        busy: Number(form.busy),
        critical: Number(form.critical),
        service_time_seconds: Number(form.service_time_seconds),
      })
      thresholds.reload()
      setMessage('Thresholds saved.')
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="flex h-full flex-col gap-3 overflow-hidden">
      {/* Top Header */}
      <div className="shrink-0">
        <div className="flex items-center gap-2">
          <h1 className="text-base font-bold tracking-tight text-zinc-900 dark:text-zinc-100">
            System Settings
          </h1>
          <span className="inline-flex items-center gap-1 rounded bg-indigo-500/10 px-1.5 py-0.5 text-[10px] font-bold text-indigo-600 dark:text-indigo-400">
            <Sliders className="h-3 w-3" /> Runtime Parameters
          </span>
        </div>
        <p className="truncate text-[11px] text-zinc-500 dark:text-zinc-400">
          Global baseline heuristics and server environment variables
        </p>
      </div>

      {/* 2-Zone Workspace (Fills Remaining Viewport) */}
      <div className="grid min-h-0 flex-1 grid-cols-1 gap-3 overflow-hidden lg:grid-cols-12">
        {/* Left Area (5 cols): Threshold Configuration */}
        <SurfaceCard
          title="Queue Wait Thresholds"
          subtitle="Heuristic thresholds applied to queues"
          icon={<Sliders className="h-4 w-4 text-indigo-500" />}
          className="flex-1 min-h-0 overflow-hidden lg:col-span-5"
        >
          <AsyncContent state={thresholds} empty="Thresholds unavailable" rows={2}>
            {() => (
              <div className="space-y-3">
                <div>
                  <label className="mb-1 block text-[10px] font-semibold uppercase text-zinc-500">
                    Busy Threshold (Persons)
                  </label>
                  <input
                    type="number"
                    min="1"
                    value={form.busy}
                    onChange={(e) => setForm({ ...form, busy: Number(e.target.value) })}
                    className="precision-input w-full rounded-md px-2.5 py-1.5 text-xs font-semibold"
                  />
                  <p className="mt-0.5 text-[10px] text-zinc-400">
                    Triggers warning status when queue length reaches this point.
                  </p>
                </div>

                <div>
                  <label className="mb-1 block text-[10px] font-semibold uppercase text-zinc-500">
                    Critical Threshold (Persons)
                  </label>
                  <input
                    type="number"
                    min="1"
                    value={form.critical}
                    onChange={(e) => setForm({ ...form, critical: Number(e.target.value) })}
                    className="precision-input w-full rounded-md px-2.5 py-1.5 text-xs font-semibold"
                  />
                  <p className="mt-0.5 text-[10px] text-zinc-400">
                    Triggers critical alert when queue length exceeds this count.
                  </p>
                </div>

                <div>
                  <label className="mb-1 block text-[10px] font-semibold uppercase text-zinc-500">
                    Mean Service Time (Seconds)
                  </label>
                  <input
                    type="number"
                    min="1"
                    value={form.service_time_seconds}
                    onChange={(e) =>
                      setForm({ ...form, service_time_seconds: Number(e.target.value) })
                    }
                    className="precision-input w-full rounded-md px-2.5 py-1.5 text-xs font-semibold"
                  />
                  <p className="mt-0.5 text-[10px] text-zinc-400">
                    Expected transaction time per customer in queue model.
                  </p>
                </div>

                <div className="flex items-center gap-2 pt-2">
                  <ActionButton
                    variant="primary"
                    size="sm"
                    onClick={save}
                    disabled={busy}
                    icon={busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Save className="h-3.5 w-3.5" />}
                  >
                    {busy ? 'Saving…' : 'Save Thresholds'}
                  </ActionButton>
                  {message && (
                    <span className="flex items-center gap-1 text-[11px] font-semibold text-emerald-600 dark:text-emerald-400">
                      <CheckCircle2 className="h-3.5 w-3.5" />
                      {message}
                    </span>
                  )}
                </div>
              </div>
            )}
          </AsyncContent>
        </SurfaceCard>

        {/* Right Area (7 cols): Server Environment Variables Table */}
        <SurfaceCard
          title="Runtime Server Configuration"
          subtitle="Effective backend settings"
          icon={<Server className="h-4 w-4 text-zinc-400" />}
          scrollable
          className="flex-1 min-h-0 overflow-hidden lg:col-span-7"
        >
          <AsyncContent state={config} empty="Configuration unavailable" rows={4}>
            {(data) => (
              <div className="space-y-1">
                {Object.entries(data as Record<string, unknown>).map(([key, value]) => (
                  <div
                    key={key}
                    className="flex items-start justify-between gap-3 rounded border border-zinc-200/50 bg-white/40 p-2 text-xs transition-colors dark:border-white/5 dark:bg-zinc-900/40"
                  >
                    <span className="font-mono text-[11px] font-semibold text-indigo-600 dark:text-indigo-400">
                      {key}
                    </span>
                    <span className="max-w-[280px] break-all text-right font-mono text-[11px] text-zinc-600 sm:max-w-md dark:text-zinc-300">
                      {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </AsyncContent>
        </SurfaceCard>
      </div>
    </div>
  )
}
