import { useMemo, useState } from 'react'
import {
  AlertTriangle,
  ShieldAlert,
} from 'lucide-react'
import { api } from '../../services/api'
import { useApi, useLiveEvents, usePolling } from '../../hooks/useAsync'
import {
  AsyncContent,
  Badge,
  SurfaceCard,
  ActionButton,
  fmtTime,
} from '../../components/ui'
import type { Alert } from '../../types/api'

const STATUSES = ['all', 'active', 'acknowledged', 'resolved'] as const

export default function Alerts() {
  const [status, setStatus] = useState<(typeof STATUSES)[number]>('all')
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const alerts = useApi(
    () => api.alerts({ status: status === 'all' ? undefined : status, limit: 100 }),
    [status],
  )

  useLiveEvents((event) => {
    if (event.event_type.startsWith('alert')) alerts.reload()
  })
  usePolling(() => alerts.reload(), 15000)

  const rows = useMemo(() => {
    const items = alerts.data?.items ?? []
    return [...items].sort((a, b) => {
      const rank = { critical: 0, warning: 1, info: 2 } as Record<string, number>
      const bySeverity = (rank[a.severity] ?? 3) - (rank[b.severity] ?? 3)
      if (bySeverity !== 0) return bySeverity
      return new Date(b.last_observed_at).getTime() - new Date(a.last_observed_at).getTime()
    })
  }, [alerts.data])

  async function act(alert: Alert, action: 'acknowledge' | 'resolve') {
    setBusy(alert.id)
    setError(null)
    try {
      if (action === 'acknowledge') {
        await api.acknowledgeAlert(alert.id)
      } else {
        await api.resolveAlert(alert.id, 'Resolved from the dashboard')
      }
      alerts.reload()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(null)
    }
  }

  return (
    <div className="flex h-full flex-col gap-3 overflow-hidden">
      {/* Top Header */}
      <div className="flex shrink-0 items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-base font-bold tracking-tight text-zinc-900 dark:text-zinc-100">
              Incident Alert Center
            </h1>
            <span className="inline-flex items-center gap-1 rounded bg-indigo-500/10 px-1.5 py-0.5 text-[10px] font-bold text-indigo-600 dark:text-indigo-400">
              <ShieldAlert className="h-3 w-3" /> De-duplicated
            </span>
          </div>
          <p className="truncate text-[11px] text-zinc-500 dark:text-zinc-400">
            Persistent vision detection alerts with automated cooldown windows
          </p>
        </div>

        {/* Filter Pills */}
        <div className="surface-card flex items-center rounded-lg p-0.5">
          {STATUSES.map((value) => {
            const active = status === value
            return (
              <button
                key={value}
                type="button"
                onClick={() => setStatus(value)}
                className={`rounded-md px-2.5 py-1 text-xs font-semibold capitalize transition-all ${
                  active
                    ? 'bg-indigo-600 text-white shadow-sm'
                    : 'text-zinc-600 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100'
                }`}
              >
                {value}
              </button>
            )
          })}
        </div>
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

      {/* Main Contained Scrollable List of Incidents */}
      <SurfaceCard
        title="Incident Feed"
        subtitle={`${rows.length} alert${rows.length === 1 ? '' : 's'} recorded`}
        icon={<AlertTriangle className="h-4 w-4 text-amber-500" />}
        scrollable
        className="flex-1 min-h-0 overflow-hidden"
      >
        <AsyncContent
          state={alerts}
          empty={`No ${status === 'all' ? '' : status} alerts recorded`}
          isEmpty={() => rows.length === 0}
          rows={4}
        >
          {() => (
            <div className="space-y-2">
              {rows.map((alert) => {
                const isCritical = alert.severity === 'critical'
                const isWarning = alert.severity === 'warning'
                return (
                  <div
                    key={alert.id}
                    className={`flex flex-col justify-between gap-2.5 rounded-lg border p-3 transition-colors sm:flex-row sm:items-center ${
                      isCritical
                        ? 'border-rose-500/30 bg-rose-500/5'
                        : isWarning
                          ? 'border-amber-500/30 bg-amber-500/5'
                          : 'border-zinc-200/70 bg-white/40 dark:border-white/5 dark:bg-zinc-900/40'
                    }`}
                  >
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="text-xs font-bold text-zinc-900 dark:text-zinc-100">
                          {alert.title}
                        </span>
                        <Badge value={alert.severity} />
                        <Badge value={alert.status} />
                      </div>

                      <p className="mt-1 text-xs text-zinc-600 dark:text-zinc-300">
                        {alert.message}
                      </p>

                      <div className="mt-1.5 flex flex-wrap items-center gap-2 text-[10px] text-zinc-400">
                        <span className="font-mono text-zinc-500 dark:text-zinc-400">
                          {alert.alert_type}
                        </span>
                        {alert.sku && <span>· SKU: {alert.sku}</span>}
                        <span>·</span>
                        <span>{alert.occurrence_count}× recurrence</span>
                        <span>·</span>
                        <span>Last: {fmtTime(alert.last_observed_at)}</span>
                        {alert.resolved_at && (
                          <span className="text-emerald-600 dark:text-emerald-400">
                            · Resolved {fmtTime(alert.resolved_at)}
                          </span>
                        )}
                      </div>
                    </div>

                    <div className="flex shrink-0 items-center gap-1.5">
                      {alert.status === 'active' && (
                        <ActionButton
                          variant="secondary"
                          size="xs"
                          disabled={busy === alert.id}
                          onClick={() => act(alert, 'acknowledge')}
                        >
                          {busy === alert.id ? '…' : 'Acknowledge'}
                        </ActionButton>
                      )}
                      {(alert.status === 'active' || alert.status === 'acknowledged') && (
                        <ActionButton
                          variant="primary"
                          size="xs"
                          disabled={busy === alert.id}
                          onClick={() => act(alert, 'resolve')}
                        >
                          {busy === alert.id ? '…' : 'Resolve'}
                        </ActionButton>
                      )}
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </AsyncContent>
      </SurfaceCard>
    </div>
  )
}
