import { useMemo, useRef, useState } from 'react'
import { Bell, Siren, Check, CheckCheck, X } from 'lucide-react'
import { api } from '../services/api'
import { useApi, useLiveEvents } from '../hooks/useAsync'
import type { Alert } from '../types/api'

/**
 * Site-wide alert overlay, pinned above every page.
 *
 * Shows ACTIVE and ACKNOWLEDGED alerts (critical first) and clears itself as
 * soon as the backend resolves them — including the automatic queue alerts:
 * when the queue returns to normal the banner disappears on its own, no
 * manual dismissal needed.
 */

const SEVERITY_ORDER: Record<string, number> = { critical: 0, warning: 1, info: 2 }

export default function AlertBanner() {
  const alerts = useApi<Alert[]>(() => api.activeAlerts(), [])
  const [expanded, setExpanded] = useState(false)
  const [busyId, setBusyId] = useState<string | null>(null)
  // Remember ids we have already acted on so buttons don't double-fire while
  // the next poll lands.
  const actedOn = useRef<Set<string>>(new Set())

  const refresh = () => alerts.reload()

  useLiveEvents((event) => {
    // alert.raised / alert.resolved / alert.acknowledged all arrive here; a
    // resolved queue alert must clear the banner WITHOUT waiting for the poll.
    if (event.event_type.startsWith('alert')) refresh()
  })

  const open = useMemo(() => {
    const list = (alerts.data ?? []).filter((a) => a.status === 'active' || a.status === 'acknowledged')
    return list.sort(
      (a, b) =>
        (SEVERITY_ORDER[a.severity] ?? 3) - (SEVERITY_ORDER[b.severity] ?? 3) ||
        b.created_at.localeCompare(a.created_at),
    )
  }, [alerts.data])

  if (open.length === 0) return null

  const critical = open.filter((a) => a.severity === 'critical')
  const headline = critical[0] ?? open[0]
  const rest = open.filter((a) => a.id !== headline.id)
  const shown = expanded ? rest : rest.slice(0, 2)
  const isQueue = headline.alert_type.startsWith('queue')

  async function act(alert: Alert, action: 'ack' | 'resolve') {
    if (actedOn.current.has(`${alert.id}:${action}`)) return
    actedOn.current.add(`${alert.id}:${action}`)
    setBusyId(alert.id)
    try {
      if (action === 'ack') await api.acknowledgeAlert(alert.id, 'Acknowledged from alert banner')
      else await api.resolveAlert(alert.id, 'Resolved from alert banner')
      refresh()
    } catch {
      // The poll will reconcile state; do not spam the operator with toasts.
    } finally {
      setBusyId(null)
    }
  }

  return (
    <div
      role="alert"
      className={`shrink-0 overflow-hidden rounded-xl border shadow-lg ${
        headline.severity === 'critical'
          ? 'border-rose-500/50 bg-rose-600/10 dark:bg-rose-950/40'
          : 'border-amber-500/50 bg-amber-500/10 dark:bg-amber-950/30'
      }`}
    >
      {/* Headline row */}
      <div className="flex items-center gap-2.5 px-3 py-2">
        {headline.severity === 'critical' ? (
          <Siren className="h-5 w-5 shrink-0 animate-pulse text-rose-500" />
        ) : (
          <Bell className="h-4 w-4 shrink-0 animate-pulse text-amber-500" />
        )}
        <div className="min-w-0 flex-1">
          <p
            className={`truncate text-xs font-extrabold uppercase tracking-wide ${
              headline.severity === 'critical'
                ? 'text-rose-600 dark:text-rose-300'
                : 'text-amber-700 dark:text-amber-300'
            }`}
          >
            {headline.title}
          </p>
          <p className="truncate text-[11px] text-zinc-600 dark:text-zinc-300">{headline.message}</p>
        </div>

        <span
          className={`hidden shrink-0 rounded px-1.5 py-0.5 text-[10px] font-bold sm:inline ${
            headline.status === 'acknowledged'
              ? 'bg-zinc-500/15 text-zinc-600 dark:text-zinc-300'
              : headline.severity === 'critical'
                ? 'bg-rose-600 text-white'
                : 'bg-amber-500 text-zinc-900'
          }`}
        >
          {headline.status.toUpperCase()}
        </span>

        {headline.status === 'active' && (
          <button
            type="button"
            onClick={() => act(headline, 'ack')}
            disabled={busyId === headline.id}
            title="Acknowledge"
            className="shrink-0 rounded-md border border-zinc-300/60 p-1.5 text-zinc-600 transition-colors hover:bg-zinc-500/10 disabled:opacity-50 dark:border-white/10 dark:text-zinc-300"
          >
            <Check className="h-3.5 w-3.5" />
          </button>
        )}
        <button
          type="button"
          onClick={() => act(headline, 'resolve')}
          disabled={busyId === headline.id}
          title="Resolve"
          className="shrink-0 rounded-md border border-emerald-500/40 bg-emerald-500/10 p-1.5 text-emerald-600 transition-colors hover:bg-emerald-500/20 disabled:opacity-50 dark:text-emerald-400"
        >
          <CheckCheck className="h-3.5 w-3.5" />
        </button>
        {rest.length > 0 && (
          <button
            type="button"
            onClick={() => setExpanded((v) => !v)}
            title={expanded ? 'Collapse' : `Show ${rest.length} more`}
            className="shrink-0 rounded-md border border-zinc-300/60 p-1.5 text-zinc-500 transition-colors hover:bg-zinc-500/10 dark:border-white/10 dark:text-zinc-400"
          >
            {expanded ? <X className="h-3.5 w-3.5" /> : (
              <span className="px-0.5 text-[10px] font-bold">+{rest.length}</span>
            )}
          </button>
        )}
      </div>

      {/* Additional alerts */}
      {expanded &&
        shown.map((alert) => (
          <div
            key={alert.id}
            className="flex items-center gap-2.5 border-t border-inherit px-3 py-1.5 opacity-90"
          >
            <Bell className="h-3.5 w-3.5 shrink-0 text-zinc-400" />
            <div className="min-w-0 flex-1">
              <p className="truncate text-[11px] font-bold text-zinc-700 dark:text-zinc-200">
                {alert.title}
              </p>
              <p className="truncate text-[10px] text-zinc-500 dark:text-zinc-400">{alert.message}</p>
            </div>
            <span className="shrink-0 rounded bg-zinc-500/15 px-1.5 py-0.5 text-[9px] font-bold text-zinc-600 dark:text-zinc-300">
              {alert.status.toUpperCase()}
            </span>
            {alert.status === 'active' && (
              <button
                type="button"
                onClick={() => act(alert, 'ack')}
                disabled={busyId === alert.id}
                className="shrink-0 rounded-md border border-zinc-300/60 p-1 text-zinc-500 disabled:opacity-50 dark:border-white/10 dark:text-zinc-400"
              >
                <Check className="h-3 w-3" />
              </button>
            )}
            <button
              type="button"
              onClick={() => act(alert, 'resolve')}
              disabled={busyId === alert.id}
              className="shrink-0 rounded-md border border-emerald-500/40 bg-emerald-500/10 p-1 text-emerald-600 disabled:opacity-50 dark:text-emerald-400"
            >
              <CheckCheck className="h-3 w-3" />
            </button>
          </div>
        ))}

      {isQueue && headline.severity === 'critical' && (
        <p className="border-t border-rose-500/20 bg-rose-600/10 px-3 py-1 text-center text-[10px] font-bold uppercase tracking-widest text-rose-700 dark:text-rose-300">
          Open another counter to split the queue — this banner clears automatically when the queue is back to normal
        </p>
      )}
    </div>
  )
}
