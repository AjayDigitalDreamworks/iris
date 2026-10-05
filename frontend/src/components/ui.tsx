import type { ReactNode } from 'react'

/** Precision-milled glass surface card with specular top highlight and contained scrollable body */
export interface SurfaceCardProps {
  title?: string
  subtitle?: string
  icon?: ReactNode
  badge?: ReactNode
  actions?: ReactNode
  footer?: ReactNode
  children: ReactNode
  className?: string
  bodyClassName?: string
  headerClassName?: string
  scrollable?: boolean
}

export function SurfaceCard({
  title,
  subtitle,
  icon,
  badge,
  actions,
  footer,
  children,
  className = '',
  bodyClassName = '',
  headerClassName = '',
  scrollable = false,
}: SurfaceCardProps) {
  return (
    <div
      className={`surface-card surface-specular flex flex-col overflow-hidden rounded-xl ${className}`}
    >
      {(title || icon || badge || actions) && (
        <div
          className={`flex shrink-0 items-center justify-between border-b border-zinc-200/80 px-4 py-2.5 dark:border-white/10 ${headerClassName}`}
        >
          <div className="flex items-center gap-2.5 min-w-0">
            {icon && (
              <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-zinc-100 text-zinc-600 dark:bg-white/5 dark:text-zinc-300">
                {icon}
              </div>
            )}
            <div className="min-w-0">
              {title && (
                <h3 className="truncate text-xs font-semibold uppercase tracking-wider text-zinc-900 dark:text-zinc-100">
                  {title}
                </h3>
              )}
              {subtitle && (
                <p className="truncate text-[11px] text-zinc-500 dark:text-zinc-400">{subtitle}</p>
              )}
            </div>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            {badge}
            {actions}
          </div>
        </div>
      )}

      <div
        className={`flex-1 p-3.5 ${
          scrollable ? 'overflow-y-auto custom-scrollbar' : ''
        } ${bodyClassName}`}
      >
        {children}
      </div>

      {footer && (
        <div className="shrink-0 border-t border-zinc-200/80 bg-zinc-50/50 px-4 py-2 dark:border-white/5 dark:bg-zinc-950/30">
          {footer}
        </div>
      )}
    </div>
  )
}

/** Precision Metric Tile with high information density */
export function MetricTile({
  label,
  value,
  hint,
  tone = 'default',
  icon,
  badge,
}: {
  label: string
  value: ReactNode
  hint?: string
  tone?: 'default' | 'good' | 'warn' | 'bad'
  icon?: ReactNode
  badge?: ReactNode
}) {
  const tones = {
    default: 'text-zinc-900 dark:text-zinc-100',
    good: 'text-emerald-600 dark:text-emerald-400',
    warn: 'text-amber-600 dark:text-amber-400',
    bad: 'text-rose-600 dark:text-rose-400',
  }

  return (
    <div className="surface-card surface-specular surface-interactive flex flex-col justify-between rounded-xl p-3">
      <div className="flex items-center justify-between gap-2">
        <span className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
          {label}
        </span>
        {icon && (
          <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md bg-zinc-100 text-zinc-600 dark:bg-white/5 dark:text-zinc-300">
            {icon}
          </div>
        )}
      </div>

      <div className="mt-2 flex items-baseline justify-between gap-2">
        <span className={`text-2xl font-bold tracking-tight tabular-nums ${tones[tone]}`}>
          {value}
        </span>
        {badge}
      </div>

      {hint && (
        <p className="mt-1 truncate text-[11px] text-zinc-500 dark:text-zinc-400">
          {hint}
        </p>
      )}
    </div>
  )
}

/** Legacy & Bento compatible Card */
export function Card({
  title,
  subtitle,
  actions,
  children,
  className = '',
  icon,
}: {
  title?: string
  subtitle?: string
  actions?: ReactNode
  children: ReactNode
  className?: string
  icon?: ReactNode
  glowing?: boolean
  interactive?: boolean
}) {
  return (
    <SurfaceCard
      title={title}
      subtitle={subtitle}
      icon={icon}
      actions={actions}
      className={className}
    >
      {children}
    </SurfaceCard>
  )
}

/** BentoTile matching existing references */
export function BentoTile({
  colSpan = 1,
  rowSpan = 1,
  title,
  subtitle,
  icon,
  badge,
  actions,
  children,
  className = '',
}: {
  colSpan?: 1 | 2 | 3 | 4
  rowSpan?: 1 | 2
  title?: string
  subtitle?: string
  icon?: ReactNode
  badge?: ReactNode
  actions?: ReactNode
  children: ReactNode
  className?: string
}) {
  const colSpanClasses = {
    1: 'col-span-1',
    2: 'col-span-1 md:col-span-2',
    3: 'col-span-1 md:col-span-2 xl:col-span-3',
    4: 'col-span-1 md:col-span-2 xl:col-span-4',
  }

  const rowSpanClasses = {
    1: 'row-span-1',
    2: 'row-span-1 md:row-span-2',
  }

  return (
    <SurfaceCard
      title={title}
      subtitle={subtitle}
      icon={icon}
      badge={badge}
      actions={actions}
      className={`${colSpanClasses[colSpan]} ${rowSpanClasses[rowSpan]} ${className}`}
    >
      {children}
    </SurfaceCard>
  )
}

/** SpotlightCard wrapper */
export function SpotlightCard({
  children,
  className = '',
  onClick,
}: {
  children: ReactNode
  className?: string
  onClick?: () => void
}) {
  return (
    <div
      onClick={onClick}
      className={`surface-card surface-specular surface-interactive ${className}`}
    >
      {children}
    </div>
  )
}

/** Stat primitive matching existing vitest tests */
export function Stat({
  label,
  value,
  hint,
  tone = 'default',
  icon,
}: {
  label: string
  value: ReactNode
  hint?: string
  tone?: 'default' | 'good' | 'warn' | 'bad'
  icon?: ReactNode
}) {
  return <MetricTile label={label} value={value} hint={hint} tone={tone} icon={icon} />
}

const BADGE_CONFIG: Record<string, { bg: string; dot: string }> = {
  online: {
    bg: 'bg-emerald-500/10 border-emerald-500/30 text-emerald-700 dark:bg-emerald-500/15 dark:text-emerald-300',
    dot: 'bg-emerald-500',
  },
  healthy: {
    bg: 'bg-emerald-500/10 border-emerald-500/30 text-emerald-700 dark:bg-emerald-500/15 dark:text-emerald-300',
    dot: 'bg-emerald-500',
  },
  ok: {
    bg: 'bg-emerald-500/10 border-emerald-500/30 text-emerald-700 dark:bg-emerald-500/15 dark:text-emerald-300',
    dot: 'bg-emerald-500',
  },
  normal: {
    bg: 'bg-emerald-500/10 border-emerald-500/30 text-emerald-700 dark:bg-emerald-500/15 dark:text-emerald-300',
    dot: 'bg-emerald-500',
  },
  busy: {
    bg: 'bg-amber-500/10 border-amber-500/30 text-amber-700 dark:bg-amber-500/15 dark:text-amber-300',
    dot: 'bg-amber-500',
  },
  low_stock: {
    bg: 'bg-amber-500/10 border-amber-500/30 text-amber-700 dark:bg-amber-500/15 dark:text-amber-300',
    dot: 'bg-amber-500',
  },
  warning: {
    bg: 'bg-amber-500/10 border-amber-500/30 text-amber-700 dark:bg-amber-500/15 dark:text-amber-300',
    dot: 'bg-amber-500',
  },
  degraded: {
    bg: 'bg-amber-500/10 border-amber-500/30 text-amber-700 dark:bg-amber-500/15 dark:text-amber-300',
    dot: 'bg-amber-500',
  },
  connecting: {
    bg: 'bg-sky-500/10 border-sky-500/30 text-sky-700 dark:bg-sky-500/15 dark:text-sky-300',
    dot: 'bg-sky-500 animate-pulse',
  },
  candidate: {
    bg: 'bg-sky-500/10 border-sky-500/30 text-sky-700 dark:bg-sky-500/15 dark:text-sky-300',
    dot: 'bg-sky-500',
  },
  active: {
    bg: 'bg-rose-500/10 border-rose-500/30 text-rose-700 dark:bg-rose-500/15 dark:text-rose-300',
    dot: 'bg-rose-500 animate-pulse',
  },
  critical: {
    bg: 'bg-rose-500/10 border-rose-500/30 text-rose-700 dark:bg-rose-500/15 dark:text-rose-300',
    dot: 'bg-rose-500 animate-ping',
  },
  out_of_stock: {
    bg: 'bg-rose-500/10 border-rose-500/30 text-rose-700 dark:bg-rose-500/15 dark:text-rose-300',
    dot: 'bg-rose-500',
  },
  error: {
    bg: 'bg-rose-500/10 border-rose-500/30 text-rose-700 dark:bg-rose-500/15 dark:text-rose-300',
    dot: 'bg-rose-500',
  },
  offline: {
    bg: 'bg-rose-500/10 border-rose-500/30 text-rose-700 dark:bg-rose-500/15 dark:text-rose-300',
    dot: 'bg-rose-400',
  },
  acknowledged: {
    bg: 'bg-indigo-500/10 border-indigo-500/30 text-indigo-700 dark:bg-indigo-500/15 dark:text-indigo-300',
    dot: 'bg-indigo-500',
  },
  resolved: {
    bg: 'bg-zinc-500/10 border-zinc-500/30 text-zinc-700 dark:bg-zinc-500/15 dark:text-zinc-300',
    dot: 'bg-zinc-400',
  },
  disabled: {
    bg: 'bg-zinc-500/10 border-zinc-500/30 text-zinc-600 dark:bg-zinc-500/15 dark:text-zinc-400',
    dot: 'bg-zinc-400',
  },
  stopped: {
    bg: 'bg-zinc-500/10 border-zinc-500/30 text-zinc-600 dark:bg-zinc-500/15 dark:text-zinc-400',
    dot: 'bg-zinc-400',
  },
  suppressed: {
    bg: 'bg-zinc-500/10 border-zinc-500/30 text-zinc-600 dark:bg-zinc-500/15 dark:text-zinc-400',
    dot: 'bg-zinc-400',
  },
  unknown: {
    bg: 'bg-zinc-500/10 border-zinc-500/30 text-zinc-600 dark:bg-zinc-500/15 dark:text-zinc-400',
    dot: 'bg-zinc-400',
  },
}

export function Badge({
  value,
  label,
  showDot = true,
}: {
  value: string
  label?: string
  showDot?: boolean
}) {
  const config = BADGE_CONFIG[value] ?? BADGE_CONFIG.unknown

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-md border px-2 py-0.5 text-[11px] font-semibold transition-colors ${config.bg}`}
    >
      {showDot && <span className={`h-1.5 w-1.5 rounded-full ${config.dot}`} />}
      <span className="capitalize">{label ?? value.replace(/_/g, ' ')}</span>
    </span>
  )
}

export function Skeleton({ rows = 3 }: { rows?: number }) {
  return (
    <div className="space-y-2.5" aria-busy="true" aria-label="Loading">
      {Array.from({ length: rows }).map((_, index) => (
        <div
          key={index}
          className="animate-shimmer h-9 rounded-lg border border-zinc-200/50 bg-zinc-200/50 dark:border-white/5 dark:bg-zinc-800/40"
          style={{ animationDelay: `${index * 120}ms` }}
        />
      ))}
    </div>
  )
}

export function EmptyState({
  title,
  hint,
  icon,
}: {
  title: string
  hint?: string
  icon?: ReactNode
}) {
  return (
    <div className="surface-card flex flex-col items-center justify-center rounded-xl border-dashed border-zinc-300/80 px-6 py-10 text-center dark:border-zinc-800">
      {icon ? (
        <div className="mb-2.5 flex h-10 w-10 items-center justify-center rounded-xl bg-zinc-100 text-zinc-500 dark:bg-zinc-800/80 dark:text-zinc-400">
          {icon}
        </div>
      ) : (
        <div className="mb-2.5 flex h-9 w-9 items-center justify-center rounded-full bg-zinc-500/10 text-zinc-400">
          <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4" />
          </svg>
        </div>
      )}
      <p className="text-xs font-semibold text-zinc-900 dark:text-zinc-100">{title}</p>
      {hint && (
        <p className="mt-1 max-w-xs text-[11px] leading-relaxed text-zinc-500 dark:text-zinc-400">
          {hint}
        </p>
      )}
    </div>
  )
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div
      role="alert"
      className="surface-card rounded-xl border border-rose-500/30 bg-rose-50/60 p-3.5 text-xs text-rose-900 dark:bg-rose-950/20 dark:text-rose-200"
    >
      <div className="flex items-start gap-2.5">
        <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-lg bg-rose-500/20 text-rose-600 dark:text-rose-400">
          <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
        </div>
        <div className="min-w-0 flex-1">
          <p className="font-semibold tracking-tight">Could not load this data.</p>
          <p className="mt-0.5 break-words text-[11px] text-rose-700/80 dark:text-rose-300/80">
            {message}
          </p>
          {onRetry && (
            <button
              type="button"
              onClick={onRetry}
              className="mt-2 inline-flex items-center gap-1 rounded-md border border-rose-500/30 bg-rose-500/10 px-2.5 py-1 text-[11px] font-semibold text-rose-700 transition hover:bg-rose-500/20 active:scale-95 dark:text-rose-300"
            >
              Retry
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

/** Renders a load/error/empty/content state so no panel is blank */
export function AsyncContent<T>({
  state,
  empty,
  isEmpty,
  rows,
  children,
}: {
  state: { data: T | null; loading: boolean; error: string | null; reload: () => void }
  empty: string
  isEmpty?: (data: T) => boolean
  rows?: number
  children: (data: T) => ReactNode
}) {
  if (state.loading && state.data === null) return <Skeleton rows={rows} />
  if (state.error) return <ErrorState message={state.error} onRetry={state.reload} />
  if (state.data === null) return <EmptyState title={empty} />
  if (isEmpty?.(state.data)) return <EmptyState title={empty} />
  return <>{children(state.data)}</>
}

/** Precision Action Button */
export function ActionButton({
  children,
  onClick,
  disabled = false,
  variant = 'primary',
  size = 'sm',
  type = 'button',
  className = '',
  icon,
}: {
  children: ReactNode
  onClick?: () => void
  disabled?: boolean
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger'
  size?: 'xs' | 'sm' | 'md'
  type?: 'button' | 'submit' | 'reset'
  className?: string
  icon?: ReactNode
}) {
  const base =
    'inline-flex items-center justify-center font-semibold transition-all duration-150 active:scale-[0.98] disabled:opacity-50 disabled:pointer-events-none rounded-lg'

  const sizes = {
    xs: 'px-2 py-1 text-[11px] gap-1',
    sm: 'px-3 py-1.5 text-xs gap-1.5',
    md: 'px-4 py-2 text-xs sm:text-sm gap-2',
  }

  const variants = {
    primary:
      'bg-indigo-600 text-white hover:bg-indigo-500 shadow-sm border border-indigo-400/30 dark:bg-indigo-600 dark:hover:bg-indigo-500',
    secondary:
      'surface-card text-zinc-700 hover:text-zinc-900 hover:bg-zinc-100/80 dark:text-zinc-200 dark:hover:text-white dark:hover:bg-zinc-800/80 border border-zinc-200/80 dark:border-white/10',
    ghost:
      'text-zinc-600 hover:text-zinc-900 hover:bg-zinc-100/60 dark:text-zinc-400 dark:hover:text-zinc-100 dark:hover:bg-zinc-800/60',
    danger:
      'bg-rose-500/10 text-rose-600 border border-rose-500/30 hover:bg-rose-500/20 dark:text-rose-400 dark:bg-rose-500/15 dark:hover:bg-rose-500/25',
  }

  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`${base} ${sizes[size]} ${variants[variant]} ${className}`}
    >
      {icon && <span className="shrink-0">{icon}</span>}
      {children}
    </button>
  )
}

export function Button(props: Parameters<typeof ActionButton>[0]) {
  return <ActionButton {...props} />
}

export function StatusIndicator({
  state,
  label,
}: {
  state: 'online' | 'busy' | 'offline' | 'connecting'
  label?: string
}) {
  const styles = {
    online: 'bg-emerald-500',
    busy: 'bg-amber-500',
    offline: 'bg-rose-500',
    connecting: 'bg-sky-500 animate-pulse',
  }

  return (
    <div className="inline-flex items-center gap-1.5">
      <span className="relative flex h-2 w-2">
        {state === 'online' && (
          <span className="absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75 animate-ping" />
        )}
        <span className={`relative inline-flex h-2 w-2 rounded-full ${styles[state]}`} />
      </span>
      {label && <span className="text-[11px] font-semibold capitalize text-zinc-600 dark:text-zinc-300">{label}</span>}
    </div>
  )
}

export const fmtPct = (value: number | null | undefined, digits = 1): string =>
  value === null || value === undefined || value < 0 ? '—' : `${value.toFixed(digits)}%`

export const fmtNumber = (value: number | null | undefined, digits = 2): string =>
  value === null || value === undefined ? '—' : value.toFixed(digits)

export const fmtTime = (iso: string | null | undefined): string => {
  if (!iso) return '—'
  const date = new Date(iso)
  return Number.isNaN(date.getTime()) ? '—' : date.toLocaleString()
}

export const fmtSeconds = (value: number | null | undefined): string => {
  if (value === null || value === undefined) return '—'
  if (value < 60) return `${Math.round(value)}s`
  return `${Math.floor(value / 60)}m ${Math.round(value % 60)}s`
}

export function RadialGauge({
  value,
  max = 100,
  label,
  sublabel,
  size = 96,
  strokeWidth = 8,
  color = '#2563eb',
  trackColor = '#e2e8f0',
  suffix = '',
  displayValue,
}: {
  value: number
  max?: number
  label?: string
  sublabel?: string
  size?: number
  strokeWidth?: number
  color?: string
  trackColor?: string
  suffix?: string
  displayValue?: string | number
}) {
  const radius = (size - strokeWidth) / 2
  const circumference = 2 * Math.PI * radius
  const normalizedValue = Math.min(Math.max(value, 0), max)
  const percent = normalizedValue / max
  const dashOffset = circumference * (1 - percent)

  return (
    <div className="flex flex-col items-center justify-center">
      <div className="relative flex items-center justify-center" style={{ width: size, height: size }}>
        <svg className="w-full h-full -rotate-90" viewBox={`0 0 ${size} ${size}`}>
          {/* Precision Perimeter Ticks */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius + 3}
            fill="none"
            stroke="currentColor"
            strokeWidth={1}
            strokeDasharray="1.5 5.5"
            className="text-slate-300 dark:text-slate-700 opacity-60"
          />
          {/* Background Track */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke={trackColor}
            strokeWidth={strokeWidth}
            className="dark:stroke-slate-800"
          />
          {/* Active Arc */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke={color}
            strokeWidth={strokeWidth}
            strokeDasharray={circumference}
            strokeDashoffset={dashOffset}
            strokeLinecap="round"
            className="transition-all duration-700 ease-out"
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center text-center select-none">
          <span className="text-base font-extrabold tracking-tight text-slate-800 dark:text-slate-100 font-sans leading-none tabular-nums">
            {displayValue !== undefined ? displayValue : `${Math.round(normalizedValue)}${suffix}`}
          </span>
          {sublabel && (
            <span className="text-[9px] font-semibold text-slate-400 dark:text-slate-500 mt-1 uppercase tracking-wider">
              {sublabel}
            </span>
          )}
        </div>
      </div>
      {label && (
        <span className="text-[11px] font-medium text-slate-600 dark:text-slate-300 mt-1.5 text-center">
          {label}
        </span>
      )}
    </div>
  )
}

