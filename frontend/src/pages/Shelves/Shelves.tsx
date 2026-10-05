import {
  Layers,
  Grid,
  Camera,
  PackageX,
} from 'lucide-react'
import { api } from '../../services/api'
import { useApi, usePolling } from '../../hooks/useAsync'
import {
  AsyncContent,
  Badge,
  SurfaceCard,
  EmptyState,
  fmtNumber,
  fmtPct,
} from '../../components/ui'

// Mirrors the live `/analytics/shelves` payload. Field names are taken from
// the backend response, never invented — drift here silently shows blanks.
interface ShelfRow {
  shelf_id?: string
  shelf_code?: string
  shelf_name?: string
  camera_id?: string | null
  capacity_units?: number | null
  observed_units?: number | null
  estimated_quantity?: number | null
  stock_percentage?: number | null
  occupancy_pct?: number | null
  confidence?: number | null
  detected_skus?: number | null
  empty_positions?: number | null
  planogram_compliance_pct?: number | null
  last_observed_at?: string | null
  state?: string
  /** True only when the shelf was actually observed to be empty. */
  is_out_of_stock?: boolean
  out_of_stock_reason?: string | null
}

interface ShelfAnalytics {
  rows: ShelfRow[]
  note?: string
  range_start?: string
  range_end?: string
}

export default function Shelves() {
  const shelves = useApi(() => api.shelves({ limit: 100 }), [])
  const analytics = useApi<ShelfAnalytics>(
    () => api.shelfAnalytics() as unknown as Promise<ShelfAnalytics>,
    [],
  )
  usePolling(() => {
    shelves.reload()
    analytics.reload()
  }, 20000)

  const rows = analytics.data?.rows ?? []
  const totalUnits = rows.reduce((sum, row) => sum + (row.observed_units ?? 0), 0)

  return (
    <div className="flex h-full flex-col gap-3 overflow-hidden">
      {/* Top Header */}
      <div className="flex shrink-0 items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-base font-bold tracking-tight text-zinc-900 dark:text-zinc-100">
              Shelf Analytics
            </h1>
            <span className="inline-flex items-center gap-1 rounded bg-indigo-500/10 px-1.5 py-0.5 text-[10px] font-bold text-indigo-600 dark:text-indigo-400">
              <Layers className="h-3 w-3" /> Planogram
            </span>
          </div>
          <p className="truncate text-[11px] text-zinc-500 dark:text-zinc-400">
            Visual area ratio and empty slot position tracking across retail fixtures
          </p>
        </div>

        <div className="flex items-center gap-2">
          <span className="rounded-md border border-zinc-200/80 bg-white/60 px-2 py-1 font-mono text-[11px] font-semibold text-zinc-700 dark:border-white/10 dark:bg-zinc-900/60 dark:text-zinc-300">
            Detected: {fmtNumber(totalUnits, 0)} units
          </span>
        </div>
      </div>

      {/* 2-Zone Viewport Grid (Fills Remaining Viewport) */}
      <div className="grid min-h-0 flex-1 grid-cols-1 gap-3 overflow-hidden lg:grid-cols-12">
        {/* Left Primary Area (7 cols): Configured Shelves Table */}
        <SurfaceCard
          title="Planogram Shelves"
          subtitle={`${shelves.data?.items?.length ?? 0} physical fixtures`}
          icon={<Grid className="h-4 w-4 text-indigo-500" />}
          scrollable
          className="flex-1 min-h-0 overflow-hidden lg:col-span-7"
          bodyClassName="p-0 overflow-hidden flex flex-col"
        >
          <AsyncContent
            state={shelves}
            empty="No shelves configured"
            isEmpty={(p) => p.items.length === 0}
            rows={4}
          >
            {(page) => (
              <div className="custom-scrollbar flex-1 overflow-auto">
                <table className="w-full text-left text-xs">
                  <thead className="sticky top-0 z-10 border-b border-zinc-200 bg-zinc-50/95 backdrop-blur dark:border-white/10 dark:bg-zinc-900/95">
                    <tr className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400">
                      <th className="py-2.5 pl-4 pr-3">Code</th>
                      <th className="py-2.5 pr-3">Name</th>
                      <th className="py-2.5 pr-3">Aisle / Level</th>
                      <th className="py-2.5 pr-3 text-right">Capacity</th>
                      <th className="py-2.5 pr-3">Camera</th>
                      <th className="py-2.5 pr-4">Active</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-zinc-200/60 dark:divide-white/5">
                    {page.items.map((shelf) => (
                      <tr
                        key={shelf.id}
                        className="transition-colors hover:bg-zinc-100/60 dark:hover:bg-zinc-800/40"
                      >
                        <td className="py-2 pl-4 pr-3 font-mono text-[11px] font-semibold">
                          {shelf.code}
                        </td>
                        <td className="py-2 pr-3 font-medium text-zinc-800 dark:text-zinc-200">
                          {shelf.name}
                        </td>
                        <td className="py-2 pr-3 text-zinc-500">
                          {shelf.aisle ? `Aisle ${shelf.aisle}` : 'Aisle —'} · Level {shelf.level ?? '—'}
                        </td>
                        <td className="py-2 pr-3 text-right font-mono tabular-nums">
                          {shelf.capacity_units}
                        </td>
                        <td className="py-2 pr-3 font-mono text-[11px] text-zinc-400">
                          {shelf.camera_id ? (
                            <span className="flex items-center gap-1">
                              <Camera className="h-3 w-3 text-indigo-500" />
                              {shelf.camera_id.slice(0, 8)}
                            </span>
                          ) : (
                            '—'
                          )}
                        </td>
                        <td className="py-2 pr-4">
                          <Badge
                            value={shelf.is_active ? 'online' : 'disabled'}
                            label={shelf.is_active ? 'active' : 'inactive'}
                          />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </AsyncContent>
        </SurfaceCard>

        {/* Right Meta Area (5 cols): Region Observations List */}
        <SurfaceCard
          title="Region Observations"
          subtitle={analytics.data?.note ?? 'Occupied area ratio and slot empty positions'}
          icon={<Layers className="h-4 w-4 text-emerald-500" />}
          scrollable
          className="flex-1 min-h-0 overflow-hidden lg:col-span-5"
        >
          <AsyncContent
            state={analytics}
            empty="No shelf observations yet"
            isEmpty={() => rows.length === 0}
            rows={3}
          >
            {() => (
              <div className="space-y-2.5">
                {rows.map((row, idx) => {
                  const occ = row.occupancy_pct ?? 0
                  return (
                    <div
                      key={row.shelf_id ?? idx}
                      className={`rounded-lg border p-3 transition-colors ${
                        row.is_out_of_stock
                          ? 'border-rose-400/70 bg-rose-50/60 ring-2 ring-rose-500/40 dark:border-rose-500/50 dark:bg-rose-950/25'
                          : 'border-zinc-200/70 bg-white/40 dark:border-white/5 dark:bg-zinc-900/40'
                      }`}
                    >
                      {/* The empty-rack badge. Driven by the backend's own
                          decision (same rule the alert used), so the badge on
                          screen can never disagree with the alert in the log.
                          An unobserved shelf is never badged. */}
                      {row.is_out_of_stock && (
                        <div className="mb-2.5 flex items-center gap-2 rounded-md bg-rose-600 px-2.5 py-1.5 text-white shadow-sm">
                          <PackageX className="h-4 w-4 shrink-0" />
                          <div className="min-w-0">
                            <p className="text-[10px] font-bold uppercase tracking-widest">
                              Out of stock
                            </p>
                            <p className="truncate text-[10px] text-rose-100">
                              {row.out_of_stock_reason ?? 'No units visible on this shelf.'}
                            </p>
                          </div>
                        </div>
                      )}
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-xs font-bold text-zinc-800 dark:text-zinc-200">
                          {row.shelf_code ?? row.shelf_id ?? 'Shelf Region'}
                        </span>
                        <Badge
                          value={row.state === 'unknown' ? 'unknown' : 'ok'}
                          label={row.state === 'unknown' ? 'unobserved' : 'observed'}
                        />
                      </div>

                      {/* Mini Occupancy Bar */}
                      <div className="mt-2 space-y-1">
                        <div className="flex justify-between text-[11px]">
                          <span className="text-zinc-500">Visual Occupancy</span>
                          <span className="font-mono font-semibold text-zinc-800 dark:text-zinc-200">
                            {fmtPct(row.occupancy_pct)}
                          </span>
                        </div>
                        <div className="h-1.5 w-full overflow-hidden rounded-full bg-zinc-200/80 dark:bg-zinc-800">
                          <div
                            className="h-full bg-indigo-600 transition-all duration-300"
                            style={{ width: `${Math.min(100, Math.max(0, occ))}%` }}
                          />
                        </div>
                      </div>

                      <div className="mt-2.5 grid grid-cols-3 gap-1.5 text-center text-[10px]">
                        <div className="rounded border border-zinc-200/50 bg-zinc-50/50 p-1.5 dark:border-white/5 dark:bg-zinc-800/30">
                          <span className="text-zinc-400">Stock %</span>
                          <p className="mt-0.5 font-mono font-semibold text-zinc-700 dark:text-zinc-300">
                            {fmtPct(row.stock_percentage)}
                          </p>
                        </div>
                        <div className="rounded border border-zinc-200/50 bg-zinc-50/50 p-1.5 dark:border-white/5 dark:bg-zinc-800/30">
                          <span className="text-zinc-400">Detected</span>
                          <p className="mt-0.5 font-mono font-semibold text-emerald-600 dark:text-emerald-400">
                            {row.observed_units ?? '—'}
                          </p>
                        </div>
                        <div className="rounded border border-zinc-200/50 bg-zinc-50/50 p-1.5 dark:border-white/5 dark:bg-zinc-800/30">
                          <span className="text-zinc-400">SKUs</span>
                          <p className="mt-0.5 font-mono font-semibold text-amber-600 dark:text-amber-400">
                            {row.detected_skus ?? '—'}
                          </p>
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </AsyncContent>
        </SurfaceCard>
      </div>

      {rows.length === 0 && shelves.error === null && (
        <EmptyState
          title="No shelf analytics available"
          hint="Link a shelf boundary zone to a camera observing a real fixture to begin stream processing."
          icon={<Layers className="h-6 w-6 text-zinc-400" />}
        />
      )}
    </div>
  )
}

