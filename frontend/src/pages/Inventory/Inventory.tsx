import { useMemo, useState } from 'react'
import {
  Package,
  Search,
  Clock,
  Layers,
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
  fmtTime,
} from '../../components/ui'
import type { InventoryEstimate } from '../../types/api'

type Filter = 'all' | 'low_stock' | 'out_of_stock' | 'unknown'

const FILTER_OPTIONS: { id: Filter; label: string }[] = [
  { id: 'all', label: 'All Items' },
  { id: 'low_stock', label: 'Low Stock' },
  { id: 'out_of_stock', label: 'Out of Stock' },
  { id: 'unknown', label: 'Unknown State' },
]

export default function Inventory() {
  const [filter, setFilter] = useState<Filter>('all')
  const [search, setSearch] = useState('')
  const [uploading, setUploading] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const estimates = useApi<InventoryEstimate[]>(() => api.inventory(), [])
  usePolling(() => estimates.reload(), 15000)

  // A product photo is what makes a SKU recognisable, so this is the one action
  // that turns "unknown" detections into a named product.
  async function uploadReference(productId: string, sku: string, file: File) {
    setUploading(productId)
    setNotice(null)
    try {
      const result = await api.uploadReferenceImage(productId, file)
      setNotice(`${sku}: ${result.message}`)
      estimates.reload()
    } catch (err) {
      setNotice(`${sku}: ${err instanceof Error ? err.message : String(err)}`)
    } finally {
      setUploading(null)
    }
  }

  const rows = useMemo(() => {
    const all = estimates.data ?? []
    return all
      .filter((row) => (filter === 'all' ? true : row.state === filter))
      .filter((row) => row.sku.toLowerCase().includes(search.toLowerCase()))
      .sort((a, b) => (a.state === b.state ? 0 : a.state.localeCompare(b.state)))
  }, [estimates.data, filter, search])

  const counts = useMemo(() => {
    const all = estimates.data ?? []
    return {
      all: all.length,
      low_stock: all.filter((r) => r.state === 'low_stock').length,
      out_of_stock: all.filter((r) => r.state === 'out_of_stock').length,
      unknown: all.filter((r) => r.state === 'unknown').length,
    }
  }, [estimates.data])

  return (
    <div className="flex h-full flex-col gap-3 overflow-hidden">
      {/* Top Header & Filters Bar (Fixed Height) */}
      <div className="flex shrink-0 flex-wrap items-center justify-between gap-2.5">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-base font-bold tracking-tight text-zinc-900 dark:text-zinc-100">
              Inventory Intelligence
            </h1>
            <span className="inline-flex items-center gap-1 rounded bg-indigo-500/10 px-1.5 py-0.5 text-[10px] font-bold text-indigo-600 dark:text-indigo-400">
              <Package className="h-3 w-3" /> Shelf Vision
            </span>
          </div>
          <p className="truncate text-[11px] text-zinc-500 dark:text-zinc-400">
            Visible shelf estimates measured by camera observations vs registered inventory
          </p>
        </div>

        {/* Filter Pills + Search Input in Compact Bar */}
        <div className="flex items-center gap-2">
          <div className="surface-card flex items-center rounded-lg p-0.5">
            {FILTER_OPTIONS.map((opt) => {
              const active = filter === opt.id
              const count = counts[opt.id]
              return (
                <button
                  key={opt.id}
                  type="button"
                  onClick={() => setFilter(opt.id)}
                  className={`flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-semibold transition-all ${
                    active
                      ? 'bg-indigo-600 text-white shadow-sm'
                      : 'text-zinc-600 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100'
                  }`}
                >
                  <span>{opt.label}</span>
                  <span
                    className={`rounded px-1 py-0.1 text-[10px] font-bold ${
                      active
                        ? 'bg-white/20 text-white'
                        : 'bg-zinc-200/80 text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400'
                    }`}
                  >
                    {count}
                  </span>
                </button>
              )
            })}
          </div>

          <div className="relative w-48 sm:w-56">
            <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-zinc-400" />
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search SKU…"
              aria-label="Search by SKU"
              className="precision-input w-full rounded-lg py-1 pl-8 pr-2.5 text-xs placeholder:text-zinc-400"
            />
          </div>
        </div>
      </div>

      {/* Main Viewport Data Grid (Fills Remaining Viewport) */}
      <SurfaceCard
        title="Active Shelf Estimates"
        subtitle={`${rows.length} product SKU records active`}
        icon={<Layers className="h-4 w-4 text-indigo-500" />}
        scrollable
        className="flex-1 min-h-0 overflow-hidden"
        bodyClassName="p-0 overflow-hidden flex flex-col"
        footer={
          rows.some((row) => row.observation_age_seconds !== null && row.is_stale) ? (
            <div className="flex items-center gap-1.5 text-[11px] text-amber-600 dark:text-amber-400">
              <Clock className="h-3.5 w-3.5 shrink-0" />
              <span>
                Observations flagged as stale exceed the configured freshness threshold.
              </span>
            </div>
          ) : (
            <div className="text-[11px] text-zinc-400">
              Direct sensor estimates reflecting visible facings.
            </div>
          )
        }
      >
        <AsyncContent
          state={estimates}
          empty="No stock estimates recorded yet"
          isEmpty={() => rows.length === 0}
          rows={6}
        >
          {() => (
            <div className="custom-scrollbar flex-1 overflow-auto">
              {notice && (
                <p className="border-b border-zinc-200 bg-indigo-50/90 px-4 py-1.5 text-[11px] text-indigo-700 dark:border-white/10 dark:bg-indigo-950/40 dark:text-indigo-300">
                  {notice}
                </p>
              )}
              <table className="w-full text-left text-xs">
                <thead className="sticky top-0 z-10 border-b border-zinc-200 bg-zinc-50/95 backdrop-blur dark:border-white/10 dark:bg-zinc-900/95">
                  <tr className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400">
                    <th className="py-2.5 pl-4 pr-3">SKU</th>
                    <th className="py-2.5 pr-3">Product Name</th>
                    <th className="py-2.5 pr-3 text-right">Observed</th>
                    <th className="py-2.5 pr-3 text-right">Estimated</th>
                    <th className="py-2.5 pr-3 text-right">% Capacity</th>
                    <th className="py-2.5 pr-3 text-right">Confidence</th>
                    <th className="py-2.5 pr-3">State</th>
                    <th className="py-2.5 pr-3">Registered</th>
                    <th className="py-2.5 pr-4">Observed At</th>
                    <th className="py-2.5 pr-4">Reference</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-200/60 dark:divide-white/5">
                  {rows.map((row) => (
                    <tr
                      key={`${row.sku}-${row.shelf_id ?? 'none'}`}
                      className="transition-colors hover:bg-zinc-100/60 dark:hover:bg-zinc-800/40"
                    >
                      <td className="py-2 pl-4 pr-3">
                        <span className="rounded bg-zinc-100 px-1.5 py-0.5 font-mono text-[11px] font-semibold text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300">
                          {row.sku}
                        </span>
                      </td>
                      <td className="py-2 pr-3 font-medium text-zinc-800 dark:text-zinc-200">
                        {row.product_name ?? '—'}
                      </td>
                      <td className="py-2 pr-3 text-right font-mono font-semibold tabular-nums text-zinc-900 dark:text-zinc-100">
                        {row.observed_quantity}
                      </td>
                      <td className="py-2 pr-3 text-right font-mono tabular-nums text-zinc-700 dark:text-zinc-300">
                        {fmtNumber(row.estimated_quantity)}
                      </td>
                      <td className="py-2 pr-3 text-right font-mono tabular-nums">
                        <span
                          className={`font-semibold ${
                            (row.stock_percentage ?? 0) < 20
                              ? 'text-rose-600 dark:text-rose-400'
                              : (row.stock_percentage ?? 0) < 50
                                ? 'text-amber-600 dark:text-amber-400'
                                : 'text-emerald-600 dark:text-emerald-400'
                          }`}
                        >
                          {fmtPct(row.stock_percentage)}
                        </span>
                      </td>
                      <td className="py-2 pr-3 text-right font-mono tabular-nums text-zinc-500">
                        {fmtPct(row.confidence * 100, 0)}
                      </td>
                      <td className="py-2 pr-3">
                        <Badge value={row.state} />
                      </td>
                      <td className="py-2 pr-3 tabular-nums text-zinc-500">
                        <span>{row.registered_quantity ?? '—'}</span>
                        {row.discrepancy !== null && (
                          <span
                            className={`ml-1 text-[10px] font-semibold ${
                              row.discrepancy < 0
                                ? 'text-rose-600 dark:text-rose-400'
                                : 'text-emerald-600 dark:text-emerald-400'
                            }`}
                          >
                            ({row.discrepancy > 0 ? '+' : ''}
                            {fmtNumber(row.discrepancy, 0)})
                          </span>
                        )}
                      </td>
                      <td className="py-2 pr-4 text-[10px] text-zinc-400">{fmtTime(row.observed_at)}</td>
                      <td className="py-2 pr-4">
                        {row.product_id ? (
                          <label className="cursor-pointer rounded border border-zinc-200/70 px-1.5 py-0.5 text-[10px] font-semibold text-zinc-600 hover:bg-zinc-100 dark:border-white/10 dark:text-zinc-300 dark:hover:bg-zinc-800/60">
                            {uploading === row.product_id ? 'Uploading…' : 'Add image'}
                            <input
                              type="file"
                              accept="image/*"
                              className="hidden"
                              onChange={(event) => {
                                const file = event.target.files?.[0]
                                if (file && row.product_id) {
                                  void uploadReference(row.product_id, row.sku, file)
                                }
                                event.target.value = ''
                              }}
                            />
                          </label>
                        ) : (
                          <span className="text-zinc-400">—</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </AsyncContent>
      </SurfaceCard>

      {estimates.data && estimates.data.length === 0 && (
        <EmptyState
          title="No SKU has been detected on shelves yet"
          hint="Estimates appear once a camera recognizes catalog items."
          icon={<Package className="h-6 w-6 text-zinc-400" />}
        />
      )}
    </div>
  )
}
