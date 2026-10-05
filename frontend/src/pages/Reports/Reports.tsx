import { useMemo, useState } from 'react'
import {
  FileBarChart2,
  Download,
  FileSpreadsheet,
  FileJson,
  FileText,
  Sparkles,
  Loader2,
  CheckCircle2,
  Layers,
  Clock,
  Bell,
  AlertTriangle,
  Package,
} from 'lucide-react'
import { api, API_BASE } from '../../services/api'
import { useApi } from '../../hooks/useAsync'
import {
  SurfaceCard,
  ActionButton,
  ErrorState,
  fmtNumber,
  fmtTime,
} from '../../components/ui'
import type { ReportFormat, ReportResult, ReportType } from '../../types/api'

const REPORTS: { id: ReportType; label: string; hint: string; icon: typeof Package }[] = [
  {
    id: 'inventory',
    label: 'Inventory estimates',
    hint: 'SKU-level visible stock estimates with confidence metrics. Visual estimates rather than counts.',
    icon: Package,
  },
  {
    id: 'footfall',
    label: 'Customer Footfall',
    hint: 'Hourly entrance and exit crossing counts derived from configured line crossings.',
    icon: Clock,
  },
  {
    id: 'queue',
    label: 'Queue Analytics',
    hint: 'Station counts, queue lengths, and estimated wait times across service counters.',
    icon: Clock,
  },
  {
    id: 'shelves',
    label: 'Shelf Occupancy',
    hint: 'Occupancy area surface ratio and empty positions per observed planogram fixture.',
    icon: Layers,
  },
  {
    id: 'alerts',
    label: 'Incident Alerts',
    hint: 'Historical record of raised alerts with severity categorization and resolution timestamps.',
    icon: Bell,
  },
  {
    id: 'anomalies',
    label: 'Shelf Anomaly Candidates',
    hint: 'Visual-change variance candidates for staff review. Informational cues, not evidence of theft.',
    icon: AlertTriangle,
  },
  {
    id: 'detections',
    label: 'Raw Vision Detections',
    hint: 'Granular optical detection records. Substantial dataset size, best used with concise intervals.',
    icon: Sparkles,
  },
]

const FORMATS: { id: ReportFormat; label: string; icon: typeof FileSpreadsheet }[] = [
  { id: 'csv', label: 'CSV Spreadsheet', icon: FileSpreadsheet },
  { id: 'json', label: 'JSON Dataset', icon: FileJson },
  { id: 'pdf', label: 'PDF Document', icon: FileText },
]

const RANGES = [
  { id: '24h', label: 'Last 24 Hours', hours: 24 },
  { id: '7d', label: 'Last 7 Days', hours: 24 * 7 },
  { id: '30d', label: 'Last 30 Days', hours: 24 * 30 },
]

function isoRange(hours: number): { start: string; end: string } {
  const end = new Date()
  const start = new Date(end.getTime() - hours * 3600 * 1000)
  return { start: start.toISOString(), end: end.toISOString() }
}

export default function Reports() {
  const [reportType, setReportType] = useState<ReportType>('inventory')
  const [format, setFormat] = useState<ReportFormat>('csv')
  const [range, setRange] = useState('7d')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<ReportResult | null>(null)

  const cameras = useApi(() => api.cameras({ limit: 100 }), [])
  const [cameraId, setCameraId] = useState('')

  const selected = useMemo(
    () => REPORTS.find((r) => r.id === reportType) ?? REPORTS[0],
    [reportType],
  )

  async function generate() {
    setBusy(true)
    setError(null)
    setResult(null)
    try {
      const hours = RANGES.find((r) => r.id === range)?.hours ?? 168
      const { start, end } = isoRange(hours)
      const created = await api.generateReport({
        report_type: reportType,
        format,
        range_start: start,
        range_end: end,
        ...(cameraId ? { camera_id: cameraId } : {}),
      })
      setResult(created)
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc))
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
            Export &amp; Reporting
          </h1>
          <span className="inline-flex items-center gap-1 rounded bg-indigo-500/10 px-1.5 py-0.5 text-[10px] font-bold text-indigo-600 dark:text-indigo-400">
            <FileBarChart2 className="h-3 w-3" /> Data Queries
          </span>
        </div>
        <p className="truncate text-[11px] text-zinc-500 dark:text-zinc-400">
          Export verified telemetry over defined time intervals as CSV, JSON, or PDF
        </p>
      </div>

      {/* 2-Zone Workspace (Fills Remaining Viewport) */}
      <div className="grid min-h-0 flex-1 grid-cols-1 gap-3 overflow-hidden lg:grid-cols-12">
        {/* Left Form Area (7 cols) */}
        <SurfaceCard
          title="Build Report Query"
          subtitle="Configure report parameters and scope"
          icon={<FileBarChart2 className="h-4 w-4 text-indigo-500" />}
          scrollable
          className="flex-1 min-h-0 overflow-hidden lg:col-span-7"
        >
          <div className="space-y-3">
            <div className="grid gap-2.5 sm:grid-cols-2">
              <div>
                <label className="mb-1 block text-[10px] font-semibold uppercase text-zinc-500">
                  Report Type
                </label>
                <select
                  value={reportType}
                  onChange={(e) => setReportType(e.target.value as ReportType)}
                  className="precision-input w-full rounded-md px-2.5 py-1.5 text-xs font-medium"
                >
                  {REPORTS.map((r) => (
                    <option key={r.id} value={r.id}>
                      {r.label}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="mb-1 block text-[10px] font-semibold uppercase text-zinc-500">
                  Time Range
                </label>
                <select
                  value={range}
                  onChange={(e) => setRange(e.target.value)}
                  className="precision-input w-full rounded-md px-2.5 py-1.5 text-xs font-medium"
                >
                  {RANGES.map((r) => (
                    <option key={r.id} value={r.id}>
                      {r.label}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="mb-1 block text-[10px] font-semibold uppercase text-zinc-500">
                  Format
                </label>
                <select
                  value={format}
                  onChange={(e) => setFormat(e.target.value as ReportFormat)}
                  className="precision-input w-full rounded-md px-2.5 py-1.5 text-xs font-medium uppercase"
                >
                  {FORMATS.map((f) => (
                    <option key={f.id} value={f.id}>
                      {f.label}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="mb-1 block text-[10px] font-semibold uppercase text-zinc-500">
                  Camera Scope (Optional)
                </label>
                <select
                  value={cameraId}
                  onChange={(e) => setCameraId(e.target.value)}
                  className="precision-input w-full rounded-md px-2.5 py-1.5 text-xs font-medium"
                >
                  <option value="">All cameras</option>
                  {(cameras.data?.items ?? []).map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.name}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* Hint Box */}
            <div className="rounded-lg border border-zinc-200/60 bg-zinc-50/50 p-2.5 text-[11px] text-zinc-600 dark:border-white/5 dark:bg-zinc-900/40 dark:text-zinc-300">
              <span className="font-semibold text-zinc-800 dark:text-zinc-200">{selected.label}: </span>
              <span>{selected.hint}</span>
            </div>

            {/* Generate Action */}
            <div className="flex items-center gap-2 pt-1">
              <ActionButton
                variant="primary"
                size="sm"
                onClick={generate}
                disabled={busy}
                icon={busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Download className="h-3.5 w-3.5" />}
              >
                {busy ? 'Compiling…' : `Generate ${selected.label}`}
              </ActionButton>
              {busy && (
                <span className="text-[11px] text-zinc-400">
                  Server is computing table export.
                </span>
              )}
            </div>

            {error && (
              <ErrorState message={`Report execution failed: ${error}`} onRetry={generate} />
            )}

            {/* Download Box */}
            {result && (
              <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/10 p-3 text-xs dark:bg-emerald-950/20">
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
                    <div>
                      <p className="font-bold text-zinc-900 dark:text-zinc-100">
                        {fmtNumber(result.row_count)} rows in {result.report_type}
                      </p>
                      <p className="text-[10px] text-zinc-500 dark:text-zinc-400">
                        {fmtTime(result.generated_at)} as {result.format.toUpperCase()}
                      </p>
                    </div>
                  </div>

                  <a
                    className="inline-flex items-center gap-1 rounded bg-emerald-600 px-2.5 py-1 text-xs font-semibold text-white transition hover:bg-emerald-500 active:scale-95"
                    href={`${API_BASE}/api/v1/reports/download?path=${encodeURIComponent(result.filename)}`}
                  >
                    <Download className="h-3 w-3" />
                    Download
                  </a>
                </div>

                {result.row_count === 0 && (
                  <p className="mt-1 text-[10px] text-zinc-500">
                    Empty dataset in this query range.
                  </p>
                )}
              </div>
            )}
          </div>
        </SurfaceCard>

        {/* Right Catalog Area (5 cols) */}
        <SurfaceCard
          title="Catalog Capabilities"
          subtitle="Overview of available datasets"
          icon={<Layers className="h-4 w-4 text-zinc-400" />}
          scrollable
          className="flex-1 min-h-0 overflow-hidden lg:col-span-5"
        >
          <div className="space-y-1.5">
            {REPORTS.map((r) => {
              const Icon = r.icon
              const isSelected = r.id === reportType
              return (
                <div
                  key={r.id}
                  onClick={() => setReportType(r.id)}
                  className={`cursor-pointer rounded-lg border p-2.5 transition-all ${
                    isSelected
                      ? 'border-indigo-500/50 bg-indigo-500/10'
                      : 'border-zinc-200/60 bg-white/40 hover:bg-zinc-100 dark:border-white/5 dark:bg-zinc-900/40 dark:hover:bg-zinc-800/60'
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <Icon className="h-3.5 w-3.5 text-indigo-500" />
                    <span className="text-xs font-bold text-zinc-800 dark:text-zinc-200">
                      {r.label}
                    </span>
                  </div>
                  <p className="mt-1 text-[11px] leading-relaxed text-zinc-500 dark:text-zinc-400">
                    {r.hint}
                  </p>
                </div>
              )
            })}
          </div>
        </SurfaceCard>
      </div>
    </div>
  )
}
