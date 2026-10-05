import {
  Cpu,
  HardDrive,
} from 'lucide-react'
import { api } from '../../services/api'
import { useApi } from '../../hooks/useAsync'
import {
  AsyncContent,
  Badge,
  SurfaceCard,
} from '../../components/ui'

export default function Models() {
  const models = useApi(() => api.models(), [])
  const registry = useApi(() => api.modelRegistry(), [])

  return (
    <div className="flex h-full flex-col gap-3 overflow-hidden">
      {/* Top Header */}
      <div className="shrink-0">
        <div className="flex items-center gap-2">
          <h1 className="text-base font-bold tracking-tight text-zinc-900 dark:text-zinc-100">
            Model Registry
          </h1>
          <span className="inline-flex items-center gap-1 rounded bg-indigo-500/10 px-1.5 py-0.5 text-[10px] font-bold text-indigo-600 dark:text-indigo-400">
            <Cpu className="h-3 w-3" /> Inferencing
          </span>
        </div>
        <p className="truncate text-[11px] text-zinc-500 dark:text-zinc-400">
          Neural network checkpoints, compute device targets, and accuracy validation metrics
        </p>
      </div>

      {/* 2-Zone Workspace (Fills Remaining Viewport) */}
      <div className="grid min-h-0 flex-1 grid-cols-1 gap-3 overflow-hidden lg:grid-cols-12">
        {/* Left Area (7 cols): Models Table */}
        <SurfaceCard
          title="Registered Vision Models"
          subtitle={`${models.data?.length ?? 0} active checkpoints`}
          icon={<Cpu className="h-4 w-4 text-indigo-500" />}
          scrollable
          className="flex-1 min-h-0 overflow-hidden lg:col-span-7"
          bodyClassName="p-0 overflow-hidden flex flex-col"
        >
          <AsyncContent state={models} empty="No models registered" isEmpty={(d) => d.length === 0} rows={4}>
            {(data) => (
              <div className="custom-scrollbar flex-1 overflow-auto">
                <table className="w-full text-left text-xs">
                  <thead className="sticky top-0 z-10 border-b border-zinc-200 bg-zinc-50/95 backdrop-blur dark:border-white/10 dark:bg-zinc-900/95">
                    <tr className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400">
                      <th className="py-2.5 pl-4 pr-3">Model</th>
                      <th className="py-2.5 pr-3">Task</th>
                      <th className="py-2.5 pr-3">Version</th>
                      <th className="py-2.5 pr-3">Status</th>
                      <th className="py-2.5 pr-3">Device</th>
                      <th className="py-2.5 pr-3 text-right">Threshold</th>
                      <th className="py-2.5 pr-4">Metrics</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-zinc-200/60 dark:divide-white/5">
                    {data.map((model) => (
                      <tr
                        key={model.id}
                        className="transition-colors hover:bg-zinc-100/60 dark:hover:bg-zinc-800/40"
                      >
                        <td className="py-2 pl-4 pr-3 font-semibold text-zinc-900 dark:text-zinc-100">
                          {model.name}
                        </td>
                        <td className="py-2 pr-3 capitalize text-zinc-500">
                          {model.task}
                        </td>
                        <td className="py-2 pr-3 font-mono text-[11px] text-indigo-600 dark:text-indigo-400">
                          {model.version}
                        </td>
                        <td className="py-2 pr-3">
                          <Badge value={model.status} />
                        </td>
                        <td className="py-2 pr-3 font-mono text-[11px] text-zinc-500">
                          {model.device ?? '—'}
                        </td>
                        <td className="py-2 pr-3 text-right font-mono tabular-nums">
                          {model.confidence_threshold ?? '—'}
                        </td>
                        <td className="py-2 pr-4 text-[10px]">
                          {model.metrics ? (
                            <code className="block max-w-[140px] truncate rounded bg-zinc-100 px-1 py-0.5 font-mono text-zinc-600 dark:bg-zinc-800 dark:text-zinc-300">
                              {JSON.stringify(model.metrics)}
                            </code>
                          ) : (
                            <span className="text-zinc-400">Unmeasured</span>
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

        {/* Right Area (5 cols): Runtime Status JSON */}
        <SurfaceCard
          title="Runtime Provider State"
          subtitle="Execution provider and weights verification status"
          icon={<HardDrive className="h-4 w-4 text-zinc-400" />}
          scrollable
          className="flex-1 min-h-0 overflow-hidden lg:col-span-5"
          bodyClassName="p-2 font-mono text-xs overflow-auto custom-scrollbar"
        >
          <AsyncContent state={registry} empty="Registry status unavailable" rows={3}>
            {(data) => (
              <pre className="rounded-lg bg-zinc-950 p-3 text-[11px] text-zinc-300 overflow-auto">
                {JSON.stringify(data, null, 2)}
              </pre>
            )}
          </AsyncContent>
        </SurfaceCard>
      </div>
    </div>
  )
}