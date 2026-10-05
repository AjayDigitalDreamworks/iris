import type {
  Alert,
  ApiError,
  Camera,
  CameraHealth,
  CameraZone,
  EventEnvelope,
  InventoryEstimate,
  InventoryItem,
  InventorySummary,
  LiveDetections,
  ModelVersion,
  Page,
  ReportFormat,
  ReportResult,
  ReportType,
  Product,
  QueueMeasurement,
  Shelf,
  StreamTelemetry,
  Store,
  YOLO11Result,
  DashboardData,
  FootfallResponse,
} from '../types/api'


/** Backend base URL. Empty means "same origin", which the Vite dev proxy forwards. */
const BASE = import.meta.env.VITE_API_BASE ?? ''

export class ApiRequestError extends Error {
  readonly status: number
  readonly code: string
  readonly requestId?: string

  constructor(status: number, error: ApiError) {
    super(error.message)
    this.name = 'ApiRequestError'
    this.status = status
    this.code = error.code
    this.requestId = error.request_id
  }
}

type Query = Record<string, string | number | boolean | undefined | null>

function buildUrl(path: string, query?: Query): string {
  const url = `${BASE}/api/v1${path}`
  if (!query) return url
  const params = new URLSearchParams()
  for (const [key, value] of Object.entries(query)) {
    if (value !== undefined && value !== null && value !== '') params.set(key, String(value))
  }
  const qs = params.toString()
  return qs ? `${url}?${qs}` : url
}

async function request<T>(path: string, init?: RequestInit & { query?: Query }): Promise<T> {
  const { query, ...rest } = init ?? {}
  const response = await fetch(buildUrl(path, query), {
    ...rest,
    headers: { 'Content-Type': 'application/json', ...(rest.headers ?? {}) },
  })

  const text = await response.text()
  let parsed: unknown = null
  if (text) {
    try {
      parsed = JSON.parse(text)
    } catch {
      parsed = null
    }
  }

  if (!response.ok) {
    const envelope = (parsed as { error?: ApiError } | null)?.error
    throw new ApiRequestError(
      response.status,
      envelope ?? { code: 'http_error', message: `Request failed (${response.status})` },
    )
  }
  return parsed as T
}

const get = <T>(path: string, query?: Query) => request<T>(path, { query })
const post = <T>(path: string, body?: unknown) =>
  request<T>(path, {
    method: 'POST',
    body: body === undefined ? undefined : JSON.stringify(body),
  })
const patch = <T>(path: string, body: unknown) =>
  request<T>(path, { method: 'PATCH', body: JSON.stringify(body) })
const del = <T>(path: string) => request<T>(path, { method: 'DELETE' })

export const api = {
  health: () => get<Record<string, unknown>>('/system/health'),
  stats: () => get<Record<string, unknown>>('/system/stats'),
  config: () => get<Record<string, unknown>>('/system/config'),
  settings: (query?: Query) => get<Page<Record<string, unknown>>>('/system/settings', query),

  /** Stores own the cameras; the registration form needs a store_id. */
  stores: () => get<Store[]>('/stores'),

  cameras: (query?: Query) => get<Page<Camera>>('/cameras', query),
  camera: (id: string) => get<Camera>(`/cameras/${id}`),
  createCamera: (body: unknown) => post<Camera>('/cameras', body),
  updateCamera: (id: string, body: unknown) => patch<Camera>(`/cameras/${id}`, body),
  deleteCamera: (id: string) => del<{ message: string }>(`/cameras/${id}`),
  startCamera: (id: string) => post<{ message: string }>(`/cameras/${id}/start`),
  stopCamera: (id: string) => post<{ message: string }>(`/cameras/${id}/stop`),
  restartCamera: (id: string) => post<{ message: string }>(`/cameras/${id}/restart`),
  cameraHealth: (id: string) => get<CameraHealth>(`/cameras/${id}/health`),
  zones: (cameraId: string) => get<CameraZone[]>(`/cameras/${cameraId}/zones`),
  /** Give a camera default zones (+ a shelf) so analytics have a region to measure. */
  provisionZones: (cameraId: string) =>
    post<{ camera_id: string; zones: number; shelves: number; note: string }>(
      `/cameras/${cameraId}/zones/provision`,
    ),
  provisionAllZones: () =>
    post<{ zones: number; shelves: number }>('/cameras/provision-zones'),

  shelves: (query?: Query) => get<Page<Shelf>>('/shelves', query),
  products: (query?: Query) => get<Page<Product>>('/products', query),
  productBySku: (sku: string) => get<Product>(`/products/by-sku/${encodeURIComponent(sku)}`),
  // Row-level detections (x1/y1/x2/y2/class_name/confidence). Note: the
  // `/analytics/detections` path is an aggregate SUMMARY, not rows — pointing
  // the Live overlay at it silently produced zero boxes.
  detections: (query?: Query) => get<Page<Record<string, unknown>>>('/detections', query),
  dashboard: (query?: Query) => get<DashboardData>('/analytics/dashboard', query),
  shelfAnalytics: (query?: Query) => get<Record<string, unknown>>('/analytics/shelves', query),

  inventory: (query?: Query) => get<InventoryEstimate[]>('/inventory', query),
  inventoryItems: (query?: Query) => get<Page<InventoryItem>>('/inventory/snapshots', query),
  inventorySummary: () => get<InventorySummary>('/inventory/summary'),
  inventoryBySku: (sku: string) =>
    get<Record<string, unknown>>(`/inventory/${encodeURIComponent(sku)}`),
  adjustInventory: (sku: string, body: unknown) =>
    post<Record<string, unknown>>(`/inventory/${encodeURIComponent(sku)}/adjust`, body),

  footfall: (query?: Query) => get<FootfallResponse>('/customers/footfall', query),
  occupancy: () => get<Record<string, unknown>>('/customers/occupancy'),
  queues: (query?: Query) => get<QueueMeasurement[]>('/queues', query),
  queueThresholds: () => get<Record<string, unknown>>('/queues/thresholds'),
  updateQueueThresholds: (body: unknown) =>
    patch<Record<string, unknown>>('/queues/thresholds', body),

  alerts: (query?: Query) => get<Page<Alert>>('/alerts', query),
  activeAlerts: () => get<Alert[]>('/alerts/active'),
  alertSummary: () => get<Record<string, unknown>>('/alerts/summary'),
  acknowledgeAlert: (id: string, note?: string) =>
    post<Alert>(`/alerts/${id}/acknowledge`, { note: note ?? '' }),
  resolveAlert: (id: string, note: string) => post<Alert>(`/alerts/${id}/resolve`, { note }),

  models: () => get<ModelVersion[]>('/models'),
  modelRegistry: () => get<Record<string, unknown>>('/models/registry'),

  recentEvents: (limit = 25) => get<EventEnvelope[]>('/events/recent', { limit }),
  detectFrame: (formData: FormData, conf = 0.25) =>
    post<YOLO11Result>(`/inference/detect?conf=${conf}`, formData),
  streamTelemetry: (cameraId: string) =>
    get<StreamTelemetry>(`/inference/telemetry/${cameraId}`),

  /** Boxes for the CURRENT frame (from memory). Use this for the live overlay,
   *  never the detection history — that stacks many frames' boxes at once. */
  liveDetections: (cameraId: string) =>
    get<LiveDetections>(`/streams/${cameraId}/detections`),

  /** Upload a product photo so SKU recognition can name that product. */
  uploadReferenceImage: async (productId: string, file: File) => {
    const form = new FormData()
    form.append('file', file)
    // NOTE: no Content-Type header — the browser sets the multipart boundary.
    const response = await fetch(`${BASE}/api/v1/products/${productId}/reference-image`, {
      method: 'POST',
      body: form,
    })
    const text = await response.text()
    let parsed: unknown = null
    if (text) {
      try {
        parsed = JSON.parse(text)
      } catch {
        parsed = null
      }
    }
    if (!response.ok) {
      const envelope = (parsed as { error?: ApiError } | null)?.error
      throw new ApiRequestError(
        response.status,
        envelope ?? { code: 'http_error', message: `Upload failed (${response.status})` },
      )
    }
    return parsed as { ok: boolean; message: string; detail: { dimension: number } }
  },

  /** Generate a report file (csv / json / pdf) and return its metadata. */

  generateReport: (body: {
    report_type: ReportType
    format: ReportFormat
    range_start?: string
    range_end?: string
    camera_id?: string
    shelf_id?: string
  }) => post<ReportResult>('/reports/export', body),
}

export { BASE as API_BASE }