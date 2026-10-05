/**
 * Types mirroring the IRIS backend OpenAPI schemas.
 *
 * Field names and types are taken from the live `/openapi.json`
 * (`CameraRead`, `InventoryEstimate`, `AlertRead`, `QueueResponse`, …).
 * Drift between backend and frontend is a contract bug — never widen a field
 * to `any` to silence a type error.
 */

export type CameraStatus = 'online' | 'offline' | 'error' | 'disabled' | 'connecting' | 'stopped'
export type CameraSourceType = 'rtsp' | 'webcam' | 'ip_webcam' | 'video_file' | 'image' | 'synthetic' | 'url'
export type ZoneType = 'shelf' | 'queue' | 'entrance_line' | 'exit_line' | 'zone' | 'privacy'
export type AlertStatus = 'candidate' | 'suppressed' | 'active' | 'acknowledged' | 'resolved'
export type AlertSeverity = 'info' | 'warning' | 'critical'
export type QueueStatus = 'normal' | 'busy' | 'critical'
export type StockState = 'ok' | 'low_stock' | 'out_of_stock' | 'unknown'

export interface Page<T> {
  items: T[]
  total: number
  limit: number
  offset: number
  has_more: boolean
}

/** A physical store — the camera form needs its id to register a camera. */
export interface Store {
  id: string
  code: string
  name: string
  location: string | null
  timezone: string
  currency: string
  is_active: boolean
}

export interface Camera {
  id: string
  store_id: string
  name: string
  location: string | null
  role: string
  source_type: CameraSourceType
  status: CameraStatus
  is_enabled: boolean
  run_inference: boolean
  privacy_mask: string | null
  capture_fps: number
  frame_skip: number
  width: number | null
  height: number | null
  codec: string | null
  native_fps: number | null
  /** True when a credential-bearing stream URL is stored; the URL itself is never returned. */
  has_credentials: boolean
  stream_url_preview: string | null
  last_frame_at: string | null
  last_connected_at: string | null
  last_disconnected_at: string | null
  last_error: string | null
  reconnect_count: number
  total_frames: number
  dropped_frames: number
  notes: string | null
  created_at: string
  updated_at: string
}

export interface CameraZone {
  id: string
  camera_id: string
  name: string
  zone_type: ZoneType
  polygon: number[][] | null
  line: number[][] | null
  service_stations: number
  is_active: boolean
  description: string | null
}

export interface Shelf {
  id: string
  store_id: string
  camera_id: string | null
  zone_id: string | null
  code: string
  name: string
  aisle: string | null
  level: number | null
  capacity_units: number
  polygon: number[][] | null
  is_active: boolean
  description: string | null
}

export interface Product {
  id: string
  store_id: string
  category_id: string | null
  sku: string
  name: string
  brand: string | null
  barcode: string | null
  unit_size: string | null
  price: number | null
  expected_quantity: number | null
  min_threshold: number | null
  facing_capacity: number | null
  quantity_source: string | null
  last_counted_at: string | null
  reference_image_path: string | null
  shelf_id: string | null
  detector_class_name: string | null
  is_active: boolean
  notes: string | null
  created_at: string
  updated_at: string
}

export interface InventoryEstimate {
  product_id: string | null
  product_name: string | null
  sku: string
  shelf_id: string | null
  shelf_code: string | null
  camera_id: string | null
  observed_quantity: number
  estimated_quantity: number
  /** -1 means "unknown" — capacity is not configured. Never render that as 0%. */
  stock_percentage: number
  capacity_units: number
  confidence: number
  occupancy_pct: number | null
  observation_count: number
  observation_age_seconds: number | null
  method: string
  source: string
  state: StockState
  is_stale: boolean
  low_stock_threshold_pct: number | null
  out_of_stock_threshold_pct: number | null
  registered_quantity: number | null
  registered_source: string | null
  discrepancy: number | null
  observed_at: string
}

export interface InventoryItem {
  id: string
  product_id: string | null
  sku: string
  shelf_id: string | null
  camera_id: string | null
  observed_quantity: number
  estimated_quantity: number
  stock_percentage: number
  capacity_units: number
  confidence: number
  occupancy_pct: number | null
  observation_count: number
  observation_age_seconds: number | null
  method: string
  source: string
  notes: string[] | null
  observed_at: string
}

export interface InventorySummary {
  generated_at: string
  total_skus: number
  observed_skus: number
  healthy_count: number
  low_stock_count: number
  out_of_stock_count: number
  unknown_count: number
  stale_skus: number
  mean_stock_percentage: number | null
  mean_confidence: number | null
}

export interface Alert {
  id: string
  store_id: string | null
  camera_id: string | null
  shelf_id: string | null
  zone_id: string | null
  product_id: string | null
  sku: string | null
  alert_type: string
  severity: AlertSeverity
  status: AlertStatus
  title: string
  message: string
  confidence: number
  observation_count: number
  occurrence_count: number
  dedup_key: string
  payload: Record<string, unknown> | null
  evidence_path: string | null
  evidence_expires_at: string | null
  created_at: string
  last_observed_at: string
  acknowledged_at: string | null
  acknowledged_by: string | null
  resolved_at: string | null
  resolved_by: string | null
  resolution_note: string | null
  resolution_source: string | null
}

export interface QueuePoint {
  bucket: string
  queue_length: number
  service_zone_occupancy: number
  estimated_wait_seconds: number | null
  status: QueueStatus
}

export interface QueueMeasurement {
  zone_id: string
  zone_name: string
  camera_id: string | null
  current_queue_length: number
  service_zone_occupancy: number
  service_stations: number
  estimated_wait_seconds: number | null
  status: QueueStatus
  busy_threshold: number
  critical_threshold: number
  mean_queue_length: number
  peak_queue_length: number
  granularity: string
  range_start: string
  range_end: string
  note: string
  points: QueuePoint[]
}

export interface ModelVersion {
  id: string
  name: string
  task: string
  version: string
  status: string
  weights_path: string
  weights_sha256: string | null
  provider: string
  source: string | null
  device: string | null
  input_size: number | null
  confidence_threshold: number | null
  labels: string[] | null
  metrics: Record<string, unknown> | null
  metrics_measured_at: string | null
  is_active: boolean
  notes: string | null
  created_at: string
  updated_at: string
}

export interface Detection {
  id: string
  camera_id: string | null
  zone_id: string | null
  shelf_id: string | null
  class_id: number
  class_name: string
  task: string
  product_id: string | null
  sku: string | null
  similarity: number | null
  x1: number
  y1: number
  x2: number
  y2: number
  confidence: number
  track_id: number | null
  source: string
  model_id: string | null
  job_id: string | null
  frame_timestamp: string
  inference_latency_ms: number | null
}

export interface CameraHealth {
  camera_id: string
  name: string
  status: CameraStatus
  is_enabled: boolean
  is_stale: boolean
  is_frozen: boolean
  capture_fps: number
  inference_fps: number
  inference_latency_ms: number
  frames_processed: number
  frames_dropped: number
  drop_ratio: number
  queue_depth: number
  reconnect_count: number
  seconds_since_last_frame: number
  width: number | null
  height: number | null
  last_error: string | null
  uptime_ratio_24h: number | null
  /** Free-form per-camera diagnostics (source state, frame pacing, errors). */
  detail: Record<string, unknown>
}

// ── reports (README §5) ──────────────────────────────────────────────────
export type ReportType =
  | 'inventory'
  | 'footfall'
  | 'queue'
  | 'alerts'
  | 'shelves'
  | 'detections'
  | 'anomalies'
export type ReportFormat = 'csv' | 'json' | 'pdf'

export interface ReportResult {
  report_type: string
  format: string
  filename: string
  row_count: number
  generated_at: string
  download_path: string
}

export interface EventEnvelope {
  event_id: string
  event_type: string
  timestamp: string
  camera_id: string | null
  severity: string | null
  payload: Record<string, unknown>
  evidence?: Record<string, unknown>
}

export interface ApiError {
  code: string
  message: string
  request_id?: string
  details?: { errors?: { loc: (string | number)[]; msg: string; type: string }[] }
}

export interface YOLO11DetectionItem {
  id: string
  label: string
  confidence: number
  bbox: [number, number, number, number]
  raw_bbox?: [number, number, number, number]
  color: string
  track_id?: number | null
  is_multipack?: boolean
}

export interface YOLO11Telemetry {
  preprocess_ms: number
  inference_ms: number
  postprocess_ms: number
  total_ms: number
  effective_fps: number
  device: string
  model: string
  hardware_provider: string
  cuda_available: boolean
  fp16_enabled: boolean
}

export interface YOLO11Result {
  detections: YOLO11DetectionItem[]
  telemetry: YOLO11Telemetry
  total_items: number
  class_breakdown: Record<string, number>
  frame_width: number
  frame_height: number
}

export interface LiveDetection {
  class_name: string
  task: string
  confidence: number
  x1: number
  y1: number
  x2: number
  y2: number
  track_id: number | null
  sku: string | null
  is_person: boolean
  is_multipack: boolean
}

export interface LiveDetections {
  camera_id: string
  detections: LiveDetection[]
  frame: {
    width?: number
    height?: number
    frame_index?: number
    captured_at?: string
    latency_ms?: number
  }
}

export interface StreamTelemetry {
  camera_id: string
  name: string
  status: string
  capture_fps: number
  inference_fps: number
  inference_latency_ms: number | null
  frames_processed: number
  frames_dropped: number
  drop_ratio: number
  hardware: {
    device: string
    provider: string
    cuda_available: boolean
    platform: string
  }
}

export interface FootfallPoint {
  bucket: string
  entries: number
  exits: number
  net: number
  current_occupancy?: number | null
}

export interface FootfallResponse {
  camera_id: string | null
  camera_name: string | null
  granularity: string
  total_entries: number
  total_exits: number
  current_occupancy: number
  peak_occupancy: number
  mean_dwell_seconds: number | null
  max_dwell_seconds: number | null
  active_tracks: number
  points: FootfallPoint[]
  note?: string
}

export interface DashboardData {
  generated_at: string
  store_name: string | null
  cameras: {
    total: number
    online: number
    offline: number
    degraded: number
    fps_average: number
  }
  inventory: InventorySummary
  alerts: {
    critical: number
    warning: number
    info: number
    total_open: number
    mean_resolution_seconds: number | null
  }
  customers: FootfallResponse
  queues: QueueMeasurement[]
  ai: Record<string, unknown>
  recent_events: Array<{
    event_id: string
    event_type: string
    timestamp: string
    severity: string
    message: string
    payload?: Record<string, unknown>
  }>
  skus_configured: number
}
