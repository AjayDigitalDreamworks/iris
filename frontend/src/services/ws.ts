/**
 * WebSocket client for `/ws/events`.
 *
 * Reconnects with bounded exponential backoff. The dashboard relies on this for
 * live updates, but every page also has a REST fallback, so a dropped socket
 * degrades to polling rather than a frozen screen.
 */

export type ConnectionState = 'connecting' | 'open' | 'closed' | 'reconnecting'

export interface LiveEvent {
  event_id: string
  event_type: string
  timestamp: string
  camera_id?: string | null
  severity?: string | null
  payload?: Record<string, unknown>
}

type Listener = (event: LiveEvent) => void
type StateListener = (state: ConnectionState) => void

const MAX_BACKOFF_MS = 15_000
const BASE_BACKOFF_MS = 500

export class EventSocket {
  private socket: WebSocket | null = null
  private listeners = new Set<Listener>()
  private stateListeners = new Set<StateListener>()
  private attempt = 0
  private timer: number | null = null
  private closed = false
  private state: ConnectionState = 'closed'

  connect(): void {
    this.closed = false
    this.open()
  }

  disconnect(): void {
    this.closed = true
    if (this.timer !== null) {
      window.clearTimeout(this.timer)
      this.timer = null
    }
    this.socket?.close()
    this.socket = null
    this.setState('closed')
  }

  subscribe(listener: Listener): () => void {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }

  onStateChange(listener: StateListener): () => void {
    this.stateListeners.add(listener)
    listener(this.state)
    return () => this.stateListeners.delete(listener)
  }

  get connectionState(): ConnectionState {
    return this.state
  }

  private setState(state: ConnectionState): void {
    this.state = state
    this.stateListeners.forEach((listener) => listener(state))
  }

  private url(): string {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const base = import.meta.env.VITE_WS_BASE
    if (base) return `${base}/ws/events`
    return `${protocol}//${window.location.host}/ws/events`
  }

  private open(): void {
    this.setState(this.attempt === 0 ? 'connecting' : 'reconnecting')
    try {
      const socket = new WebSocket(this.url())
      this.socket = socket

      socket.onopen = () => {
        this.attempt = 0
        this.setState('open')
      }
      socket.onmessage = (message) => {
        try {
          const parsed = JSON.parse(message.data as string) as LiveEvent
          this.listeners.forEach((listener) => listener(parsed))
        } catch {
          // A malformed frame must not kill the socket.
        }
      }
      socket.onclose = () => {
        this.socket = null
        if (this.closed) return
        this.scheduleReconnect()
      }
      socket.onerror = () => socket.close()
    } catch {
      this.scheduleReconnect()
    }
  }

  private scheduleReconnect(): void {
    this.setState('reconnecting')
    const delay = Math.min(MAX_BACKOFF_MS, BASE_BACKOFF_MS * 2 ** this.attempt)
    this.attempt += 1
    this.timer = window.setTimeout(() => this.open(), delay)
  }
}

export const eventSocket = new EventSocket()