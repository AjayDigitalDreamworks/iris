import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiRequestError } from '../services/api'
import { eventSocket, type ConnectionState, type LiveEvent } from '../services/ws'

export interface AsyncState<T> {
  data: T | null
  loading: boolean
  error: string | null
  reload: () => void
}

/** Fetch-on-mount with explicit loading/error state and a manual reload. */
export function useApi<T>(loader: () => Promise<T>, deps: unknown[] = []): AsyncState<T> {
  const [data, setData] = useState<T | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [nonce, setNonce] = useState(0)
  const mounted = useRef(true)

  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
    }
  }, [])

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)
    loader()
      .then((value) => {
        if (!cancelled) {
          setData(value)
          setLoading(false)
        }
      })
      .catch((err: unknown) => {
        if (cancelled) return
        setError(err instanceof ApiRequestError ? err.message : String(err))
        setLoading(false)
      })
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce])

  const reload = useCallback(() => setNonce((n) => n + 1), [])
  return { data, loading, error, reload }
}

/** Subscribe to the live event socket. */
export function useLiveEvents(handler: (event: LiveEvent) => void): void {
  const ref = useRef(handler)
  ref.current = handler
  useEffect(() => eventSocket.subscribe((event) => ref.current(event)), [])
}

/** Track the socket connection state so the UI can show a live/stale banner. */
export function useSocketState(): ConnectionState {
  const [state, setState] = useState<ConnectionState>(eventSocket.connectionState)
  useEffect(() => eventSocket.onStateChange(setState), [])
  return state
}

/** Re-run `loader` on an interval. Paused while the tab is hidden. */
export function usePolling(loader: () => void, intervalMs: number, enabled = true): void {
  const ref = useRef(loader)
  ref.current = loader
  useEffect(() => {
    if (!enabled) return
    const id = window.setInterval(() => {
      if (document.visibilityState === 'visible') ref.current()
    }, intervalMs)
    return () => window.clearInterval(id)
  }, [intervalMs, enabled])
}