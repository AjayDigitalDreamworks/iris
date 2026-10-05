import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest'
import { api, ApiRequestError } from '../services/api'

describe('api client', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
  })
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  const respond = (status: number, body: unknown) =>
    vi.fn().mockResolvedValue({
      ok: status < 400,
      status,
      text: () => Promise.resolve(JSON.stringify(body)),
    })

  it('requests cameras with pagination parameters', async () => {
    const fetchMock = respond(200, { items: [], total: 0, limit: 50, offset: 0, has_more: false })
    vi.stubGlobal('fetch', fetchMock)

    await api.cameras({ limit: 100, source_type: 'rtsp' })

    const [url] = fetchMock.mock.calls[0]
    expect(url).toContain('/api/v1/cameras')
    expect(url).toContain('limit=100')
    expect(url).toContain('source_type=rtsp')
  })

  it('omits empty query parameters instead of sending blanks', async () => {
    const fetchMock = respond(200, { items: [], total: 0, limit: 50, offset: 0, has_more: false })
    vi.stubGlobal('fetch', fetchMock)

    await api.cameras({ status: undefined, search: '' })

    const [url] = fetchMock.mock.calls[0]
    expect(url).not.toContain('status=')
    expect(url).not.toContain('search=')
  })

  it('surfaces the backend error envelope instead of a raw status', async () => {
    const fetchMock = respond(404, {
      error: { code: 'not_found', message: 'Camera x does not exist.', request_id: 'abc123' },
    })
    vi.stubGlobal('fetch', fetchMock)

    await expect(api.camera('x')).rejects.toThrowError(ApiRequestError)

    try {
      await api.camera('x')
      expect.unreachable('should have thrown')
    } catch (error) {
      const err = error as ApiRequestError
      expect(err.status).toBe(404)
      expect(err.code).toBe('not_found')
      expect(err.requestId).toBe('abc123')
    }
  })

  it('does not leak a stack trace when the body is not JSON', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: false,
        status: 500,
        text: () => Promise.resolve('<html>Internal Server Error</html>'),
      }),
    )

    await expect(api.health()).rejects.toMatchObject({ status: 500, code: 'http_error' })
  })

  it('sends a JSON body on alert acknowledgement', async () => {
    const fetchMock = respond(200, { id: 'a1', status: 'acknowledged' })
    vi.stubGlobal('fetch', fetchMock)

    await api.acknowledgeAlert('a1', 'seen')

    const [, init] = fetchMock.mock.calls[0]
    expect(init.method).toBe('POST')
    expect(JSON.parse(init.body)).toEqual({ note: 'seen' })
  })

  it('encodes SKUs in the path so they cannot break out of the URL', async () => {
    const fetchMock = respond(200, { id: 'p1' })
    vi.stubGlobal('fetch', fetchMock)

    await api.productBySku('A/B?x=1')

    const [url] = fetchMock.mock.calls[0]
    expect(url).toContain('A%2FB%3Fx%3D1')
  })
})