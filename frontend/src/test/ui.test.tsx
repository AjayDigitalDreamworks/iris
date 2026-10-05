import { describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { AsyncContent, Badge, Stat, fmtPct, fmtSeconds } from '../components/ui'

describe('UI primitives', () => {
  it('renders an error state with a working retry', async () => {
    const reload = vi.fn()
    render(
      <AsyncContent
        state={{ data: null, loading: false, error: 'backend unreachable', reload }}
        empty="Nothing here"
      >
        {() => <div>content</div>}
      </AsyncContent>,
    )
    expect(screen.getByRole('alert')).toHaveTextContent('backend unreachable')
    await userEvent.click(screen.getByRole('button', { name: /retry/i }))
    expect(reload).toHaveBeenCalledOnce()
  })

  it('shows the empty state rather than rendering children for an empty collection', () => {
    render(
      <AsyncContent
        state={{ data: [], loading: false, error: null, reload: () => {} }}
        empty="No alerts"
        isEmpty={(rows) => rows.length === 0}
      >
        {() => <div>should not render</div>}
      </AsyncContent>,
    )
    expect(screen.getByText('No alerts')).toBeInTheDocument()
    expect(screen.queryByText('should not render')).not.toBeInTheDocument()
  })

  it('renders children once data has loaded', () => {
    render(
      <AsyncContent state={{ data: { value: 7 }, loading: false, error: null, reload: () => {} }} empty="none">
        {(data) => <span>value {data.value}</span>}
      </AsyncContent>,
    )
    expect(screen.getByText('value 7')).toBeInTheDocument()
  })

  it('labels an unknown stock percentage as unknown, never 0%', () => {
    // -1 is the backend's "capacity not configured" sentinel.
    expect(fmtPct(-1)).toBe('—')
    expect(fmtPct(null)).toBe('—')
    expect(fmtPct(0)).toBe('0.0%')
  })

  it('formats wait estimates for humans', () => {
    expect(fmtSeconds(45)).toBe('45s')
    expect(fmtSeconds(90)).toBe('1m 30s')
    expect(fmtSeconds(null)).toBe('—')
  })

  it('renders a badge with a readable label', () => {
    render(<Badge value="low_stock" />)
    expect(screen.getByText('low stock')).toBeInTheDocument()
  })

  it('shows a stat with its hint', () => {
    render(<Stat label="Entries" value={12} hint="24h window" tone="good" />)
    expect(screen.getByText('Entries')).toBeInTheDocument()
    expect(screen.getByText('12')).toBeInTheDocument()
    expect(screen.getByText('24h window')).toBeInTheDocument()
  })
})