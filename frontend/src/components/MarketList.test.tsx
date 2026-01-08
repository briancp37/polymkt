import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { MarketList } from './MarketList'
import type { MarketSelection } from './MarketList'
import type { MarketSearchResult } from '../types'

const createMockMarket = (id: string, overrides?: Partial<MarketSearchResult>): MarketSearchResult => ({
  id,
  question: `Market ${id}`,
  relevance_score: 0.9 - parseInt(id.replace('market', '')) * 0.1,
  category: 'politics',
  closed_time: '2024-12-31T00:00:00Z',
  event_id: 'event1',
  tags: ['election', 'politics'],
  snippet: `Snippet for market ${id}`,
  bm25_score: undefined,
  semantic_score: undefined,
  ...overrides,
})

describe('MarketList', () => {
  describe('rendering', () => {
    it('renders empty state when no markets', () => {
      render(
        <MarketList
          markets={[]}
          selection={{}}
          onSelectionChange={() => {}}
        />
      )
      expect(screen.getByText('No markets found. Try a different search term.')).toBeInTheDocument()
    })

    it('renders markets with checkboxes', () => {
      const markets = [createMockMarket('market1'), createMockMarket('market2')]
      render(
        <MarketList
          markets={markets}
          selection={{ market1: true, market2: true }}
          onSelectionChange={() => {}}
        />
      )
      expect(screen.getByText('Market market1')).toBeInTheDocument()
      expect(screen.getByText('Market market2')).toBeInTheDocument()
    })

    it('shows selected count', () => {
      const markets = [createMockMarket('market1'), createMockMarket('market2')]
      render(
        <MarketList
          markets={markets}
          selection={{ market1: true, market2: false }}
          onSelectionChange={() => {}}
          totalCount={10}
        />
      )
      expect(screen.getByText('1 of 10 selected')).toBeInTheDocument()
    })
  })

  describe('selection', () => {
    it('calls onSelectionChange when market is clicked', () => {
      const onSelectionChange = vi.fn()
      const markets = [createMockMarket('market1')]
      render(
        <MarketList
          markets={markets}
          selection={{ market1: true }}
          onSelectionChange={onSelectionChange}
        />
      )

      fireEvent.click(screen.getByText('Market market1'))
      expect(onSelectionChange).toHaveBeenCalledWith({ market1: false })
    })

    it('toggles selection from selected to unselected', () => {
      const onSelectionChange = vi.fn()
      const markets = [createMockMarket('market1')]
      render(
        <MarketList
          markets={markets}
          selection={{ market1: true }}
          onSelectionChange={onSelectionChange}
        />
      )

      fireEvent.click(screen.getByText('Market market1'))
      expect(onSelectionChange).toHaveBeenCalledWith({ market1: false })
    })

    it('toggles selection from unselected to selected', () => {
      const onSelectionChange = vi.fn()
      const markets = [createMockMarket('market1')]
      render(
        <MarketList
          markets={markets}
          selection={{ market1: false }}
          onSelectionChange={onSelectionChange}
        />
      )

      fireEvent.click(screen.getByText('Market market1'))
      expect(onSelectionChange).toHaveBeenCalledWith({ market1: true })
    })

    it('uses default selection (true) for markets not in selection map', () => {
      const markets = [createMockMarket('market1')]
      const { container } = render(
        <MarketList
          markets={markets}
          selection={{}} // Empty selection - should default to true
          onSelectionChange={() => {}}
        />
      )

      // The market row should have blue background (selected)
      const marketRow = container.querySelector('[class*="bg-blue-50"]')
      expect(marketRow).toBeInTheDocument()
    })
  })

  describe('select all / deselect all', () => {
    it('selects all markets when clicking Select All', () => {
      const onSelectionChange = vi.fn()
      const markets = [createMockMarket('market1'), createMockMarket('market2')]
      render(
        <MarketList
          markets={markets}
          selection={{ market1: false, market2: false }}
          onSelectionChange={onSelectionChange}
        />
      )

      fireEvent.click(screen.getByText('Select All'))
      expect(onSelectionChange).toHaveBeenCalledWith({ market1: true, market2: true })
    })

    it('deselects all markets when all are selected', () => {
      const onSelectionChange = vi.fn()
      const markets = [createMockMarket('market1'), createMockMarket('market2')]
      render(
        <MarketList
          markets={markets}
          selection={{ market1: true, market2: true }}
          onSelectionChange={onSelectionChange}
        />
      )

      fireEvent.click(screen.getByText('Deselect All'))
      expect(onSelectionChange).toHaveBeenCalledWith({ market1: false, market2: false })
    })
  })

  describe('load more', () => {
    it('shows Load More button when hasMore is true', () => {
      const markets = [createMockMarket('market1')]
      render(
        <MarketList
          markets={markets}
          selection={{ market1: true }}
          onSelectionChange={() => {}}
          hasMore={true}
          onLoadMore={() => {}}
        />
      )
      expect(screen.getByText('Load More')).toBeInTheDocument()
    })

    it('hides Load More button when hasMore is false', () => {
      const markets = [createMockMarket('market1')]
      render(
        <MarketList
          markets={markets}
          selection={{ market1: true }}
          onSelectionChange={() => {}}
          hasMore={false}
          onLoadMore={() => {}}
        />
      )
      expect(screen.queryByText('Load More')).not.toBeInTheDocument()
    })

    it('calls onLoadMore when Load More button is clicked', () => {
      const onLoadMore = vi.fn()
      const markets = [createMockMarket('market1')]
      render(
        <MarketList
          markets={markets}
          selection={{ market1: true }}
          onSelectionChange={() => {}}
          hasMore={true}
          onLoadMore={onLoadMore}
        />
      )

      fireEvent.click(screen.getByText('Load More'))
      expect(onLoadMore).toHaveBeenCalled()
    })

    it('shows loading state while loading more', () => {
      const markets = [createMockMarket('market1')]
      render(
        <MarketList
          markets={markets}
          selection={{ market1: true }}
          onSelectionChange={() => {}}
          hasMore={true}
          onLoadMore={() => {}}
          isLoadingMore={true}
        />
      )
      expect(screen.getByText('Loading...')).toBeInTheDocument()
    })
  })

  describe('sorting', () => {
    it('sorts by relevance by default (descending)', () => {
      const markets = [
        createMockMarket('market1', { relevance_score: 0.5 }),
        createMockMarket('market2', { relevance_score: 0.9 }),
        createMockMarket('market3', { relevance_score: 0.7 }),
      ]
      const { container } = render(
        <MarketList
          markets={markets}
          selection={{ market1: true, market2: true, market3: true }}
          onSelectionChange={() => {}}
        />
      )

      // Get all market headings in order
      const headings = container.querySelectorAll('h4')
      expect(headings[0].textContent).toBe('Market market2')
      expect(headings[1].textContent).toBe('Market market3')
      expect(headings[2].textContent).toBe('Market market1')
    })

    it('toggles sort direction on repeated click', () => {
      const markets = [
        createMockMarket('market1', { relevance_score: 0.5 }),
        createMockMarket('market2', { relevance_score: 0.9 }),
      ]
      const { container } = render(
        <MarketList
          markets={markets}
          selection={{ market1: true, market2: true }}
          onSelectionChange={() => {}}
        />
      )

      // Click relevance to toggle direction
      fireEvent.click(screen.getByText('Relevance'))

      const headings = container.querySelectorAll('h4')
      expect(headings[0].textContent).toBe('Market market1')
      expect(headings[1].textContent).toBe('Market market2')
    })
  })
})

describe('Selection persistence across pages', () => {
  it('preserves selections when new markets are added', () => {
    // Simulating the behavior of CreateDatasetPage
    // This tests the pattern used in handleLoadMore

    // Page 1 markets - used to initialize selection
    const _page1Markets = [createMockMarket('market1'), createMockMarket('market2')]
    void _page1Markets // Acknowledge usage for documentation purposes
    const page2Markets = [createMockMarket('market3'), createMockMarket('market4')]

    // Initial selection from page 1
    let selection: MarketSelection = { market1: true, market2: false }

    // User modifies selection on page 1
    selection = { ...selection, market1: false }
    expect(selection.market1).toBe(false)
    expect(selection.market2).toBe(false)

    // Simulate load more - add new markets while preserving existing selections
    // This is the pattern from CreateDatasetPage lines 91-98
    const updated = { ...selection }
    page2Markets.forEach((m) => {
      if (!(m.id in updated)) {
        updated[m.id] = true // New markets default to selected
      }
    })
    selection = updated

    // Verify: page 1 selections are preserved
    expect(selection.market1).toBe(false) // User unselected this
    expect(selection.market2).toBe(false) // User unselected this

    // Verify: page 2 markets are selected by default
    expect(selection.market3).toBe(true)
    expect(selection.market4).toBe(true)
  })

  it('does not override user selections when loading more', () => {
    // Simulating loading the same market_id twice (edge case)
    const duplicateMarkets = [createMockMarket('market1')]

    let selection: MarketSelection = { market1: false } // User explicitly unselected

    // Try to add the same market again (as if API returned duplicate)
    const updated = { ...selection }
    duplicateMarkets.forEach((m) => {
      if (!(m.id in updated)) {
        updated[m.id] = true
      }
    })
    selection = updated

    // The user's choice should be preserved
    expect(selection.market1).toBe(false)
  })

  it('correctly counts selected across all pages', () => {
    const allMarkets = [
      createMockMarket('market1'),
      createMockMarket('market2'),
      createMockMarket('market3'),
      createMockMarket('market4'),
    ]

    const selection: MarketSelection = {
      market1: true,
      market2: false,
      market3: true,
      market4: true,
    }

    // Count should be based on the selection object, not the rendered markets
    const selectedCount = Object.values(selection).filter(Boolean).length
    expect(selectedCount).toBe(3)

    // Verify this matches what MarketList computes
    render(
      <MarketList
        markets={allMarkets}
        selection={selection}
        onSelectionChange={() => {}}
        totalCount={100} // Total across all pages
      />
    )

    expect(screen.getByText('3 of 100 selected')).toBeInTheDocument()
  })
})
