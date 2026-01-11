import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { ConfirmationModal, DatasetConfirmationContent } from './ConfirmationModal'

describe('ConfirmationModal', () => {
  const defaultProps = {
    isOpen: true,
    onClose: vi.fn(),
    onConfirm: vi.fn(),
    title: 'Test Modal',
    children: <p>Modal content</p>,
  }

  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('rendering', () => {
    it('renders when isOpen is true', () => {
      render(<ConfirmationModal {...defaultProps} />)
      expect(screen.getByText('Test Modal')).toBeInTheDocument()
      expect(screen.getByText('Modal content')).toBeInTheDocument()
    })

    it('does not render when isOpen is false', () => {
      render(<ConfirmationModal {...defaultProps} isOpen={false} />)
      expect(screen.queryByText('Test Modal')).not.toBeInTheDocument()
    })

    it('renders default button text', () => {
      render(<ConfirmationModal {...defaultProps} />)
      expect(screen.getByText('Confirm')).toBeInTheDocument()
      expect(screen.getByText('Cancel')).toBeInTheDocument()
    })

    it('renders custom button text', () => {
      render(
        <ConfirmationModal
          {...defaultProps}
          confirmText="Create Dataset"
          cancelText="Go Back"
        />
      )
      expect(screen.getByText('Create Dataset')).toBeInTheDocument()
      expect(screen.getByText('Go Back')).toBeInTheDocument()
    })
  })

  describe('interactions', () => {
    it('calls onClose when cancel button is clicked', () => {
      const onClose = vi.fn()
      render(<ConfirmationModal {...defaultProps} onClose={onClose} />)
      fireEvent.click(screen.getByText('Cancel'))
      expect(onClose).toHaveBeenCalled()
    })

    it('calls onConfirm when confirm button is clicked', () => {
      const onConfirm = vi.fn()
      render(<ConfirmationModal {...defaultProps} onConfirm={onConfirm} />)
      fireEvent.click(screen.getByText('Confirm'))
      expect(onConfirm).toHaveBeenCalled()
    })

    it('calls onClose when X button is clicked', () => {
      const onClose = vi.fn()
      render(<ConfirmationModal {...defaultProps} onClose={onClose} />)
      // X button is the one with the X icon
      const closeButtons = screen.getAllByRole('button')
      const xButton = closeButtons.find(btn => btn.querySelector('svg'))
      if (xButton) {
        fireEvent.click(xButton)
        expect(onClose).toHaveBeenCalled()
      }
    })

    it('calls onClose when backdrop is clicked', () => {
      const onClose = vi.fn()
      const { container } = render(<ConfirmationModal {...defaultProps} onClose={onClose} />)
      const backdrop = container.querySelector('.bg-black')
      if (backdrop) {
        fireEvent.click(backdrop)
        expect(onClose).toHaveBeenCalled()
      }
    })
  })

  describe('loading state', () => {
    it('shows loading text when isLoading is true', () => {
      render(<ConfirmationModal {...defaultProps} isLoading={true} />)
      expect(screen.getByText('Processing...')).toBeInTheDocument()
    })

    it('disables buttons when isLoading is true', () => {
      render(<ConfirmationModal {...defaultProps} isLoading={true} />)
      const buttons = screen.getAllByRole('button')
      const confirmButton = buttons.find(btn => btn.textContent === 'Processing...')
      const cancelButton = buttons.find(btn => btn.textContent === 'Cancel')
      expect(confirmButton).toBeDisabled()
      expect(cancelButton).toBeDisabled()
    })
  })
})

describe('DatasetConfirmationContent', () => {
  const defaultProps = {
    name: 'Test Dataset',
    includedCount: 50,
    excludedCount: 10,
  }

  it('renders dataset name', () => {
    render(<DatasetConfirmationContent {...defaultProps} />)
    expect(screen.getByText('Test Dataset')).toBeInTheDocument()
  })

  it('renders description when provided', () => {
    render(<DatasetConfirmationContent {...defaultProps} description="A test description" />)
    expect(screen.getByText('A test description')).toBeInTheDocument()
  })

  it('does not render description section when not provided', () => {
    render(<DatasetConfirmationContent {...defaultProps} />)
    expect(screen.queryByText('Description')).not.toBeInTheDocument()
  })

  it('renders included and excluded counts', () => {
    render(<DatasetConfirmationContent {...defaultProps} />)
    expect(screen.getByText('50')).toBeInTheDocument()
    expect(screen.getByText('10')).toBeInTheDocument()
    expect(screen.getByText('included')).toBeInTheDocument()
    expect(screen.getByText('excluded')).toBeInTheDocument()
  })

  it('renders search query when provided', () => {
    render(<DatasetConfirmationContent {...defaultProps} searchQuery="election" />)
    expect(screen.getByText(/Search: "election"/)).toBeInTheDocument()
  })

  it('renders category when provided', () => {
    render(<DatasetConfirmationContent {...defaultProps} searchCategory="politics" />)
    expect(screen.getByText(/Category: politics/)).toBeInTheDocument()
  })

  it('renders both search and category when provided', () => {
    render(
      <DatasetConfirmationContent
        {...defaultProps}
        searchQuery="election"
        searchCategory="politics"
      />
    )
    expect(screen.getByText(/Search: "election"/)).toBeInTheDocument()
    expect(screen.getByText(/Category: politics/)).toBeInTheDocument()
  })

  it('does not render selection basis when neither search nor category provided', () => {
    render(<DatasetConfirmationContent {...defaultProps} />)
    expect(screen.queryByText('Selection Basis')).not.toBeInTheDocument()
  })
})

describe('Dataset confirmation flow', () => {
  it('shows confirmation modal with correct content', () => {
    render(
      <ConfirmationModal
        isOpen={true}
        onClose={() => {}}
        onConfirm={() => {}}
        title="Confirm Dataset Creation"
        confirmText="Create Dataset"
      >
        <DatasetConfirmationContent
          name="Election Markets 2024"
          description="All markets related to 2024 elections"
          includedCount={42}
          excludedCount={8}
          searchQuery="election"
          searchCategory="politics"
        />
      </ConfirmationModal>
    )

    // Modal title
    expect(screen.getByText('Confirm Dataset Creation')).toBeInTheDocument()

    // Dataset details
    expect(screen.getByText('Election Markets 2024')).toBeInTheDocument()
    expect(screen.getByText('All markets related to 2024 elections')).toBeInTheDocument()

    // Market counts
    expect(screen.getByText('42')).toBeInTheDocument()
    expect(screen.getByText('8')).toBeInTheDocument()

    // Selection basis
    expect(screen.getByText(/Search: "election"/)).toBeInTheDocument()
    expect(screen.getByText(/Category: politics/)).toBeInTheDocument()

    // Buttons
    expect(screen.getByText('Create Dataset')).toBeInTheDocument()
    expect(screen.getByText('Cancel')).toBeInTheDocument()
  })

  it('does not start save until confirmation is accepted', () => {
    const onConfirm = vi.fn()
    render(
      <ConfirmationModal
        isOpen={true}
        onClose={() => {}}
        onConfirm={onConfirm}
        title="Confirm"
        confirmText="Save"
      >
        <p>Are you sure?</p>
      </ConfirmationModal>
    )

    // Confirm has not been called yet
    expect(onConfirm).not.toHaveBeenCalled()

    // Click confirm
    fireEvent.click(screen.getByText('Save'))
    expect(onConfirm).toHaveBeenCalledTimes(1)
  })
})
