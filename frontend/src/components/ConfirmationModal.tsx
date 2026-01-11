import { X, CheckCircle, AlertCircle } from 'lucide-react';
import { Button } from './Button';

interface ConfirmationModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  title: string;
  children: React.ReactNode;
  confirmText?: string;
  cancelText?: string;
  isLoading?: boolean;
  variant?: 'default' | 'danger';
}

export function ConfirmationModal({
  isOpen,
  onClose,
  onConfirm,
  title,
  children,
  confirmText = 'Confirm',
  cancelText = 'Cancel',
  isLoading = false,
  variant = 'default',
}: ConfirmationModalProps) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black bg-opacity-50 transition-opacity"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="flex min-h-full items-center justify-center p-4">
        <div className="relative bg-white rounded-lg shadow-xl max-w-lg w-full">
          {/* Header */}
          <div className="flex items-center justify-between p-4 border-b border-gray-200">
            <div className="flex items-center gap-2">
              {variant === 'danger' ? (
                <AlertCircle className="w-5 h-5 text-red-500" />
              ) : (
                <CheckCircle className="w-5 h-5 text-blue-500" />
              )}
              <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
            </div>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-500 transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Content */}
          <div className="p-4">{children}</div>

          {/* Footer */}
          <div className="flex items-center justify-end gap-3 p-4 border-t border-gray-200 bg-gray-50 rounded-b-lg">
            <Button variant="secondary" onClick={onClose} disabled={isLoading}>
              {cancelText}
            </Button>
            <Button
              variant={variant === 'danger' ? 'danger' : 'primary'}
              onClick={onConfirm}
              disabled={isLoading}
            >
              {isLoading ? 'Processing...' : confirmText}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

interface DatasetConfirmationContentProps {
  name: string;
  description?: string;
  includedCount: number;
  excludedCount: number;
  searchQuery?: string;
  searchCategory?: string;
}

export function DatasetConfirmationContent({
  name,
  description,
  includedCount,
  excludedCount,
  searchQuery,
  searchCategory,
}: DatasetConfirmationContentProps) {
  return (
    <div className="space-y-4">
      <p className="text-gray-600">
        Please review the dataset details before saving:
      </p>

      <div className="bg-gray-50 rounded-lg p-4 space-y-3">
        {/* Dataset name */}
        <div>
          <span className="text-sm font-medium text-gray-500">Dataset Name</span>
          <p className="text-gray-900 font-medium">{name}</p>
        </div>

        {/* Description */}
        {description && (
          <div>
            <span className="text-sm font-medium text-gray-500">Description</span>
            <p className="text-gray-700 text-sm">{description}</p>
          </div>
        )}

        {/* Selection basis */}
        {(searchQuery || searchCategory) && (
          <div>
            <span className="text-sm font-medium text-gray-500">Selection Basis</span>
            <div className="text-sm text-gray-700">
              {searchQuery && <span>Search: "{searchQuery}"</span>}
              {searchQuery && searchCategory && <span> • </span>}
              {searchCategory && <span>Category: {searchCategory}</span>}
            </div>
          </div>
        )}

        {/* Market counts */}
        <div className="pt-2 border-t border-gray-200">
          <span className="text-sm font-medium text-gray-500">Market Selection</span>
          <div className="flex gap-6 mt-1">
            <div>
              <span className="text-2xl font-bold text-green-600">{includedCount}</span>
              <span className="text-sm text-gray-500 ml-1">included</span>
            </div>
            <div>
              <span className="text-2xl font-bold text-gray-400">{excludedCount}</span>
              <span className="text-sm text-gray-500 ml-1">excluded</span>
            </div>
          </div>
        </div>
      </div>

      <p className="text-sm text-gray-500">
        Click "Create Dataset" to save this dataset with the selected markets.
      </p>
    </div>
  );
}
