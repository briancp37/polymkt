import type {
  DatasetListResponse,
  Dataset,
  BacktestListResponse,
  Backtest,
  HealthResponse,
} from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${url}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new ApiError(response.status, errorText || response.statusText);
  }

  return response.json();
}

// Health
export async function getHealth(): Promise<HealthResponse> {
  return fetchJson<HealthResponse>('/health');
}

// Datasets
export async function listDatasets(
  limit = 50,
  offset = 0
): Promise<DatasetListResponse> {
  return fetchJson<DatasetListResponse>(
    `/api/datasets?limit=${limit}&offset=${offset}`
  );
}

export async function getDataset(id: string): Promise<Dataset> {
  return fetchJson<Dataset>(`/api/datasets/${id}`);
}

export async function createDataset(
  data: Partial<Dataset>
): Promise<Dataset> {
  return fetchJson<Dataset>('/api/datasets', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export async function updateDataset(
  id: string,
  data: Partial<Dataset>
): Promise<Dataset> {
  return fetchJson<Dataset>(`/api/datasets/${id}`, {
    method: 'PUT',
    body: JSON.stringify(data),
  });
}

export async function deleteDataset(id: string): Promise<void> {
  await fetchJson(`/api/datasets/${id}`, {
    method: 'DELETE',
  });
}

// Backtests
export async function listBacktests(
  limit = 50,
  offset = 0,
  datasetId?: string
): Promise<BacktestListResponse> {
  let url = `/api/backtests?limit=${limit}&offset=${offset}`;
  if (datasetId) {
    url += `&dataset_id=${datasetId}`;
  }
  return fetchJson<BacktestListResponse>(url);
}

export async function getBacktest(id: string): Promise<Backtest> {
  return fetchJson<Backtest>(`/api/backtests/${id}`);
}

export async function createBacktest(
  data: Partial<Backtest>
): Promise<Backtest> {
  return fetchJson<Backtest>('/api/backtests', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export async function executeBacktest(id: string): Promise<Backtest> {
  return fetchJson<Backtest>(`/api/backtests/${id}/execute`, {
    method: 'POST',
  });
}

export async function deleteBacktest(id: string): Promise<void> {
  await fetchJson(`/api/backtests/${id}`, {
    method: 'DELETE',
  });
}

export { ApiError };
