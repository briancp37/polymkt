import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Layout } from './components';
import {
  HomePage,
  DatasetsPage,
  DatasetDetailPage,
  CreateDatasetPage,
  BacktestsPage,
  BacktestDetailPage,
} from './pages';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60, // 1 minute
      refetchOnWindowFocus: false,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<HomePage />} />
            <Route path="datasets" element={<DatasetsPage />} />
            <Route path="datasets/new" element={<CreateDatasetPage />} />
            <Route path="datasets/:id" element={<DatasetDetailPage />} />
            <Route path="backtests" element={<BacktestsPage />} />
            <Route path="backtests/:id" element={<BacktestDetailPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;
