import { useState, useCallback } from 'react';
import SearchBar from './components/SearchBar';
import ResultsList from './components/ResultsList';
import QuestionPanel from './components/QuestionPanel';
import { search as apiSearch } from './api/client';
import { SearchResponse } from './types';

export default function App() {
  const [mode, setMode] = useState<'search' | 'ask'>('search');
  const [query, setQuery] = useState('');
  const [clipMinutes, setClipMinutes] = useState(2);
  const [results, setResults] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [from, setFrom] = useState(0);
  const SIZE = 10;

  const doSearch = useCallback(async (q: string, mins: number, offset: number, append: boolean) => {
    setLoading(true);
    setError(null);
    try {
      const resp = await apiSearch(q, mins, offset, SIZE);
      if (append) {
        setResults((prev) => prev ? { ...resp, clips: [...prev.clips, ...resp.clips] } : resp);
      } else {
        setResults(resp);
      }
    } catch (e: any) {
      setError(e.message || 'Search failed');
    } finally {
      setLoading(false);
    }
  }, []);

  const handleSearch = (q: string, mins: number) => {
    setQuery(q);
    setClipMinutes(mins);
    setFrom(0);
    setResults(null);
    doSearch(q, mins, 0, false);
  };

  const handleLoadMore = () => {
    const nextFrom = from + SIZE;
    setFrom(nextFrom);
    doSearch(query, clipMinutes, nextFrom, true);
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center px-4 py-12">
      <h1 className="text-3xl font-bold text-gray-900 mb-8">Podcast Search</h1>
      <nav aria-label="Podcast tools" className="flex gap-2 mb-6">
        {(['search', 'ask'] as const).map((tab) => (
          <button key={tab} aria-pressed={mode === tab} onClick={() => setMode(tab)}
            className={`px-5 py-2 rounded-md ${mode === tab ? 'bg-blue-600 text-white' : 'bg-white text-blue-700 border border-gray-300'}`}>
            {tab === 'search' ? 'Search' : 'Ask'}
          </button>
        ))}
      </nav>
      <div hidden={mode !== 'search'} className={`w-full flex-col items-center ${mode === 'search' ? 'flex' : 'hidden'}`}>
        <SearchBar onSearch={handleSearch} loading={loading} />
        <div className="mt-8 w-full flex justify-center">
          <ResultsList results={results} loading={loading} error={error} onLoadMore={handleLoadMore} />
        </div>
      </div>
      {mode === 'ask' && <QuestionPanel />}
    </div>
  );
}
