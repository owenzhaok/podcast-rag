import { useState } from 'react';
import DurationSelector from './DurationSelector';

interface SearchBarProps {
  onSearch: (q: string, clipMinutes: number) => void;
  loading: boolean;
}

export default function SearchBar({ onSearch, loading }: SearchBarProps) {
  const [query, setQuery] = useState('');
  const [clipMinutes, setClipMinutes] = useState(2);

  const handleSubmit = () => {
    if (query.trim()) {
      onSearch(query.trim(), clipMinutes);
    }
  };

  return (
    <div className="flex gap-2 w-full max-w-2xl">
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={(e) => e.key === 'Enter' && handleSubmit()}
        placeholder="Search podcasts..."
        disabled={loading}
        className="flex-1 border border-gray-300 rounded-md px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
      />
      <DurationSelector value={clipMinutes} onChange={setClipMinutes} />
      <button
        onClick={handleSubmit}
        disabled={loading}
        className="bg-blue-600 text-white px-6 py-2 rounded-md hover:bg-blue-700 disabled:opacity-50"
      >
        {loading ? 'Searching...' : 'Search'}
      </button>
    </div>
  );
}
