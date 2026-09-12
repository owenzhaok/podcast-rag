import { SearchResponse } from '../types';
import ClipCard from './ClipCard';

interface ResultsListProps {
  results: SearchResponse | null;
  loading: boolean;
  error: string | null;
  onLoadMore: () => void;
}

export default function ResultsList({ results, loading, error, onLoadMore }: ResultsListProps) {
  if (loading && !results) {
    return (
      <div className="space-y-4 w-full max-w-2xl">
        {[1, 2, 3].map((i) => (
          <div key={i} data-testid="skeleton" className="h-32 bg-gray-100 rounded-lg animate-pulse" />
        ))}
      </div>
    );
  }

  if (error) {
    return <p className="text-red-600">{error}</p>;
  }

  if (results && results.total === 0) {
    return <p className="text-gray-500">No results found.</p>;
  }

  if (!results) {
    return null;
  }

  const hasMore = results.clips.length < results.total;

  return (
    <div className="space-y-4 w-full max-w-2xl">
      {results.clips.map((clip, i) => (
        <ClipCard key={`${clip.episode_id}-${clip.clip_start_ms}-${i}`} clip={clip} />
      ))}
      {hasMore && (
        <button
          onClick={onLoadMore}
          disabled={loading}
          className="w-full py-2 text-blue-600 hover:text-blue-800 disabled:opacity-50"
        >
          {loading ? 'Loading...' : 'Load more'}
        </button>
      )}
    </div>
  );
}
