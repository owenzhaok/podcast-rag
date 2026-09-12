import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import ResultsList from '../src/components/ResultsList';
import { SearchResponse, ClipResult } from '../src/types';

function makeClip(id: string): ClipResult {
  return {
    podcast_id: 'show_abc',
    episode_id: id,
    show_name: 'Show',
    episode_name: `Episode ${id}`,
    clip_start_ms: 0,
    clip_end_ms: 120000,
    score: 5.0,
    highlight: 'text',
    word_timestamps: [],
    speakers: [1],
    audio_link: null,
  };
}

function makeResponse(total: number, clipCount: number): SearchResponse {
  return {
    query: 'test',
    total,
    clips: Array.from({ length: clipCount }, (_, i) => makeClip(`ep${i}`)),
    took_ms: 10,
  };
}

describe('ResultsList', () => {
  it('shows loading skeleton', () => {
    render(<ResultsList results={null} loading={true} error={null} onLoadMore={vi.fn()} />);
    expect(screen.getAllByTestId('skeleton').length).toBe(3);
  });

  it('shows error message', () => {
    render(<ResultsList results={null} loading={false} error="Network error" onLoadMore={vi.fn()} />);
    expect(screen.getByText('Network error')).toBeDefined();
  });

  it('shows empty state', () => {
    render(<ResultsList results={makeResponse(0, 0)} loading={false} error={null} onLoadMore={vi.fn()} />);
    expect(screen.getByText('No results found.')).toBeDefined();
  });

  it('renders clip cards', () => {
    render(<ResultsList results={makeResponse(3, 3)} loading={false} error={null} onLoadMore={vi.fn()} />);
    expect(screen.getAllByText(/Episode ep/).length).toBe(3);
  });

  it('load more button visible when more results', () => {
    render(<ResultsList results={makeResponse(20, 10)} loading={false} error={null} onLoadMore={vi.fn()} />);
    expect(screen.getByText('Load more')).toBeDefined();
  });

  it('load more button hidden when exhausted', () => {
    render(<ResultsList results={makeResponse(10, 10)} loading={false} error={null} onLoadMore={vi.fn()} />);
    expect(screen.queryByText('Load more')).toBeNull();
  });
});
