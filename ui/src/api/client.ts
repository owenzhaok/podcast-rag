import { SearchResponse } from '../types';

export class SearchError extends Error {
  constructor(message: string, public status: number) {
    super(message);
  }
}

export async function search(
  q: string,
  clipMinutes: number,
  from: number,
  size: number,
): Promise<SearchResponse> {
  const params = new URLSearchParams({
    q,
    clip_minutes: String(clipMinutes),
    from: String(from),
    size: String(size),
  });
  const resp = await fetch(`/search?${params}`);
  if (!resp.ok) {
    throw new SearchError(`Search failed: ${resp.statusText}`, resp.status);
  }
  return resp.json();
}
