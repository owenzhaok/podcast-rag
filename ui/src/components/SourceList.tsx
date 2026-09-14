import type { RagSource } from '../types';

export function sourceAnchor(sourceId: string) {
  return `ask-source-${encodeURIComponent(sourceId)}`;
}

function timestamp(ms: number) {
  const seconds = Math.floor(ms / 1000);
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
}

export default function SourceList({ sources }: { sources: RagSource[] }) {
  if (!sources.length) return null;
  return (
    <section aria-labelledby="ask-sources-heading" className="space-y-4 mt-8">
      <h2 id="ask-sources-heading" className="text-xl font-semibold">Sources</h2>
      {sources.map((source) => (
        <article key={source.source_id} id={sourceAnchor(source.source_id)} tabIndex={-1}
          className="bg-white border border-gray-200 rounded-lg p-4 scroll-mt-4">
          <h3 className="font-semibold">[{source.source_id}] {source.show_name || 'Unknown show'}</h3>
          <p className="text-sm text-gray-600">{source.episode_name || 'Unknown episode'}</p>
          <p className="text-sm text-gray-500 my-2">
            {timestamp(source.clip_start_ms)} – {timestamp(source.clip_end_ms)}
          </p>
          {!source.metadata_available && <p className="text-sm text-gray-500">Some episode details are unavailable.</p>}
          <p className="text-gray-700 whitespace-pre-wrap break-words">{source.excerpt}</p>
        </article>
      ))}
    </section>
  );
}
