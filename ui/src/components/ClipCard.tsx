import { ClipResult } from '../types';

interface ClipCardProps {
  clip: ClipResult;
}

function formatMs(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${String(seconds).padStart(2, '0')}`;
}

export default function ClipCard({ clip }: ClipCardProps) {
  return (
    <div className="border border-gray-200 rounded-lg p-4 shadow-sm hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between mb-2">
        <div>
          <h3 className="font-semibold text-gray-900">{clip.show_name}</h3>
          <p className="text-sm text-gray-600">{clip.episode_name}</p>
        </div>
        {clip.speakers.length > 1 && (
          <span className="bg-purple-100 text-purple-700 text-xs px-2 py-1 rounded-full">
            {clip.speakers.length} speakers
          </span>
        )}
      </div>
      <p className="text-xs text-gray-400 mb-2">
        {formatMs(clip.clip_start_ms)} &ndash; {formatMs(clip.clip_end_ms)}
      </p>
      <div
        className="text-sm text-gray-700 leading-relaxed"
        dangerouslySetInnerHTML={{ __html: clip.highlight }}
      />
      {clip.audio_link && (
        <a
          href={clip.audio_link}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-block mt-3 text-sm text-blue-600 hover:underline"
        >
          Listen
        </a>
      )}
    </div>
  );
}
