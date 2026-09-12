export interface WordTimestamp {
  word: string;
  start_ms: number;
  end_ms: number;
}

export interface ClipResult {
  podcast_id: string;
  episode_id: string;
  show_name: string;
  episode_name: string;
  clip_start_ms: number;
  clip_end_ms: number;
  score: number;
  highlight: string;
  word_timestamps: WordTimestamp[];
  speakers: number[];
  audio_link: string | null;
}

export interface SearchResponse {
  query: string;
  total: number;
  clips: ClipResult[];
  took_ms: number;
}
