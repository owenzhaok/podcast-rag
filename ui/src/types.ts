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

export interface RagSource {
  source_id: string;
  chunk_id: string;
  podcast_id: string;
  episode_id: string;
  show_name: string;
  episode_name: string;
  clip_index: number;
  clip_start_ms: number;
  clip_end_ms: number;
  excerpt: string;
  audio_link: string | null;
  speakers: number[];
  metadata_available: boolean;
}

export interface AnswerParagraph {
  text: string;
  source_ids: string[];
}

export interface AskResponse {
  question: string;
  status: 'answered' | 'insufficient_context' | 'generation_unavailable' | 'invalid_generation' | 'disabled';
  reason: string | null;
  answer: { paragraphs: AnswerParagraph[] } | null;
  sources: RagSource[];
  retrieval_mode: 'bm25' | null;
  degraded: boolean;
  cached: boolean;
  took_ms: number;
}
