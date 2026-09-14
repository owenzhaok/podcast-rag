import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import QuestionPanel from '../src/components/QuestionPanel';
import AnswerPanel from '../src/components/AnswerPanel';
import App from '../src/App';
import type { AskResponse } from '../src/types';

const response: AskResponse = {
  question: 'What is discussed?', status: 'answered', reason: null,
  answer: { paragraphs: [
    { text: 'Learning identifies patterns.', source_ids: ['S1', 'S2'] },
    { text: 'The speakers describe research.', source_ids: ['S2'] },
  ] },
  sources: [1, 2].map((i) => ({ source_id: `S${i}`, chunk_id: `internal-${i}`,
    podcast_id: 'show', episode_id: `episode${i}`, show_name: `Show ${i}`, episode_name: `Episode ${i}`,
    clip_index: i, clip_start_ms: 60000, clip_end_ms: 180000, excerpt: `Full passage ${i}`,
    audio_link: null, speakers: [], metadata_available: true })),
  retrieval_mode: 'bm25', degraded: false, cached: false, took_ms: 100,
};
const fetchMock = vi.fn();
const ok = (body: unknown) => ({ ok: true, json: async () => body });
beforeEach(() => { fetchMock.mockReset(); vi.stubGlobal('fetch', fetchMock); });
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

async function submit(question = 'What is discussed?') {
  await userEvent.type(screen.getByLabelText('Ask a question'), question);
  await userEvent.click(screen.getByRole('button', { name: 'Ask question' }));
}

describe('Ask UI', () => {
  it('renders input and prevents empty or whitespace submission', async () => {
    render(<QuestionPanel />);
    expect(screen.getByRole('button', { name: 'Ask question' })).toHaveProperty('disabled', true);
    await userEvent.type(screen.getByLabelText('Ask a question'), '   ');
    fireEvent.submit(screen.getByLabelText('Ask a question').closest('form')!);
    expect(fetchMock).not.toHaveBeenCalled();
    expect(screen.getByLabelText('Ask a question')).toHaveProperty('maxLength', 2000);
  });

  it('posts only to the backend and renders paragraphs, linked citations and full metadata', async () => {
    fetchMock.mockResolvedValue(ok(response));
    render(<QuestionPanel />);
    await submit('  What is discussed?  ');
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, options] = fetchMock.mock.calls[0];
    expect(url).toBe('/ask');
    expect(options.method).toBe('POST');
    expect(options.headers).toEqual({ 'Content-Type': 'application/json' });
    expect(JSON.parse(options.body)).toEqual({ question: 'What is discussed?' });
    expect(options.signal).toBeInstanceOf(AbortSignal);
    expect(await screen.findByText('Learning identifies patterns.')).toBeDefined();
    expect(screen.getByText('The speakers describe research.')).toBeDefined();
    expect(screen.getAllByRole('link', { name: 'View source S2' })).toHaveLength(2);
    const citation = screen.getByRole('link', { name: 'View source S1' });
    expect(citation.getAttribute('href')).toBe('#ask-source-S1');
    expect(document.getElementById('ask-source-S1')).not.toBeNull();
    expect(screen.getByText('[S1] Show 1')).toBeDefined();
    expect(screen.getByText('Episode 1')).toBeDefined();
    expect(screen.getAllByText('1:00 – 3:00')).toHaveLength(2);
    expect(screen.getByText('Full passage 1')).toBeDefined();
    expect(screen.queryByText('internal-1')).toBeNull();
  });

  it.each([
    ['insufficient_context', 'The available podcast evidence was not sufficient'],
    ['generation_unavailable', 'Answer generation is temporarily unavailable'],
    ['invalid_generation', 'A valid grounded answer could not be produced'],
    ['disabled', 'Question answering is currently disabled'],
  ] as const)('handles %s without exposing internal reasons', async (status, message) => {
    fetchMock.mockResolvedValue(ok({ ...response, status, answer: null, reason: 'private-provider-diagnostic' }));
    render(<QuestionPanel />);
    await submit();
    expect(await screen.findByText(new RegExp(message))).toBeDefined();
    expect(screen.getByText('Full passage 1')).toBeDefined();
    expect(screen.queryByText('private-provider-diagnostic')).toBeNull();
  });

  it.each(['network', 'http', 'json'])('handles %s failure with a safe message', async (failure) => {
    if (failure === 'network') fetchMock.mockRejectedValue(new Error('private-network-diagnostic'));
    else fetchMock.mockResolvedValue({ ok: failure !== 'http', json: async () => { throw Error('private-json-diagnostic'); } });
    render(<QuestionPanel />);
    await submit();
    expect(await screen.findByRole('alert')).toHaveProperty('textContent', 'Question answering could not be reached. Please try again.');
    expect(screen.queryByText(/private-.*-diagnostic/)).toBeNull();
    expect(screen.getByRole('button', { name: 'Ask question' })).toHaveProperty('disabled', false);
  });

  it('blocks duplicates and prevents cancelled responses overwriting the latest answer', async () => {
    let finishOld!: (value: unknown) => void;
    fetchMock.mockImplementationOnce(() => new Promise((resolve) => { finishOld = resolve; }));
    render(<QuestionPanel />);
    await submit();
    expect(screen.getByRole('status').textContent).toContain('Finding evidence');
    expect(screen.getByRole('button', { name: 'Finding an answer…' })).toHaveProperty('disabled', true);
    fireEvent.submit(screen.getByLabelText('Ask a question').closest('form')!);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const signal = fetchMock.mock.calls[0][1].signal;
    await userEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(signal.aborted).toBe(true);
    fetchMock.mockResolvedValueOnce(ok({ ...response, question: 'New question' }));
    await userEvent.clear(screen.getByLabelText('Ask a question'));
    await submit('New question');
    expect(await screen.findByText('Question: New question')).toBeDefined();
    await act(async () => { finishOld(ok({ ...response, question: 'Old stale question' })); });
    expect(screen.queryByText('Question: Old stale question')).toBeNull();
    expect(screen.getByText('Question: New question')).toBeDefined();
  });

  it('aborts on unmount', async () => {
    fetchMock.mockImplementation(() => new Promise(() => {}));
    const { unmount } = render(<QuestionPanel />);
    await submit();
    const signal = fetchMock.mock.calls[0][1].signal;
    unmount();
    expect(signal.aborted).toBe(true);
  });

  it('renders evidence as text and never creates unknown citation links', () => {
    const unsafe = '<img src=x onerror=alert(1)>';
    const { container } = render(<AnswerPanel response={{ ...response,
      answer: { paragraphs: [{ text: unsafe, source_ids: ['S9'] }] },
      sources: [{ ...response.sources[0], excerpt: unsafe, metadata_available: false }],
    }} />);
    expect(container.querySelector('img')).toBeNull();
    expect(screen.getAllByText(unsafe)).toHaveLength(2);
    expect(screen.queryByRole('link')).toBeNull();
    expect(screen.getByText('Some episode details are unavailable.')).toBeDefined();
  });

  it('preserves search and pagination across Ask navigation', async () => {
    const clip = { podcast_id: 'show', episode_id: 'one', show_name: 'Legacy show', episode_name: 'First episode',
      clip_start_ms: 0, clip_end_ms: 120000, score: 1, highlight: 'Search passage', word_timestamps: [], speakers: [], audio_link: null };
    fetchMock.mockResolvedValueOnce(ok({ query: 'AI', total: 2, clips: [clip], took_ms: 1 }));
    render(<App />);
    await userEvent.type(screen.getByPlaceholderText('Search podcasts...'), 'AI{Enter}');
    expect(await screen.findByText('First episode')).toBeDefined();
    expect(fetchMock.mock.calls[0][0]).toBe('/search?q=AI&clip_minutes=2&from=0&size=10');
    await userEvent.click(screen.getByRole('button', { name: 'Ask' }));
    expect(screen.getByLabelText('Ask a question')).toBeDefined();
    await userEvent.click(screen.getByRole('button', { name: 'Search' }));
    expect(screen.getByText('First episode')).toBeDefined();
    fetchMock.mockResolvedValueOnce(ok({ query: 'AI', total: 2, clips: [{ ...clip, episode_id: 'two', episode_name: 'Second episode' }], took_ms: 1 }));
    await userEvent.click(screen.getByRole('button', { name: 'Load more' }));
    expect(await screen.findByText('Second episode')).toBeDefined();
    expect(screen.getByText('First episode')).toBeDefined();
    expect(fetchMock.mock.calls[1][0]).toBe('/search?q=AI&clip_minutes=2&from=10&size=10');
    expect(screen.queryByRole('button', { name: 'Load more' })).toBeNull();
  });

  it('contains no provider credentials or direct provider requests in browser source', () => {
    const files = import.meta.glob('../src/**/*.{ts,tsx}', { query: '?raw', import: 'default', eager: true });
    for (const content of Object.values(files)) {
      expect(content).not.toMatch(/VITE_\w*(?:KEY|SECRET)|RAG_LLM_API_KEY|api\.groq\.com|gsk_[A-Za-z0-9]{20,}/);
    }
  });
});
