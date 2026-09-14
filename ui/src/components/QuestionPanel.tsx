import { useEffect, useRef, useState } from 'react';
import { ask } from '../api/client';
import type { AskResponse } from '../types';
import AnswerPanel from './AnswerPanel';

export default function QuestionPanel() {
  const [question, setQuestion] = useState('');
  const [response, setResponse] = useState<AskResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const active = useRef<AbortController | null>(null);

  useEffect(() => () => { active.current?.abort(); active.current = null; }, []);

  function cancel() {
    active.current?.abort();
    active.current = null;
    setLoading(false);
  }

  async function submit() {
    const value = question.trim();
    if (!value || value.length > 2000 || active.current) return;
    const controller = new AbortController();
    active.current = controller;
    setLoading(true);
    setError(null);
    setResponse(null);
    try {
      const result = await ask(value, controller.signal);
      if (active.current === controller) setResponse(result);
    } catch {
      if (active.current === controller) setError('Question answering could not be reached. Please try again.');
    } finally {
      if (active.current === controller) {
        active.current = null;
        setLoading(false);
      }
    }
  }

  return (
    <section aria-label="Ask podcasts" className="w-full max-w-2xl">
      <form onSubmit={(event) => { event.preventDefault(); void submit(); }}>
        <label htmlFor="podcast-question" className="block font-semibold mb-2">Ask a question</label>
        <textarea id="podcast-question" value={question} maxLength={2000} rows={3}
          onChange={(event) => setQuestion(event.target.value)} disabled={loading}
          placeholder="What do the speakers say about machine learning?"
          className="w-full border border-gray-300 rounded-md px-4 py-2 focus:ring-2 focus:ring-blue-500 disabled:opacity-50" />
        <div className="flex gap-3 mt-3">
          <button type="submit" disabled={loading || !question.trim()}
            className="bg-blue-600 text-white px-6 py-2 rounded-md hover:bg-blue-700 disabled:opacity-50">
            {loading ? 'Finding an answer…' : 'Ask question'}
          </button>
          {loading && <button type="button" onClick={cancel} className="text-blue-700 px-3 py-2">Cancel</button>}
        </div>
      </form>
      {loading && <p role="status" className="mt-4 text-gray-600">Finding evidence and preparing an answer…</p>}
      {error && <p role="alert" className="mt-4 text-red-600">{error}</p>}
      {response && <AnswerPanel response={response} />}
    </section>
  );
}
