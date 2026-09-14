import type { AskResponse } from '../types';
import SourceList, { sourceAnchor } from './SourceList';

const messages = {
  answered: 'Answer',
  insufficient_context: 'The available podcast evidence was not sufficient to answer this question.',
  generation_unavailable: 'Answer generation is temporarily unavailable. You can still explore any sources below.',
  invalid_generation: 'A valid grounded answer could not be produced. You can still explore any sources below.',
  disabled: 'Question answering is currently disabled.',
};

export default function AnswerPanel({ response }: { response: AskResponse }) {
  const sourceIds = new Set(response.sources.map((source) => source.source_id));
  return (
    <div className="mt-6">
      <p className="text-sm text-gray-500 mb-3">Question: {response.question}</p>
      <h2 className="text-lg font-semibold" role="status">{messages[response.status]}</h2>
      {response.status === 'answered' && response.answer?.paragraphs.map((paragraph, index) => (
        <p key={index} className="mt-4 leading-relaxed whitespace-pre-wrap break-words">
          {paragraph.text}{' '}
          {paragraph.source_ids.filter((id) => sourceIds.has(id)).map((id, citationIndex) => (
            <a key={`${id}-${citationIndex}`} href={`#${sourceAnchor(id)}`}
              className="text-blue-700 underline mr-2" aria-label={`View source ${id}`}>[{id}]</a>
          ))}
        </p>
      ))}
      <SourceList sources={response.sources} />
    </div>
  );
}
