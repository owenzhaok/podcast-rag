import json
import os
from uuid import uuid4
from dataclasses import asdict, replace
from unittest.mock import MagicMock

import httpx
import pytest

from api.rag.gemini import GeminiEmbeddings
from api.rag.embeddings import FakeEmbeddings
from api.rag.embedding_input import document_input, episode_titles, DOCUMENT_INPUT_VERSION
from api.rag.vector_index import VectorConfig, vector_mapping, check_index
from api.rag.backfill import backfill
from api.tests.test_vector_preparation import store, hit


CONFIG = VectorConfig(provider="gemini", model="gemini-embedding-2", dimensions=768)
PLACEHOLDER = "test-placeholder-not-a-real-key"


def adapter(handler):
    return GeminiEmbeddings(CONFIG.model, PLACEHOLDER, transport=httpx.MockTransport(handler))


def test_wire_format_batch_and_configured_model(caplog):
    requests = []
    def handler(request):
        requests.append(request)
        assert request.method == "POST"
        assert str(request.url) == 'https://generativelanguage.googleapis.com/v1beta/models/configured-model:embedContent'
        assert request.headers['x-goog-api-key'] == PLACEHOLDER
        body = json.loads(request.content)
        assert body == {"content": {"parts": [{"text": f"document {len(requests)}"}]}, "outputDimensionality": 768}
        assert PLACEHOLDER not in request.url.query.decode() + request.content.decode()
        assert request.extensions['timeout']['read'] == 20
        return httpx.Response(200, json={"embedding": {"values": [0.1] * 768}})
    provider = GeminiEmbeddings('configured-model', PLACEHOLDER, transport=httpx.MockTransport(handler))
    assert provider.embed(['document 1', 'document 2']) == [[0.1] * 768, [0.1] * 768]
    assert len(requests) == 2 and PLACEHOLDER not in caplog.text + repr(provider)


@pytest.mark.parametrize('body', [
    {}, {"embedding": {}}, {"embedding": {"values": [1] * 767}},
    {"embedding": {"values": ['1'] * 768}}, {"embedding": {"values": [True] * 768}},
    {"embedding": {"values": [0] * 768}}, {"embedding": {"values": None}},
])
def test_invalid_responses(body):
    with pytest.raises(ValueError, match='Invalid Gemini'):
        adapter(lambda request: httpx.Response(200, json=body)).embed(['text'])


@pytest.mark.parametrize('raw', [b'not json', b'{"embedding":{"values":[NaN]}}',
    b'{"embedding":{"values":[Infinity]}}', pytest.param(b'x' * 131073, id='oversized')])
def test_invalid_json_nonfinite_and_bounded_response(raw):
    with pytest.raises(ValueError):
        adapter(lambda request: httpx.Response(200, content=raw)).embed(['text'])


@pytest.mark.parametrize('status', [400, 401, 403, 429, 500, 503, 302])
def test_http_errors_never_expose_body_or_key(status, caplog):
    with pytest.raises(ValueError) as exc:
        adapter(lambda request: httpx.Response(status, text=PLACEHOLDER)).embed(['text'])
    assert PLACEHOLDER not in str(exc.value) + caplog.text


@pytest.mark.parametrize('error', [httpx.ReadTimeout(PLACEHOLDER), httpx.ConnectError(PLACEHOLDER)])
def test_transport_errors_safe(error, caplog):
    def handler(request): raise error
    with pytest.raises(ValueError) as exc:
        adapter(handler).embed(['text'])
    assert PLACEHOLDER not in str(exc.value) + caplog.text


def test_configuration_does_not_retain_secret(monkeypatch):
    for name, value in {'PROVIDER': 'gemini', 'MODEL': 'gemini-embedding-2', 'DIMENSIONS': '768', 'API_KEY': PLACEHOLDER}.items():
        monkeypatch.setenv('RAG_EMBEDDING_' + name, value)
    config = VectorConfig.from_env()
    assert PLACEHOLDER not in repr(config) + json.dumps(asdict(config))
    assert isinstance(config.make_provider(), GeminiEmbeddings)
    monkeypatch.delenv('RAG_EMBEDDING_API_KEY')
    with pytest.raises(ValueError, match='not configured'): config.make_provider()
    with pytest.raises(ValueError): replace(config, dimensions=8).validate()


def test_input_format_and_metadata_fallback(monkeypatch):
    doc = {'podcast_id': 'show', 'episode_id': 'episode', 'chunk_text': ' exact transcript\n'}
    assert document_input(doc, '  Episode title  ') == 'title: Episode title | text:  exact transcript\n'
    assert document_input(doc) == 'title: show/episode | text:  exact transcript\n'
    assert document_input(doc, '  ') == document_input(doc)
    import psycopg2
    monkeypatch.setattr('api.rag.embedding_input.psycopg2.connect', MagicMock(side_effect=psycopg2.OperationalError(PLACEHOLDER)))
    assert episode_titles(['episode']) == {}


@pytest.fixture
def gemini_store(store):
    es, docs, source = store
    es.indices.get_mapping.side_effect = lambda index: {index: {'mappings': vector_mapping(CONFIG)}}
    es.count.return_value = {'count': 0}
    return es, docs, source


def test_real_provider_double_canonical_input_skip_and_revision(gemini_store):
    es, docs, source = gemini_store
    provider = MagicMock()
    provider.embed.side_effect = FakeEmbeddings(768, CONFIG.model, 'v1').embed
    backfill(es, CONFIG, provider, 10, progress=lambda _: None, title_lookup=lambda _: {'episode': 'Title'})
    assert provider.embed.call_args.args[0] == ['title: Title | text: ' + source[0]['_source']['clip_text']]
    second = backfill(es, CONFIG, provider, 10, progress=lambda _: None, title_lookup=lambda _: {'episode': 'Title'})
    assert second['written'] == 0 and provider.embed.call_count == 1
    assert next(iter(docs.values()))['embedding_input_version'] == DOCUMENT_INPUT_VERSION
    changed = backfill(es, replace(CONFIG, revision='v2'), provider, 10, progress=lambda _: None,
                       title_lookup=lambda _: {'episode': 'Title'})
    assert changed['written'] == 1 and provider.embed.call_count == 2
    changed = backfill(es, CONFIG, provider, 10, progress=lambda _: None, title_lookup=lambda _: {'episode': 'Changed title'})
    assert changed['written'] == 1


def test_foreign_provider_or_model_stops_before_embedding(gemini_store):
    es, docs, source = gemini_store
    es.count.return_value = {'count': 1}
    provider = MagicMock()
    with pytest.raises(ValueError, match='new versioned index'):
        backfill(es, CONFIG, provider, 10)
    provider.embed.assert_not_called()
    assert not docs
    es.indices.delete.assert_not_called()


def test_second_embedding_failure_writes_no_partial_batch(gemini_store):
    es, docs, source = gemini_store
    source[:] = [hit(episode='one'), hit(episode='two')]
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={'embedding': {'values': [0.1] * 768}}) if len(calls) == 1 else httpx.Response(429)
    with pytest.raises(ValueError, match='provider failed'):
        backfill(es, CONFIG, adapter(handler), 10)
    assert not docs


def test_input_length_rejected_without_call():
    transport = MagicMock()
    with pytest.raises(ValueError, match='8192'):
        GeminiEmbeddings(CONFIG.model, PLACEHOLDER, transport=transport).embed(['x' * 8193])
    transport.handle_request.assert_not_called()


@pytest.mark.skipif(os.environ.get('RAG_INTEGRATION_TESTS') != '1', reason='Opt-in Docker ES test')
def test_gemini_mocktransport_real_es():
    from elasticsearch import Elasticsearch
    from api.rag.vector_index import create_index
    from ingest.es_client import INDEX_MAPPING
    suffix = uuid4().hex
    config = replace(CONFIG, vector_index='rag-gemini-vector-' + suffix, source_index='rag-gemini-source-' + suffix)
    created, calls = [], []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={'embedding': {'values': [0.1] * 768}})
    with Elasticsearch(os.environ.get('ES_HOST', 'http://localhost:9200')) as es:
        try:
            es.indices.create(index=config.source_index, body=INDEX_MAPPING)
            created.append(config.source_index)
            es.index(index=config.source_index, id='one', document=hit()['_source'], refresh=True)
            create_index(es, config)
            created.append(config.vector_index)
            first = backfill(es, config, adapter(handler), 10, progress=lambda _: None)
            second = backfill(es, config, adapter(handler), 10, progress=lambda _: None)
            assert first['written'] == 1 and second['written'] == 0 and len(calls) == 1
            result = es.search(index=config.vector_index)['hits']['hits'][0]
            assert len(result['_source']['embedding']) == 768
            assert result['_source']['embedding_provider'] == 'gemini'
            assert 'title: show/episode | text:' in calls[0].content.decode()
            assert es.count(index=config.source_index)['count'] == 1
            with pytest.raises(ValueError, match='other model'):
                check_index(es, replace(config, model='different-model'))
            # A foreign vector anywhere in the index must stop preflight.
            foreign = {**result['_source'], 'embedding_provider': 'fake'}
            es.index(index=config.vector_index, id='foreign', document=foreign, refresh=True)
            with pytest.raises(ValueError, match='fake/other model'):
                check_index(es, config)
        finally:
            for index in reversed(created): es.indices.delete(index=index)
