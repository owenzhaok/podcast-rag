import pytest
from unittest.mock import AsyncMock
from api.search import build_query, execute_search
from api.models import SearchResponse
from api.cache import CacheClient


class TestBuildQuery:
    def test_build_query_contains_multi_match(self):
        q = build_query("Higgs Boson", 0, 10)
        assert q["query"]["bool"]["must"][0]["multi_match"]["query"] == "Higgs Boson"

    def test_build_query_has_highlight(self):
        q = build_query("test", 0, 10)
        assert "clip_text" in q["highlight"]["fields"]

    def test_build_query_fuzziness_auto(self):
        q = build_query("test", 0, 10)
        assert q["query"]["bool"]["must"][0]["multi_match"]["fuzziness"] == "AUTO"

    def test_build_query_pagination(self):
        q = build_query("test", 20, 5)
        assert q["from"] == 20
        assert q["size"] == 5


class TestExecuteSearch:
    @pytest.fixture
    def mock_es(self):
        es = AsyncMock()
        es.search.return_value = {
            "took": 15,
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_source": {
                            "podcast_id": "show_abc",
                            "episode_id": "ep001",
                            "clip_index": 0,
                            "clip_start_ms": 0,
                            "clip_end_ms": 120000,
                            "clip_text": "hello world",
                            "word_timestamps": [{"word": "hello", "start_ms": 0, "end_ms": 500}],
                            "speakers": [1],
                        },
                        "_score": 5.5,
                        "highlight": {"clip_text": ["<em>hello</em> world"]},
                    }
                ],
            },
        }
        return es

    @pytest.fixture
    def mock_db(self):
        db = AsyncMock()
        db.fetchrow.return_value = {
            "show_name": "My Show",
            "episode_name": "Episode 1",
            "audio_link": None,
        }
        return db

    @pytest.fixture
    def mock_cache(self):
        cache = AsyncMock(spec=CacheClient)
        cache.get.return_value = None
        cache.cache_key.return_value = "testkey"
        return cache

    @pytest.mark.asyncio
    async def test_execute_search_returns_response(self, mock_es, mock_db, mock_cache):
        result = await execute_search("hello", 0, 10, mock_es, mock_db, mock_cache)
        assert isinstance(result, SearchResponse)
        assert result.total == 1
        assert result.clips[0].podcast_id == "show_abc"
        assert result.clips[0].score == 5.5
        assert "<em>" in result.clips[0].highlight

    @pytest.mark.asyncio
    async def test_execute_search_enriches_from_db(self, mock_es, mock_db, mock_cache):
        result = await execute_search("hello", 0, 10, mock_es, mock_db, mock_cache)
        assert result.clips[0].show_name == "My Show"
        assert result.clips[0].episode_name == "Episode 1"

    @pytest.mark.asyncio
    async def test_execute_search_empty_results(self, mock_db, mock_cache):
        es = AsyncMock()
        es.search.return_value = {"took": 5, "hits": {"total": {"value": 0}, "hits": []}}
        result = await execute_search("nothing", 0, 10, es, mock_db, mock_cache)
        assert result.total == 0
        assert result.clips == []

    @pytest.mark.asyncio
    async def test_execute_search_uses_cache_on_hit(self, mock_es, mock_db, mock_cache):
        cached = SearchResponse(query="cached", total=1, clips=[], took_ms=5)
        mock_cache.get.return_value = cached
        result = await execute_search("cached", 0, 10, mock_es, mock_db, mock_cache)
        assert result.query == "cached"
        mock_es.search.assert_not_called()
