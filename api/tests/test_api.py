import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient, ASGITransport
from api.models import SearchResponse


@pytest.fixture
def mock_search():
    """Patch execute_search to return a controlled response."""
    response = SearchResponse(
        query="test",
        total=1,
        clips=[
            {
                "podcast_id": "show_abc",
                "episode_id": "ep001",
                "show_name": "Test Show",
                "episode_name": "Test Episode",
                "clip_start_ms": 0,
                "clip_end_ms": 120000,
                "score": 5.0,
                "highlight": "<em>test</em> content",
                "word_timestamps": [],
                "speakers": [1],
                "audio_link": None,
            }
        ],
        took_ms=10,
    )
    with patch("api.main.execute_search", new_callable=AsyncMock, return_value=response) as m:
        yield m


@pytest.fixture
def app(mock_search):
    """Create the FastAPI app with mocked dependencies."""
    from api.main import create_app
    return create_app()


@pytest_asyncio.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestSearchEndpoint:
    @pytest.mark.asyncio
    async def test_search_endpoint_returns_200(self, client):
        resp = await client.get("/search", params={"q": "test"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["query"] == "test"

    @pytest.mark.asyncio
    async def test_search_endpoint_validates_empty_query(self, client):
        resp = await client.get("/search", params={"q": ""})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_search_result_structure(self, client):
        resp = await client.get("/search", params={"q": "test"})
        body = resp.json()
        assert "total" in body
        assert "clips" in body
        assert "took_ms" in body
        assert len(body["clips"]) == 1
        clip = body["clips"][0]
        assert "podcast_id" in clip
        assert "highlight" in clip

    @pytest.mark.asyncio
    async def test_search_pagination(self, client, mock_search):
        resp = await client.get("/search", params={"q": "test", "from": 10, "size": 5})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_search_clip_minutes_default(self, client):
        resp = await client.get("/search", params={"q": "test"})
        assert resp.status_code == 200


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_endpoint(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
