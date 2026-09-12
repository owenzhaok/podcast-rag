from unittest.mock import MagicMock, patch
from elasticsearch import BadRequestError
from ingest.es_client import create_index, bulk_index, INDEX_MAPPING


class TestCreateIndex:
    def test_create_index_calls_indices_create(self):
        es = MagicMock()
        create_index(es, "test_index")
        es.indices.create.assert_called_once_with(index="test_index", body=INDEX_MAPPING)

    def test_create_index_idempotent_on_resource_exists(self):
        es = MagicMock()
        es.indices.create.side_effect = BadRequestError(
            message="resource_already_exists_exception",
            meta=MagicMock(),
            body={"error": {"type": "resource_already_exists_exception"}},
        )
        # Should not raise
        create_index(es, "test_index")


class TestBulkIndex:
    @patch("ingest.es_client.helpers")
    def test_bulk_index_returns_indexed_count(self, mock_helpers):
        mock_helpers.bulk.return_value = (10, [])
        es = MagicMock()
        docs = [{"_id": str(i)} for i in range(10)]
        result = bulk_index(es, "test_index", docs)
        assert result.indexed == 10
        assert result.failed == 0

    @patch("ingest.es_client.helpers")
    def test_bulk_index_captures_errors(self, mock_helpers):
        mock_helpers.bulk.return_value = (8, [{"index": {"error": "mapping error"}}])
        es = MagicMock()
        docs = [{"_id": str(i)} for i in range(10)]
        result = bulk_index(es, "test_index", docs)
        assert result.indexed == 8
        assert result.failed == 1
        assert len(result.errors) == 1
