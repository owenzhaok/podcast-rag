"""Centralized, versioned document and query embedding conventions."""

import os

import psycopg2


DOCUMENT_INPUT_VERSION = "gemini-document-v1"
QUERY_INPUT_VERSION = "gemini-query-v1"


def query_input(question: str) -> str:
    """gemini-query-v1: preserve the exact validated question."""
    return f"task: question answering | query: {question}"


def document_input(document: dict, episode_title: str | None = None) -> str:
    # Preserve exact transcript text and use a stable corpus-based metadata fallback.
    title = episode_title.strip() if isinstance(episode_title, str) and episode_title.strip() else (
        f"{document['podcast_id']}/{document['episode_id']}")
    return f"title: {title} | text: {document['chunk_text']}"


def episode_titles(episode_ids: list[str]) -> dict[str, str]:
    """Bounded read-only metadata lookup for the CLI, with deterministic fallback."""
    try:
        connection = psycopg2.connect(os.environ.get(
            "POSTGRES_DSN", "postgresql://podcast:podcast@localhost:5432/podcasts"),
            connect_timeout=3, options="-c statement_timeout=3000")
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT episode_id, name FROM episodes WHERE episode_id = ANY(%s)", (episode_ids,))
                return dict(cursor.fetchall())
        finally:
            connection.close()
    except psycopg2.Error:
        return {}  # Never print a DSN or raw database exception.
