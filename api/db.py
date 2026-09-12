"""Postgres async connection and queries."""

import asyncpg


async def create_pool(dsn: str) -> asyncpg.Pool:
    """Create a connection pool."""
    return await asyncpg.create_pool(dsn, min_size=2, max_size=10)


async def get_episode_info(pool: asyncpg.Pool, episode_id: str) -> dict | None:
    """Fetch episode + show info for display enrichment."""
    row = await pool.fetchrow(
        """
        SELECT s.name as show_name, e.name as episode_name, e.audio_link
        FROM episodes e
        JOIN shows s ON e.show_id = s.show_id
        WHERE e.episode_id = $1
        """,
        episode_id,
    )
    if row is None:
        return None
    return dict(row)
