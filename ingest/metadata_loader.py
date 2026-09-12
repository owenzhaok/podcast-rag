"""Load podcast metadata from TSV into the database."""

import csv
from dataclasses import dataclass, field


class MetadataLoadError(Exception):
    pass


@dataclass
class LoadResult:
    shows_upserted: int
    episodes_upserted: int
    errors: list[str] = field(default_factory=list)


REQUIRED_COLUMNS = {
    "show_uri", "show_name", "show_description", "publisher", "language",
    "rss_link", "episode_uri", "episode_name", "episode_description",
    "duration", "show_filename_prefix", "episode_filename_prefix",
}


def load_metadata(tsv_path: str, conn) -> LoadResult:
    """Parse TSV, upsert into shows and episodes tables.

    Uses INSERT OR REPLACE for SQLite compatibility.
    """
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not REQUIRED_COLUMNS.issubset(set(reader.fieldnames or [])):
            missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
            raise MetadataLoadError(f"Missing columns: {missing}")

        shows_seen = set()
        shows_upserted = 0
        episodes_upserted = 0
        errors = []
        cursor = conn.cursor()

        for row in reader:
            try:
                show_id = row["show_filename_prefix"]
                episode_id = row["episode_filename_prefix"]

                if show_id not in shows_seen:
                    cursor.execute(
                        "INSERT OR REPLACE INTO shows (show_id, name, description, publisher, rss_link) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (show_id, row["show_name"], row["show_description"], row["publisher"], row["rss_link"]),
                    )
                    shows_seen.add(show_id)
                    shows_upserted += 1

                cursor.execute(
                    "INSERT OR REPLACE INTO episodes (episode_id, show_id, name, description, audio_link, duration_s, language) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (episode_id, show_id, row["episode_name"], row["episode_description"],
                     None, int(float(row["duration"])), row["language"]),
                )
                episodes_upserted += 1
            except Exception as e:
                errors.append(str(e))

        conn.commit()

    return LoadResult(shows_upserted=shows_upserted, episodes_upserted=episodes_upserted, errors=errors)
