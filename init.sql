CREATE TABLE IF NOT EXISTS shows (
    show_id     TEXT PRIMARY KEY,
    name        TEXT,
    description TEXT,
    publisher   TEXT,
    rss_link    TEXT
);

CREATE TABLE IF NOT EXISTS episodes (
    episode_id   TEXT PRIMARY KEY,
    show_id      TEXT REFERENCES shows(show_id),
    name         TEXT,
    description  TEXT,
    audio_link   TEXT,
    duration_s   INTEGER,
    language     TEXT
);
