"""Generate small synthetic transcript fixtures for testing."""

import json
import os
import random


SAMPLE_WORDS = [
    "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
    "machine", "learning", "artificial", "intelligence", "neural", "network",
    "podcast", "episode", "interview", "discussion", "science", "technology",
    "quantum", "physics", "research", "university", "professor", "explains",
]


def generate_transcript(episode_id: str, duration_s: int = 300, seed: int = 42) -> dict:
    """Generate a synthetic transcript JSON matching the Spotify format."""
    rng = random.Random(seed)
    words = []
    t = 0.0
    while t < duration_s:
        word = rng.choice(SAMPLE_WORDS)
        gap = rng.uniform(0.1, 0.8)
        start = round(t, 3)
        end = round(t + 0.3, 3)
        words.append({
            "startTime": f"{start}s",
            "endTime": f"{end}s",
            "word": word,
            "speakerTag": rng.choice([1, 2]),
        })
        t += gap + 0.3

    # Build the results array: one chunk + final full-episode entry
    chunk_words_no_speaker = [
        {"startTime": w["startTime"], "endTime": w["endTime"], "word": w["word"]}
        for w in words[:50]
    ]
    results = [
        {"alternatives": [{"transcript": " ".join(w["word"] for w in words[:50]), "words": chunk_words_no_speaker}]},
        {"alternatives": [{"words": words}]},
    ]
    return {"results": results}


def generate_metadata_tsv(episodes: list[tuple[str, str]], output_path: str):
    """Generate a minimal metadata TSV for the given (show_id, episode_id) pairs."""
    lines = [
        "show_uri\tshow_name\tshow_description\tpublisher\tlanguage\trss_link\t"
        "episode_uri\tepisode_name\tepisode_description\tduration\tshow_filename_prefix\tepisode_filename_prefix"
    ]
    for show_id, episode_id in episodes:
        lines.append(
            f"spotify:show:{show_id}\tTest Show {show_id}\tA test show\tTest Publisher\ten\t"
            f"http://rss.example.com/{show_id}\tspotify:episode:{episode_id}\t"
            f"Test Episode {episode_id}\tA test episode\t300\t{show_id}\t{episode_id}"
        )
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def generate_fixtures(output_dir: str, count: int = 5):
    """Generate count transcript files + a metadata.tsv in output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    episodes = []
    show_id = "show_test001"
    show_dir = os.path.join(output_dir, "transcripts", show_id)
    os.makedirs(show_dir, exist_ok=True)

    for i in range(count):
        ep_id = f"ep_test{i:03d}"
        transcript = generate_transcript(ep_id, duration_s=300, seed=i)
        with open(os.path.join(show_dir, f"{ep_id}.json"), "w") as f:
            json.dump(transcript, f)
        episodes.append((show_id, ep_id))

    generate_metadata_tsv(episodes, os.path.join(output_dir, "metadata.tsv"))
    print(f"Generated {count} transcripts + metadata.tsv in {output_dir}")
    return episodes


if __name__ == "__main__":
    generate_fixtures("/tmp/podcast_test_fixtures")
