import json
from pathlib import Path
import urllib.request

from app.config import Settings


def load_sample_sources(settings: Settings) -> list[dict]:
    if not settings.sample_sources_path.exists():
        return []
    return json.loads(settings.sample_sources_path.read_text(encoding="utf-8"))


def download_sample_sources(settings: Settings) -> tuple[list[str], list[str]]:
    settings.sample_docs_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[str] = []
    skipped: list[str] = []

    for source in load_sample_sources(settings):
        filename = source["filename"]
        target = settings.sample_docs_dir / filename
        if target.exists() and target.stat().st_size > 0:
            skipped.append(filename)
            continue

        request = urllib.request.Request(
            source["url"],
            headers={"User-Agent": "Atman-RAG-Demo/1.0"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            target.write_bytes(response.read())
        downloaded.append(filename)

    return downloaded, skipped


def sample_files(settings: Settings) -> list[tuple[Path, dict]]:
    by_filename = {
        source["filename"]: source for source in load_sample_sources(settings)
    }
    files: list[tuple[Path, dict]] = []
    for path in sorted(settings.sample_docs_dir.glob("*")):
        if path.name in by_filename:
            files.append((path, by_filename[path.name]))
    return files
