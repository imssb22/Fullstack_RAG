from __future__ import annotations

import json
from pathlib import Path
import sys
import urllib.request


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCES_PATH = PROJECT_ROOT / "data" / "sample_sources.json"
TARGET_DIR = PROJECT_ROOT / "data" / "sample_docs"


def main() -> int:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    sources = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))

    for source in sources:
        target = TARGET_DIR / source["filename"]
        if target.exists() and target.stat().st_size > 0:
            print(f"skip {target.name}")
            continue

        request = urllib.request.Request(
            source["url"],
            headers={"User-Agent": "Atman-RAG-Demo/1.0"},
        )
        print(f"download {source['title']}")
        with urllib.request.urlopen(request, timeout=120) as response:
            target.write_bytes(response.read())
        print(f"saved {target.relative_to(PROJECT_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
