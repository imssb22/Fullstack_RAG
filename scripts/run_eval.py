from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = PROJECT_ROOT / "data" / "evaluation_questions.json"


def post_json(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a small RAG behavior check.")
    parser.add_argument(
        "--api-base",
        default="http://localhost:8000",
        help="Backend API base URL, for example https://your-api.onrender.com",
    )
    args = parser.parse_args()

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    failures = 0

    for item in questions:
        try:
            result = post_json(
                f"{args.api_base.rstrip('/')}/api/ask",
                {"question": item["question"], "document_ids": []},
            )
        except urllib.error.URLError as exc:
            print(f"ERROR {item['id']}: {exc}")
            return 2

        supported = bool(result.get("supported"))
        citations = result.get("citations") or []
        expected_supported = item["type"] == "answerable"

        passed = (
            supported == expected_supported
            and (not supported or len(citations) > 0)
        )
        if not passed:
            failures += 1

        status = "PASS" if passed else "CHECK"
        print(f"{status} {item['id']}")
        print(f"  expected: {item['type']}")
        print(f"  supported: {supported}, citations: {len(citations)}")
        print(f"  answer: {result.get('answer')}")

    if failures:
        print(f"\n{failures} item(s) need manual review or threshold tuning.")
        return 1

    print("\nEvaluation smoke check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
