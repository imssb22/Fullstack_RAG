from dataclasses import dataclass
import re


@dataclass(frozen=True)
class PageText:
    page: int | None
    text: str


@dataclass(frozen=True)
class TextChunk:
    page: int | None
    chunk_index: int
    text: str


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_pages(
    pages: list[PageText],
    chunk_chars: int,
    overlap_chars: int,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    chunk_index = 0

    for page in pages:
        text = normalize_text(page.text)
        if not text:
            continue

        start = 0
        while start < len(text):
            end = min(start + chunk_chars, len(text))
            candidate = text[start:end]

            if end < len(text):
                boundary = max(
                    candidate.rfind("\n\n"),
                    candidate.rfind(". "),
                    candidate.rfind("? "),
                    candidate.rfind("! "),
                )
                if boundary > chunk_chars * 0.55:
                    end = start + boundary + 1
                    candidate = text[start:end]

            cleaned = normalize_text(candidate)
            if cleaned:
                chunks.append(
                    TextChunk(
                        page=page.page,
                        chunk_index=chunk_index,
                        text=cleaned,
                    )
                )
                chunk_index += 1

            if end >= len(text):
                break
            start = max(0, end - overlap_chars)

    return chunks
