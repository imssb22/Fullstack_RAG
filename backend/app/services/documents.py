from datetime import datetime, timezone
from hashlib import sha256
import mimetypes
from pathlib import Path
import re

from fastapi import UploadFile
from pypdf import PdfReader

from app.services.chunker import PageText


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}


def safe_filename(filename: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", filename).strip(".-")
    return cleaned or f"upload-{int(datetime.now(tz=timezone.utc).timestamp())}"


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def document_id_for(path: Path, source_url: str | None = None) -> str:
    seed = source_url or f"{path.name}:{file_sha256(path)}"
    return sha256(seed.encode("utf-8")).hexdigest()[:16]


async def save_upload(upload: UploadFile, uploads_dir: Path) -> Path:
    uploads_dir.mkdir(parents=True, exist_ok=True)
    filename = safe_filename(upload.filename or "document")
    target = uploads_dir / filename
    suffix = target.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError("Only PDF, text, and markdown files are supported.")

    content = await upload.read()
    if not content:
        raise ValueError("Uploaded file was empty.")

    target.write_bytes(content)
    return target


def detect_file_type(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return "pdf"
    if path.suffix.lower() in {".md", ".markdown"}:
        return "markdown"
    return "text"


def load_document(path: Path) -> list[PageText]:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported document type: {suffix}")

    if suffix == ".pdf":
        return load_pdf(path)

    text = path.read_text(encoding="utf-8", errors="ignore")
    return [PageText(page=None, text=text)]


def load_pdf(path: Path) -> list[PageText]:
    reader = PdfReader(str(path))
    pages: list[PageText] = []
    for index, page in enumerate(reader.pages, start=1):
        pages.append(PageText(page=index, text=page.extract_text() or ""))
    return pages


def title_from_path(path: Path) -> str:
    stem = path.stem.replace("-", " ").replace("_", " ")
    return re.sub(r"\s+", " ", stem).strip().title()


def mime_type_for(path: Path) -> str:
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"
