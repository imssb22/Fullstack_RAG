import json
from pathlib import Path

from app.schemas import DocumentRecord


class DocumentRegistry:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[DocumentRecord]:
        return sorted(self._load().values(), key=lambda item: item.created_at)

    def get(self, document_id: str) -> DocumentRecord | None:
        return self._load().get(document_id)

    def upsert(self, record: DocumentRecord) -> None:
        records = self._load()
        records[record.id] = record
        self._save(records)

    def _load(self) -> dict[str, DocumentRecord]:
        if not self.path.exists():
            return {}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return {
            document_id: DocumentRecord.model_validate(record)
            for document_id, record in data.items()
        }

    def _save(self, records: dict[str, DocumentRecord]) -> None:
        payload = {
            document_id: record.model_dump(mode="json")
            for document_id, record in records.items()
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
