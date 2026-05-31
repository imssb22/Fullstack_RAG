import json
from typing import Any

import httpx

from app.config import Settings


class GeminiError(RuntimeError):
    pass


class GeminiClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def is_configured(self) -> bool:
        return bool(self.settings.gemini_api_key)

    def _key(self) -> str:
        if not self.settings.gemini_api_key:
            raise GeminiError("GEMINI_API_KEY is missing. Add it to the root .env file.")
        return self.settings.gemini_api_key

    async def embed_texts(self, texts: list[str], task_type: str) -> list[list[float]]:
        embeddings: list[list[float]] = []
        batch_size = 50
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            embeddings.extend(await self._embed_batch(batch, task_type))
        return embeddings

    async def embed_text(self, text: str, task_type: str) -> list[float]:
        return (await self.embed_texts([text], task_type))[0]

    async def _embed_batch(self, texts: list[str], task_type: str) -> list[list[float]]:
        model = self.settings.gemini_embedding_model
        url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            f"models/{model}:batchEmbedContents"
        )
        body = {
            "requests": [
                {
                    "model": f"models/{model}",
                    "content": {"parts": [{"text": text}]},
                    "taskType": task_type,
                }
                for text in texts
            ]
        }
        data = await self._post(url, body)
        try:
            return [item["values"] for item in data["embeddings"]]
        except KeyError as exc:
            raise GeminiError(f"Unexpected embedding response: {data}") from exc

    async def generate_json(self, prompt: str) -> dict[str, Any]:
        model = self.settings.gemini_model
        url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            f"models/{model}:generateContent"
        )
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "topP": 0.8,
                "maxOutputTokens": 1200,
                "responseMimeType": "application/json",
            },
        }
        data = await self._post(url, body)
        text = self._extract_text(data)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise GeminiError(f"Model did not return valid JSON: {text}") from exc

    async def _post(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        params = {"key": self._key()}
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(url, params=params, json=body)
        if response.status_code >= 400:
            raise GeminiError(f"Gemini API error {response.status_code}: {response.text}")
        return response.json()

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as exc:
            raise GeminiError(f"Unexpected generation response: {data}") from exc
