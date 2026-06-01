from __future__ import annotations

import json
import re
from typing import Any, Protocol

import httpx

from app.config import Settings
from app.services.gemini import GeminiClient, GeminiError


class LLMError(RuntimeError):
    pass


class LLMClient(Protocol):
    def is_configured(self) -> bool:
        ...

    async def generate_json(self, prompt: str) -> dict[str, Any]:
        ...


class GeminiLLM:
    def __init__(self, gemini: GeminiClient):
        self.gemini = gemini

    def is_configured(self) -> bool:
        return self.gemini.is_configured()

    async def generate_json(self, prompt: str) -> dict[str, Any]:
        try:
            return await self.gemini.generate_json(prompt)
        except GeminiError as exc:
            raise LLMError(str(exc)) from exc


class OpenRouterLLM:
    def __init__(self, settings: Settings):
        self.settings = settings

    def is_configured(self) -> bool:
        return bool(self.settings.openrouter_api_key)

    def _key(self) -> str:
        if not self.settings.openrouter_api_key:
            raise LLMError("OPENROUTER_API_KEY is missing. Add it to the .env file.")
        return self.settings.openrouter_api_key

    async def generate_json(self, prompt: str) -> dict[str, Any]:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._key()}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.settings.openrouter_site_url,
            "X-OpenRouter-Title": self.settings.openrouter_app_name,
        }
        body = {
            "model": self.settings.openrouter_model,
            "messages": [
                {
                    "role": "system",
                    "content": "Return valid JSON only. Do not wrap JSON in markdown.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 1200,
            "response_format": {"type": "json_object"},
        }

        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(url, headers=headers, json=body)

        if response.status_code >= 400:
            raise LLMError(
                f"OpenRouter API error {response.status_code}: {response.text}"
            )

        data = response.json()
        text = self._extract_text(data)
        return self._parse_json(text)

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Unexpected OpenRouter response: {data}") from exc

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                raise LLMError(f"Model did not return JSON: {text}")
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError as exc:
                raise LLMError(f"Model did not return valid JSON: {text}") from exc


def create_llm(settings: Settings, gemini: GeminiClient) -> LLMClient:
    provider = settings.llm_provider.lower().strip()
    if provider == "openrouter":
        return OpenRouterLLM(settings)
    if provider == "gemini":
        return GeminiLLM(gemini)
    raise ValueError("LLM_PROVIDER must be either 'openrouter' or 'gemini'.")
