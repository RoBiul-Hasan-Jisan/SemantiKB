"""
Thin client for the local Ollama server (https://ollama.com). No paid API,
no external network calls — everything runs on localhost.
"""
from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, base_url: str, model: str, temperature: float = 0.1, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def generate(self, prompt: str, system: str | None = None, max_tokens: int = 512) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": max_tokens},
        }
        if system:
            payload["system"] = system
        try:
            resp = httpx.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json().get("response", "")
        except Exception as e:
            logger.error("Ollama request failed: %s", e)
            raise

    def chat(self, messages: list[dict], max_tokens: int = 512) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": max_tokens},
        }
        resp = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "")

    def health_check(self) -> bool:
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False
