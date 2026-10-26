from __future__ import annotations
from typing import Any, Protocol
import httpx
from src.config import settings

class LLMClient(Protocol):
    def generate(self, message: list[dict], **opts:Any) -> dict: ...
    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]: ...
    def list_models(self) -> list[str]: ...

class OllamaClient:
    def __init__(self, base_url: str | None = None, timeout: float = 300.0) -> None:
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.timeout = timeout

    def generate(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        fmt: str | None = None,
        num_ctx: int | None = None,
    ) -> dict: 
        payload: dict[str, Any] = {
            "model": model or settings.llm_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx or settings.num_ctx,
            },
        }
        if fmt is not None:
            payload["format"] = fmt  # "json" for the agent and the judge

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(f"{self.base_url}/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()

        return {
            "text": data["message"]["content"],
            "tokens_in": int(data.get("prompt_eval_count", 0)),
            "tokens_out": int(data.get("eval_count", 0)),
            "done_reason": data.get("done_reason"),
        }

    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]: 
        payload = {
            "model": model or settings.embed_model, 
            "input": list(texts),
        }
        with httpx.Client(timeout=self.timeout) as client: 
            response = client.post(f"{self.base_url}/api/embed", json=payload)
        response.raise_for_status()
        vectors = response.json()["embeddings"]
        if len(vectors) != len(texts):
            raise RuntimeError(f"expected {len(texts)} embeddings, got {len(vectors)}")
        return vectors

    def list_models(self) -> list[str]:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{self.base_url}/api/tags")
        response.raise_for_status()
        return sorted(item["name"] for item in response.json().get("models", []))

def get_client() -> LLMClient:
    if settings.llm_provider == "ollama":
        return OllamaClient()
    raise ValueError(
        f"unknown LLM_PROVIDER={settings.llm_provider!r}; this phase support 'ollama'"
    )

    