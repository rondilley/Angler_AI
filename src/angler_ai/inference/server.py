"""OpenAI-compatible HTTP server using llama_cpp's `create_chat_completion`.

A minimal FastAPI app with /v1/chat/completions and /v1/status. The response
shape comes straight from llama-cpp-python's OpenAI-compatible completion
helper, so standard OpenAI clients work against this endpoint (FR-2.5).

Binding defaults to 127.0.0.1 (NFR-3.6). Override only with explicit --bind.
"""

from __future__ import annotations

import logging
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from angler_ai.inference.runtime import InferenceRuntime, LoadedModel

log = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """Subset of the OpenAI chat completion request schema. Extend at M6."""

    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = Field(default=512, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    stream: bool = False


def build_app(loaded: LoadedModel) -> FastAPI:
    """Wire a FastAPI app around the already-loaded LoadedModel.

    The app holds a reference to the LoadedModel; tearing it down releases the
    GGUF from VRAM via runtime.unload().
    """
    app = FastAPI(
        title="Angler_AI inference server",
        version="0.0.1",
        description="OpenAI-compatible /v1/chat/completions backed by llama-cpp-python.",
    )

    @app.get("/v1/status")
    def status() -> dict[str, Any]:
        """NFR-6.4: loaded-model details and basic server info."""
        return {
            "loaded_model": {
                "id": loaded.model_id,
                "quant": loaded.quant,
                "path": str(loaded.local_path),
                "n_ctx": loaded.n_ctx,
                "n_gpu_layers": loaded.n_gpu_layers,
            },
        }

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        """OpenAI-compatible model listing. Returns the one loaded model."""
        return {
            "object": "list",
            "data": [
                {
                    "id": loaded.model_id,
                    "object": "model",
                    "owned_by": "angler-ai",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest) -> dict[str, Any]:
        """OpenAI-compatible chat completion. Streaming is M1+ (return non-stream now)."""
        if req.stream:
            raise HTTPException(
                status_code=501,
                detail="Streaming responses not implemented in M1; pass stream=false.",
            )
        if req.model and req.model != loaded.model_id:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Requested model {req.model!r} not loaded. "
                    f"Server has {loaded.model_id!r}. Restart with the desired model."
                ),
            )
        # llama_cpp's Llama.create_chat_completion returns an OpenAI-shaped dict.
        result = loaded.handle.create_chat_completion(  # type: ignore[attr-defined]
            messages=[{"role": m.role, "content": m.content} for m in req.messages],
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            stream=False,
        )
        return result

    return app


def serve(
    loaded: LoadedModel,
    host: str = "127.0.0.1",
    port: int = 8089,
) -> None:
    """Start uvicorn on the given (host, port). Blocking.

    Honors NFR-3.6: callers binding 0.0.0.0 are surfaced as a warning by the
    CLI before invoking this function.
    """
    runtime = InferenceRuntime()
    runtime._loaded = loaded  # type: ignore[attr-defined]
    app = build_app(loaded)
    log.info("Serving %s/%s at http://%s:%d", loaded.model_id, loaded.quant, host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")
