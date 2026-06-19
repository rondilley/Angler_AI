"""Angler_AI-specific HTTP routes. Mounted alongside the llama-cpp-python
OpenAI-compatible server.

Bind to 127.0.0.1 by default (NFR-3.6). 0.0.0.0 requires explicit --bind.
"""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="Angler_AI", version="0.0.1")


@app.get("/v1/status")
def status() -> dict[str, object]:
    """NFR-6.4: currently loaded model, hardware backend, data freshness, disk usage."""
    raise NotImplementedError("M1 milestone.")


@app.get("/v1/reaches")
def reaches(state: str, county: str) -> list[dict[str, object]]:
    raise NotImplementedError("M3 milestone.")


@app.get("/v1/probability")
def probability(comid: int, species: str, date: str) -> dict[str, object]:
    raise NotImplementedError("M4 milestone.")


@app.get("/v1/anomaly")
def anomaly(gauge_id: str) -> dict[str, object]:
    raise NotImplementedError("M6 milestone.")


@app.get("/v1/regulations")
def regulations(state: str, water_body: str, species: str, date: str) -> dict[str, object]:
    raise NotImplementedError("M2 milestone.")


@app.get("/v1/map.geojson")
def map_geojson(county: str, species: str, date: str) -> dict[str, object]:
    raise NotImplementedError("M4 milestone.")
