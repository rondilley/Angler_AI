"""Typer-based CLI. Entry point: `angler-ai` (see pyproject.toml [project.scripts])."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table

from angler_ai import __version__
from angler_ai.config import LAUNCH_STATES, default_paths, ensure_paths
from angler_ai.hardware import probe as hardware_probe
from angler_ai.hardware.backend import select_backend
from angler_ai.inference.runtime import InferenceRuntime
from angler_ai.inference.server import serve as serve_app
from angler_ai.logging_config import configure as configure_logging
from angler_ai.registry import (
    DownloadResult,
    GGUFNotFoundError,
    ModelEntry,
    ModelType,
    QuantVariant,
    download,
    load_catalog,
    read_manifest,
    update_manifest,
)

app = typer.Typer(
    name="angler-ai",
    help="Local-first, hardware-adaptive fishing intelligence for US rivers and streams.",
    no_args_is_help=True,
)
console = Console()


def _bootstrap() -> tuple[object, object]:
    """Initialize paths and logging on every command invocation.
    Returns (Paths, HardwareProfile) for the caller to reuse.
    """
    paths = default_paths()
    ensure_paths(paths)
    configure_logging(paths.logs)
    return paths, hardware_probe()


@app.callback()
def main(
    version: Annotated[bool, typer.Option("--version", help="Print version and exit.")] = False,
) -> None:
    if version:
        console.print(f"angler-ai {__version__}")
        raise typer.Exit()


# --------------------------------------------------------------------------- probe

@app.command()
def probe(
    json_out: Annotated[bool, typer.Option("--json", help="Emit JSON instead of a rich table.")] = False,
) -> None:
    """Detect hardware and report the chosen llama.cpp backend (M1, FR-1)."""
    paths, hp = _bootstrap()
    backend = select_backend(hp)

    if json_out:
        payload = {
            "version": __version__,
            "cpu": {
                "arch": hp.cpu.arch,
                "microarch": hp.cpu.microarch,
                "features": sorted(hp.cpu.features),
                "cores_physical": hp.cpu.cores_physical,
                "cores_logical": hp.cpu.cores_logical,
            },
            "ram": {"total_gb": hp.ram.total_gb, "available_gb": hp.ram.available_gb},
            "gpus": [
                {
                    "vendor": g.vendor.value,
                    "name": g.name,
                    "vram_total_mb": g.vram_total_mb,
                    "vram_available_mb": g.vram_available_mb,
                    "compute_capability": g.compute_capability,
                    "cuda_version": g.cuda_version,
                    "unified_memory": g.unified_memory,
                }
                for g in hp.gpus
            ],
            "os": {"system": hp.os.system, "release": hp.os.release, "machine": hp.os.machine},
            "python": {"major": hp.python.major, "minor": hp.python.minor, "supported": hp.python.supported},
            "backend": {
                "variant": backend.backend.value,
                "extra_index_url": backend.extra_index_url,
                "note": backend.note,
                "runtime_cuda_version": backend.runtime_cuda_version,
            },
        }
        console.print_json(json.dumps(payload))
        return

    t = Table(title="Angler_AI hardware profile", show_header=True)
    t.add_column("Component"); t.add_column("Detail")
    t.add_row("CPU", f"{hp.cpu.microarch} ({hp.cpu.arch}) - {hp.cpu.cores_physical} physical / {hp.cpu.cores_logical} logical")
    if hp.cpu.features:
        t.add_row("CPU features", ", ".join(sorted(hp.cpu.features))[:200])
    t.add_row("RAM", f"{hp.ram.total_gb} GB total, {hp.ram.available_gb} GB available")
    for g in hp.gpus:
        t.add_row(
            f"GPU ({g.vendor.value})",
            f"{g.name} - {g.vram_available_mb} / {g.vram_total_mb} MB"
            + (f" (driver CUDA {g.cuda_version}, CC {g.compute_capability})" if g.cuda_version else "")
            + (" [unified memory]" if g.unified_memory else ""),
        )
    if not hp.gpus:
        t.add_row("GPU", "(none detected - CPU-only inference)")
    t.add_row("OS", f"{hp.os.system} {hp.os.release} ({hp.os.machine})")
    t.add_row("Python", f"{hp.python.major}.{hp.python.minor}.{hp.python.micro}"
              + ("" if hp.python.supported else " [WARNING: outside 3.10-3.12 wheel range]"))
    t.add_row("llama.cpp backend", f"{backend.backend.value} - {backend.note}")
    if backend.extra_index_url:
        t.add_row("Wheel index", backend.extra_index_url)
    console.print(t)


# --------------------------------------------------------------------------- pull-models

def _select_default_text_model(catalog: list[ModelEntry], vram_mb: int) -> tuple[ModelEntry, QuantVariant]:
    """Pick a (text-model, quant) pair for the user's VRAM tier.

    Mirrors the design 6 default tier table. Acceptance gates can override.
    """
    # Order text models by preference per tier.
    if vram_mb < 6_000:
        prefer_ids = ("llama-3.2-3b-instruct", "qwen3.5-4b")
        prefer_quant = "Q4_K_M"
    elif vram_mb < 12_000:
        prefer_ids = ("qwen3.5-9b", "qwen3.5-4b", "llama-3.2-3b-instruct")
        prefer_quant = "Q4_K_M"
    elif vram_mb < 20_000:
        prefer_ids = ("qwen3.5-35b-a3b", "qwen3.5-27b")
        prefer_quant = "IQ2_M"
    elif vram_mb < 32_000:
        prefer_ids = ("qwen3.5-35b-a3b",)
        prefer_quant = "Q4_K_XL"
    else:
        prefer_ids = ("qwen3.5-122b-a10b", "qwen3.5-35b-a3b")
        prefer_quant = "IQ2_XXS"

    by_id = {m.id: m for m in catalog if m.type == ModelType.TEXT}
    for mid in prefer_ids:
        model = by_id.get(mid)
        if model is None:
            continue
        for quant in model.quants:
            if quant.name == prefer_quant and quant.vram_required_mb <= vram_mb:
                return (model, quant)
        # Quant fallback: smallest fitting quant on this model.
        fits = [q for q in model.quants if q.vram_required_mb <= vram_mb]
        if fits:
            fits.sort(key=lambda q: q.vram_required_mb)
            return (model, fits[0])
    raise RuntimeError(
        f"No registry text model fits VRAM budget {vram_mb} MB. "
        "Consider lowering --profile or freeing GPU memory."
    )


def _confirm_license(model: ModelEntry, *, yes: bool) -> bool:
    """Surface the license and record acknowledgment (FR-3.5)."""
    console.print(
        f"\n[bold]Model:[/] {model.id}  ([cyan]{model.family}[/], {model.type.value})\n"
        f"[bold]Repo:[/]  {model.hf_repo}\n"
        f"[bold]License:[/] [yellow]{model.license}[/]\n"
    )
    if yes:
        console.print("[dim]License auto-accepted via --yes.[/]")
        return True
    return Confirm.ask(f"Accept the {model.license!r} license and download?", default=True)


@app.command(name="pull-models")
def pull_models(
    profile: Annotated[str, typer.Option(help="Tier: auto / small / mid / frontier.")] = "auto",
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip license prompts.")] = False,
    specific: Annotated[Optional[str], typer.Option("--model", help="Pull a specific model id from the catalog.")] = None,
) -> None:
    """Download appropriately-sized GGUF model(s) for the detected hardware (M1)."""
    paths, hp = _bootstrap()
    catalog = list(load_catalog())

    if specific:
        by_id = {m.id: m for m in catalog}
        if specific not in by_id:
            console.print(f"[red]Unknown model id:[/] {specific}. Use `angler-ai status` or read the catalog.")
            raise typer.Exit(code=2)
        model = by_id[specific]
        fits = [q for q in model.quants if q.vram_required_mb <= hp.available_vram_mb()]
        if not fits:
            console.print(f"[red]No quant of {specific} fits the {hp.available_vram_mb()} MB VRAM budget.[/]")
            raise typer.Exit(code=2)
        fits.sort(key=lambda q: q.vram_required_mb)
        targets = [(model, fits[0])]
    else:
        vram_mb = hp.available_vram_mb()
        # Tier override.
        tier_vram = {"small": 5_500, "mid": 16_000, "frontier": 999_999}.get(profile)
        if tier_vram is not None:
            vram_mb = min(vram_mb, tier_vram)
        targets = [_select_default_text_model(catalog, vram_mb)]

    for model, quant in targets:
        if not _confirm_license(model, yes=yes):
            console.print(f"[yellow]Skipped {model.id}/{quant.name} (license declined).[/]")
            continue
        try:
            result = download(model, quant, paths.models, license_acknowledged=True)
        except GGUFNotFoundError as exc:
            console.print(f"[red]GGUF not found for {model.id}/{quant.name}:[/] {exc}")
            continue
        update_manifest(paths.manifest, result)
        console.print(
            f"[green]Installed[/] {result.model_id}/{result.quant} "
            f"({result.bytes_downloaded // 1024 // 1024} MB) -> {result.local_path}"
        )


# --------------------------------------------------------------------------- status

@app.command()
def status() -> None:
    """Show installed models, hardware backend, and disk usage (NFR-6.4)."""
    paths, hp = _bootstrap()
    backend = select_backend(hp)
    manifest = read_manifest(paths.manifest)
    models_block: dict[str, object] = manifest.get("models", {})  # type: ignore[assignment]

    t = Table(title="Angler_AI status", show_header=True)
    t.add_column("Field"); t.add_column("Value")
    t.add_row("Version", __version__)
    t.add_row("OS", f"{hp.os.system} {hp.os.release}")
    t.add_row("llama.cpp backend", f"{backend.backend.value} - {backend.note}")
    t.add_row("VRAM budget", f"{hp.available_vram_mb()} MB")
    t.add_row("Models directory", str(paths.models))
    t.add_row("Feature store", str(paths.feature_store))
    t.add_row("Manifest", str(paths.manifest))
    t.add_row("Installed models", str(len(models_block)))
    console.print(t)

    if models_block:
        m = Table(title="Installed models", show_header=True)
        m.add_column("id/quant"); m.add_column("size (MB)"); m.add_column("license"); m.add_column("path")
        for key, entry in sorted(models_block.items()):
            e = entry  # type: ignore[assignment]
            size_mb = int(e["bytes_downloaded"]) // 1024 // 1024  # type: ignore[index]
            m.add_row(
                str(key),
                str(size_mb),
                str(e["license"]),  # type: ignore[index]
                str(e["local_path"]),  # type: ignore[index]
            )
        console.print(m)


# --------------------------------------------------------------------------- serve

@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind host. Default 127.0.0.1 (NFR-3.6).")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Bind port.")] = 8089,
    model: Annotated[Optional[str], typer.Option(help="Model id from the catalog. Default: first installed text model.")] = None,
    n_ctx: Annotated[int, typer.Option(help="Context length to allocate.")] = 8192,
) -> None:
    """Start the OpenAI-compatible inference server (FR-2.5)."""
    paths, hp = _bootstrap()
    if host == "0.0.0.0":
        console.print("[yellow]WARNING:[/] binding to 0.0.0.0 - server reachable on the network.")
    catalog = {m.id: m for m in load_catalog()}
    manifest = read_manifest(paths.manifest)
    installed: dict[str, dict] = manifest.get("models", {})  # type: ignore[assignment]
    if not installed:
        console.print("[red]No models installed.[/] Run `angler-ai pull-models` first.")
        raise typer.Exit(code=2)

    target_key: str
    target_path: Path
    target_quant: str
    target_model_id: str
    if model:
        # User specified model id; pick the first installed quant for that id.
        matches = [k for k in installed if k.startswith(f"{model}::")]
        if not matches:
            console.print(f"[red]Model {model!r} is not installed.[/] Run `angler-ai pull-models --model {model}`.")
            raise typer.Exit(code=2)
        target_key = matches[0]
    else:
        # First installed text model.
        text_ids = {m.id for m in load_catalog() if m.type == ModelType.TEXT}
        text_keys = sorted(k for k in installed if k.split("::", 1)[0] in text_ids)
        if not text_keys:
            console.print("[red]No installed text model in the manifest.[/] Run `angler-ai pull-models`.")
            raise typer.Exit(code=2)
        target_key = text_keys[0]

    entry = installed[target_key]
    target_model_id = target_key.split("::", 1)[0]
    target_quant = entry["local_path"] and target_key.split("::", 1)[1]
    target_path = Path(entry["local_path"])

    model_entry = catalog[target_model_id]
    quant_entry = next(q for q in model_entry.quants if q.name == target_quant)
    runtime = InferenceRuntime()
    loaded = runtime.ensure_loaded(model_entry, quant_entry, target_path, n_ctx=n_ctx)
    console.print(f"[green]Loaded[/] {target_model_id}/{target_quant} from {target_path}")
    serve_app(loaded, host=host, port=port)


# --------------------------------------------------------------------------- stubs for later milestones

@app.command()
def ingest(
    source: Annotated[str, typer.Option(help="Source id or 'all'.")] = "all",
    state: Annotated[Optional[str], typer.Option(help=f"State filter. Default: PA. v0 launch states: {LAUNCH_STATES}.")] = None,
    huc8: Annotated[Optional[str], typer.Option(help="8-digit HUC8 to ingest (NHDPlus HR + V2 xwalk).")] = None,
) -> None:
    """Populate the local feature store from federal + state sources (M2)."""
    paths, _hp = _bootstrap()
    target_state = (state or LAUNCH_STATES[0]).upper()
    from angler_ai.features import open_store
    from angler_ai.ingest import run as ingest_run

    extra_kwargs: dict[str, object] = {}
    if huc8:
        extra_kwargs["huc8"] = huc8

    with open_store(paths.feature_store) as store:
        store.initialize_schema()
        summaries = ingest_run(
            store,
            source=source,
            state=target_state,
            manifest_path=paths.manifest,
            extra_kwargs=extra_kwargs,
        )

    t = Table(title=f"Ingest summary ({target_state})", show_header=True)
    t.add_column("Source"); t.add_column("Rows"); t.add_column("OK"); t.add_column("Detail")
    for s in summaries:
        ok = "[green]yes[/]" if s.ok else "[red]no[/]"
        detail = s.error if s.error else f"license: {s.license}"
        t.add_row(s.source_id, str(s.rows_written), ok, (detail or "")[:80])
    console.print(t)


@app.command()
def species(
    comid: Annotated[int, typer.Option(help="NHDPlus HR COMID.")],
    top: Annotated[int, typer.Option(help="Top-K species to return.")] = 10,
    json_out: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False,
) -> None:
    """Show USGS BRT v2.0 species-presence priors for a reach (M3, FR-5.2).

    Surfaces top-K species ranked by BRT prediction probability for the
    NHDPlus HR `comid`. The HR -> V2.1 -> BRT join goes through
    xwalk_v2_to_hr; reaches lacking V2.1 mapping surface explicit
    'no V2.1 join' rather than silently dropping (DR-1.4).
    """
    paths, _ = _bootstrap()
    from angler_ai.features import open_store

    with open_store(paths.feature_store) as store:
        store.initialize_schema()
        conn = store.connect()
        # Confirm reach exists
        reach = conn.execute(
            "SELECT comid, reachcode, gnis_name, huc8 FROM reaches WHERE comid = ?",
            [comid],
        ).fetchone()
        if reach is None:
            console.print(f"[red]COMID {comid} not in reaches table.[/] Run `angler-ai ingest --source nhdplus --state PA` first.")
            raise typer.Exit(code=2)

        # Look up V2 COMIDs via xwalk
        v2_comids = [
            r[0] for r in conn.execute(
                "SELECT DISTINCT comid_v2 FROM xwalk_v2_to_hr WHERE comid_hr = ?",
                [comid],
            ).fetchall()
        ]
        if not v2_comids:
            # DR-1.4: explicit lack-of-join, not silent.
            payload = {
                "comid_hr": comid,
                "reach": {"reachcode": reach[1], "gnis_name": reach[2], "huc8": reach[3]},
                "v2_mapping": None,
                "no_v2_join": True,
                "note": (
                    "No NHDPlusV2.1 mapping for this HR reach in xwalk_v2_to_hr. "
                    "BRT predictions are V2.1-keyed; HR reaches finer than V2.1 "
                    "(many headwater tributaries) will not have priors."
                ),
                "predictions": [],
                "provenance": {
                    "model_id": "USGS_BRT_V2.0",
                    "doi": "https://doi.org/10.5066/P1UV25FW",
                },
            }
            if json_out:
                console.print_json(json.dumps(payload))
            else:
                console.print(f"[yellow]COMID {comid}: no NHDPlusV2.1 mapping[/] (reachcode {reach[1]}, {reach[2] or 'unnamed'}).")
                console.print(f"  HR reaches finer than V2.1 lack BRT priors. Source: {payload['provenance']['doi']}")
            return

        # Pull top-K predictions across all matched V2 COMIDs
        rows = conn.execute(
            """
            SELECT species, MAX(probability) AS p, COUNT(*) AS n_v2_matches
            FROM brt_priors
            WHERE comid = ANY(?)
              AND model_version = 'USGS_BRT_V2.0'
            GROUP BY species
            ORDER BY p DESC
            LIMIT ?
            """,
            [v2_comids, int(top)],
        ).fetchall()

        provenance = {
            "model_id": "USGS_BRT_V2.0",
            "model_version": "USGS_BRT_V2.0",
            "doi": "https://doi.org/10.5066/P1UV25FW",
            "citation": (
                "Yu, S.L., Cooper, A.R., Ross, J., McKerrow, A.J., Wieferich, D.J., "
                "Infante, D.M. 2023. Fluvial fish native distributions using "
                "NHDPlusV2.1 and Boosted Regression Tree models. USGS SIR 2023-5088."
            ),
        }

        if json_out:
            payload = {
                "comid_hr": comid,
                "reach": {"reachcode": reach[1], "gnis_name": reach[2], "huc8": reach[3]},
                "v2_comids": v2_comids,
                "predictions": [
                    {"species": r[0], "probability": float(r[1]), "v2_matches": int(r[2])}
                    for r in rows
                ],
                "provenance": provenance,
            }
            console.print_json(json.dumps(payload))
            return

        t = Table(
            title=f"Species priors at COMID {comid}  ({reach[2] or 'unnamed'} / reachcode {reach[1]})",
            show_header=True,
        )
        t.add_column("Rank"); t.add_column("Species"); t.add_column("Probability"); t.add_column("V2 matches")
        if not rows:
            t.add_row("-", "(no species with probability >= ingest threshold)", "-", "-")
        for i, (sp, p, n) in enumerate(rows, 1):
            t.add_row(str(i), sp, f"{p:.3f}", str(n))
        console.print(t)
        console.print(f"\n[dim]Model: {provenance['model_id']}  |  DOI: {provenance['doi']}[/]")


@app.command()
def map(  # noqa: A001 - intentional shadow of builtin in a CLI namespace
    species: Annotated[str, typer.Option(help="Common or scientific name, e.g. 'smallmouth bass' or 'Micropterus dolomieu'.")],
    date: Annotated[str, typer.Option(help="ISO date for the prediction window.")] = "2026-07-04",
    state: Annotated[Optional[str], typer.Option(help="Two-letter state code.")] = "PA",
    huc8: Annotated[Optional[str], typer.Option(help="8-digit HUC8 watershed filter.")] = None,
    limit: Annotated[Optional[int], typer.Option(help="Cap number of reaches in the output.")] = None,
    out: Annotated[Optional[str], typer.Option(help="Output GeoJSON path. Default: map.geojson in cwd.")] = None,
) -> None:
    """Emit a per-reach species-probability GeoJSON overlay (M4, FR-9.3).

    v0 uses USGS BRT v2.0 priors as the underlying probability substrate,
    wrapped in calibrated intervals via the hyperstability-aware Calibration
    Layer (FR-6.1, FR-6.4). v1 will swap in SSN2 ssn_glm(binomial).
    """
    paths, _ = _bootstrap()
    from angler_ai.features import open_store
    from angler_ai.prediction.map_export import build_feature_collection, write_geojson
    from angler_ai.prediction.species_priors import species_priors_for_geometry

    out_path = out or "map.geojson"

    with open_store(paths.feature_store) as store:
        store.initialize_schema()
        conn = store.connect()

        # Resolve species name: try scientific first, then common.
        sci = species
        common = None
        row = conn.execute(
            "SELECT scientific_name, common_name FROM brt_species WHERE scientific_name = ?",
            [species],
        ).fetchone()
        if row is None:
            normalized = species.replace("-", " ").lower()
            row = conn.execute(
                "SELECT scientific_name, common_name FROM brt_species "
                "WHERE LOWER(common_name) = ?",
                [normalized],
            ).fetchone()
        if row is None:
            console.print(f"[red]Species not found in BRT registry:[/] {species!r}")
            console.print("Try the scientific name (e.g. 'Micropterus dolomieu') or another common name.")
            raise typer.Exit(code=2)
        sci, common = row[0], row[1]

        rows = species_priors_for_geometry(
            store,
            species_scientific=sci,
            state=state,
            huc8=huc8,
            limit=limit,
        )
        if not rows:
            console.print(f"[yellow]No reaches found for {sci} with filters state={state} huc8={huc8}.[/]")
            console.print("Run `angler-ai ingest --source nhdplus --state PA` to populate reaches first.")
            raise typer.Exit(code=1)

        # Bulk-resolve real temperature per reach with honest source tag.
        from angler_ai.prediction.temperature import resolve_many
        temperatures = resolve_many(store, [sp.comid for sp, _ in rows])

        export = build_feature_collection(
            rows,
            species_scientific=sci,
            species_common=common,
            date=date,
            extra_metadata={"filters": {"state": state, "huc8": huc8, "limit": limit}},
            temperatures=temperatures,
        )
        write_geojson(export, out_path)

    console.print(
        f"[green]Wrote {export.feature_count} reaches[/] to [bold]{out_path}[/] "
        f"for {sci} ({common}) on {date}."
    )
    console.print(
        "  [dim]model: USGS_BRT_V2.0  |  hyperstability beta: 0.23  |  "
        "v0 calibrated-prior path (v1 will swap to SSN2)[/]"
    )


@app.command()
def forecast(
    species: Annotated[str, typer.Option(help="Species name (common or scientific).")],
    huc8: Annotated[str, typer.Option(help="8-digit HUC8 watershed code.")],
    out: Annotated[str, typer.Option(help="Output PNG path.")] = "forecast.png",
    days: Annotated[int, typer.Option(help="Forecast horizon in days (1-14).")] = 14,
    skip_temp_pull: Annotated[bool, typer.Option(help="Skip NWIS water-temp pull (use cached data).")] = False,
    json_out: Annotated[Optional[str], typer.Option("--json", help="Also write per-reach scores to a JSON file.")] = None,
) -> None:
    """Produce a colored PNG map of best-day RELATIVE SUITABILITY INDEX for
    a species over a forecast window, fusing USGS BRT priors x species
    thermal niche x NWS forecast x USGS water-temp observations.

    NOTE: the output is NOT a calibrated catch probability. It is a
    relative ranking in [0, 1]. See docs/forecast_analysis.md.
    """
    paths, _ = _bootstrap()
    from datetime import date
    from angler_ai.features import open_store
    from angler_ai.ingest.noaa_forecast import fetch_daily_forecast
    from angler_ai.ingest.nwis_water_temp import ingest_water_temp_for_huc8
    from angler_ai.prediction.forecast_scoring import max_score_over_window
    from angler_ai.prediction.map_render import ScoredReach, render_scored_reaches_png
    from angler_ai.prediction.species_priors import species_priors_for_geometry
    from angler_ai.prediction.temperature import resolve_many
    from angler_ai.prediction.thermal_niches import get_niche

    with open_store(paths.feature_store) as store:
        store.initialize_schema()
        conn = store.connect()

        # Resolve species
        row = conn.execute(
            "SELECT scientific_name, common_name FROM brt_species WHERE scientific_name = ?",
            [species],
        ).fetchone()
        if row is None:
            row = conn.execute(
                "SELECT scientific_name, common_name FROM brt_species WHERE LOWER(common_name) = ?",
                [species.lower()],
            ).fetchone()
        if row is None:
            console.print(f"[red]Species not found:[/] {species!r}")
            raise typer.Exit(code=2)
        sci, common = row[0], row[1]

        # HUC8 centroid for the NWS forecast point.
        centroid_row = conn.execute(
            """
            SELECT AVG(ST_X(ST_Centroid(geometry))) AS lon,
                   AVG(ST_Y(ST_Centroid(geometry))) AS lat,
                   COUNT(*) AS n
            FROM reaches WHERE huc8 = ?
            """,
            [huc8],
        ).fetchone()
        if centroid_row is None or centroid_row[2] == 0:
            console.print(f"[red]No reaches loaded for HUC8 {huc8}.[/]")
            console.print(f"Run `angler-ai ingest --source nhdplus --state <ST> --huc8 {huc8}` first.")
            raise typer.Exit(code=2)
        lon, lat, n_reaches = float(centroid_row[0]), float(centroid_row[1]), int(centroid_row[2])
        console.print(f"[dim]HUC8 {huc8} centroid: {lat:.3f}, {lon:.3f} ({n_reaches} reaches loaded)[/]")

        # NWS forecast at the centroid.
        console.print(f"[dim]Fetching NWS daily forecast at {lat:.3f},{lon:.3f}...[/]")
        forecast_window = fetch_daily_forecast(lat=lat, lon=lon, days=days)
        if not forecast_window:
            console.print("[yellow]NWS forecast unavailable; flow factor will be 1.0 (neutral).[/]")
        else:
            n_real = sum(1 for f in forecast_window if f.source == "NWS_NDFD_daily")
            n_proj = len(forecast_window) - n_real
            console.print(f"[dim]Forecast window: {n_real} NWS days + {n_proj} persistence-projected[/]")

        # Pull NWIS water-temp for the HUC8 (real observations). Skippable.
        if not skip_temp_pull:
            console.print(f"[dim]Pulling NWIS water-temperature observations for HUC8 {huc8}...[/]")
            n_temp = ingest_water_temp_for_huc8(store, huc8=huc8, lookback_days=14)
            console.print(f"[dim]NWIS_obs water-temp rows written: {n_temp}[/]")

        # BRT priors per reach for the species in this HUC8.
        rows = species_priors_for_geometry(store, species_scientific=sci, huc8=huc8)
        if not rows:
            console.print(f"[red]No BRT priors for {sci} in HUC8 {huc8}.[/]")
            console.print("Species may not be in BRT v2.0 native distribution for this watershed.")
            raise typer.Exit(code=1)
        console.print(f"[dim]BRT priors loaded: {len(rows)} reaches[/]")

        # Resolve temperature (uses 'NWIS_obs' rows we just wrote, or
        # not_modeled if none).
        comids = [sp.comid for sp, _ in rows]
        temperatures = resolve_many(store, comids)
        n_temp_resolved = sum(
            1 for c in comids if temperatures[c].source != "not_modeled"
        )

        # Project per-day water temps and score.
        from angler_ai.prediction.water_temp_model import project_daily_temps
        daily_air_high = {
            f.forecast_date: float(f.high_c) for f in (forecast_window or []) if f.high_c is not None
        }
        per_day_temps_all = project_daily_temps(
            store, comids=comids, daily_air_high_c=daily_air_high, huc8=huc8,
        ) if daily_air_high else {}

        scored: list[ScoredReach] = []
        for sp, geom_wkt in rows:
            per_day_temps_for_reach = {
                d: pt for (c, d), pt in per_day_temps_all.items() if c == sp.comid
            }
            best = max_score_over_window(
                prior=sp,
                per_day_temps=per_day_temps_for_reach,
                forecast_window=forecast_window or [],
            )
            if best is None:
                from angler_ai.prediction.forecast_scoring import score_reach_daily
                best = score_reach_daily(
                    prior=sp,
                    daily_temp=None,
                    daily_forecast=None,
                    daily_discharge_factor=None,
                    gauge_anomaly=None,
                    score_date=date.today(),
                )
            scored.append(ScoredReach(
                comid=sp.comid,
                geometry_wkt=geom_wkt,
                score=best,
            ))

        # Niche citation for the caption.
        niche = get_niche(sci)
        niche_citation = niche.citation if niche else "no thermal niche in registry (factor=1.0)"

        date_range = (
            f"{forecast_window[0].forecast_date} to {forecast_window[-1].forecast_date}"
            if forecast_window else "static (no forecast)"
        )
        title = f"Best-day suitability index: {common or sci} (HUC8 {huc8})"
        subtitle = (
            f"USGS BRT v2.0 prior x thermal suitability x NWS precip forecast"
            f"  |  window: {date_range}"
        )
        caption_lines = [
            f"Sources: USGS BRT v2.0 (DOI 10.5066/P1UV25FW) | hyperstability beta=0.23 (Charbonneau 2025 TAFS)",
            f"Thermal niche: {niche_citation}",
            f"Forecast: NOAA NWS api.weather.gov daily at HUC8 centroid {lat:.3f},{lon:.3f}",
            f"Water temp: {n_temp_resolved}/{len(comids)} reaches have NWIS_obs records (others scored thermal=1.0 neutral)",
        ]
        caption = " | ".join(caption_lines)

        out_path = render_scored_reaches_png(
            scored=scored,
            output_path=out,
            title=title,
            subtitle=subtitle,
            caption=caption,
        )
        console.print(f"[green]Wrote map:[/] {out_path}")

        if json_out:
            import json as _json
            payload = {
                "species_scientific": sci,
                "species_common": common,
                "huc8": huc8,
                "forecast_window": {
                    "lat": lat, "lon": lon,
                    "start": forecast_window[0].forecast_date.isoformat() if forecast_window else None,
                    "end": forecast_window[-1].forecast_date.isoformat() if forecast_window else None,
                    "n_days": len(forecast_window),
                    "n_real_nws_days": sum(1 for f in forecast_window if f.source == "NWS_NDFD_daily"),
                },
                "n_reaches": len(scored),
                "n_temperature_resolved": n_temp_resolved,
                "sources": {
                    "brt": "USGS_BRT_V2.0 (DOI 10.5066/P1UV25FW)",
                    "thermal_niche": niche_citation,
                    "hyperstability": "Charbonneau et al. 2025 TAFS 154(4):339, beta=0.23",
                    "water_temp": "NWIS_obs via api.waterdata.usgs.gov (parameter 00010)",
                    "forecast": "NOAA NWS api.weather.gov",
                },
                "top_reaches": [
                    {
                        "comid": s.comid,
                        "suitability_index": s.score.suitability_index,
                        "suitability_lower": s.score.suitability_lower,
                        "suitability_upper": s.score.suitability_upper,
                        "interval_confidence": s.score.interval_confidence,
                        "score_date": s.score.score_date.isoformat(),
                        "factors": {
                            "base_p": s.score.factors.base_p,
                            "thermal_factor": s.score.factors.thermal_factor,
                            "thermal_source": s.score.factors.thermal_source,
                            "thermal_input_c": s.score.factors.thermal_input_c,
                            "flow_factor": s.score.factors.flow_factor,
                            "flow_source": s.score.factors.flow_source,
                            "precip_probability_pct": s.score.factors.precip_probability_pct,
                            "seasonal_factor": s.score.factors.seasonal_factor,
                            "seasonal_source": s.score.factors.seasonal_source,
                            "anomaly_factor": s.score.factors.anomaly_factor,
                            "anomaly_source": s.score.factors.anomaly_source,
                        },
                    }
                    for s in sorted(scored, key=lambda r: r.score.suitability_index, reverse=True)[:50]
                ],
            }
            Path(json_out).write_text(_json.dumps(payload, indent=2), encoding="utf-8")
            console.print(f"[green]Wrote scores JSON:[/] {json_out}")


@app.command()
def ask(
    query: Annotated[str, typer.Argument(help="Natural-language query.")],
    explain: Annotated[bool, typer.Option("--explain", help="Include the Analyst tool-call chain.")] = False,
    state: Annotated[Optional[str], typer.Option(help="State filter (overrides Profile output).")] = None,
    huc8: Annotated[Optional[str], typer.Option(help="HUC8 filter (overrides Profile output).")] = None,
    model_id: Annotated[Optional[str], typer.Option(help="Installed model id to use.")] = None,
) -> None:
    """Run the MARSHA-style 3-agent reasoning pipeline (M6, FR-7.x).

    Loads the locally-installed llama.cpp model, calls Profile -> Planning ->
    Analyst over the real DuckDB feature store. The Analyst's narrative is
    grounded in real tool outputs (BRT priors, ATTAINS, stocking, regs,
    HydroGEM anomaly) - no fabricated values reach the user.
    """
    paths, hp = _bootstrap()
    from angler_ai.features import open_store
    from angler_ai.inference.runtime import InferenceRuntime
    from angler_ai.reasoning.agents import (
        AnalystAgent, LLMRunner, PlanningAgent, ProfileAgent,
    )
    from angler_ai.registry import load_catalog, read_manifest

    manifest = read_manifest(paths.manifest)
    installed = manifest.get("models", {})
    if not installed:
        console.print("[red]No models installed.[/] Run `angler-ai pull-models --profile small`.")
        raise typer.Exit(code=2)

    catalog = {m.id: m for m in load_catalog()}
    text_ids = {m.id for m in load_catalog() if m.type.value == "text"}

    target_key: str
    if model_id:
        matches = [k for k in installed if k.startswith(f"{model_id}::")]
        if not matches:
            console.print(f"[red]Model {model_id!r} not installed.[/]")
            raise typer.Exit(code=2)
        target_key = matches[0]
    else:
        keys = sorted(k for k in installed if k.split("::", 1)[0] in text_ids)
        if not keys:
            console.print("[red]No installed text model.[/]")
            raise typer.Exit(code=2)
        target_key = keys[0]

    entry = installed[target_key]
    mid, quant_name = target_key.split("::", 1)
    model_entry = catalog[mid]
    quant_entry = next(q for q in model_entry.quants if q.name == quant_name)
    local_path = Path(entry["local_path"])

    runtime = InferenceRuntime()
    loaded = runtime.ensure_loaded(model_entry, quant_entry, local_path, n_ctx=4096)
    llm = LLMRunner(handle=loaded.handle)
    console.print(f"[dim]Using model: {mid}/{quant_name}[/]")

    with open_store(paths.feature_store) as store:
        store.initialize_schema()
        profile = ProfileAgent(llm).parse(query)
        plan = PlanningAgent().plan(
            profile,
            default_huc8=huc8 or "02050206",
            default_state=state or profile.state or "PA",
        )
        response = AnalystAgent(llm, store).respond(plan, original_query=query)

    console.print()
    console.print("[bold]Plan:[/]", f"intent={plan.intent}", f"species={plan.species_scientific or plan.species_common}",
                  f"state={plan.state}", f"huc8={plan.huc8}")
    console.print()
    console.print(response.narrative)
    console.print()
    if response.citations:
        console.print("[bold]Citations:[/]")
        for c in response.citations:
            console.print(f"  - {c}")

    if explain:
        console.print()
        console.print("[bold]Tool calls[/] (--explain):")
        for r in response.tool_call_log:
            status = "[red]ERROR[/]" if r.error else "[green]ok[/]"
            console.print(f"  [{status}] {r.name}({r.args})")
            if r.error:
                console.print(f"     -> error: {r.error}")
            else:
                console.print(f"     -> {r.result_summary}")


if __name__ == "__main__":
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)
