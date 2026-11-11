from __future__ import annotations

import dataclasses
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer

logger = logging.getLogger("chatmem")

app = typer.Typer(help="chatmem: cross-chatbot memory retrieval")


@app.callback()
def main_callback(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose DEBUG logging.",
    ),
) -> None:
    """Global configuration callback for logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


# ============================================================================
# 1. Index Command
# ============================================================================

@app.command()
def index(
    input: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="A raw export file or a folder of exports.",
    ),
    platform: str = typer.Option(
        "auto",
        "--platform",
        "-p",
        help="Platform type: chatgpt, claude, or auto.",
    ),
    strategy: str = typer.Option(
        "all",
        "--strategy",
        "-s",
        help="Chunking strategy: pm, qa, ws, or all.",
    ),
    db: Optional[Path] = typer.Option(
        None,
        "--db",
        help="Optional override of the SQLite path.",
    ),
    skip_dense: bool = typer.Option(
        False,
        "--skip-dense",
        help="Build SQLite only, skip embedding.",
    ),
) -> None:
    """Index conversations into SQLite and Qdrant.

    To rebuild from scratch, delete data/chatmem_metadata.db and drop the Qdrant collections first.
    """
    input_path = Path(input)
    if not input_path.exists():
        typer.echo(f"Error: Input path '{input_path}' does not exist.", err=True)
        raise typer.Exit(code=2)

    # Ingest conversations
    from chatmem.ingest.normalize import load_all_exports, load_platform_export

    conversations = []
    if input_path.is_dir():
        if platform == "auto":
            conversations = load_all_exports(input_path)
        elif platform in ("chatgpt", "claude"):
            conv_file = input_path / "conversations.json"
            if not conv_file.exists():
                typer.echo(
                    f"Error: No conversations.json found in directory '{input_path}' for platform '{platform}'.",
                    err=True,
                )
                raise typer.Exit(code=2)
            conversations = load_platform_export(platform, conv_file)
        else:
            typer.echo(f"Error: Unsupported platform '{platform}'.", err=True)
            raise typer.Exit(code=2)
    else:
        # Input path is a single file
        if platform == "auto" or not platform:
            typer.echo("A single file needs --platform chatgpt|claude.", err=True)
            raise typer.Exit(code=2)
        if platform not in ("chatgpt", "claude"):
            typer.echo(f"Error: Unsupported platform '{platform}'.", err=True)
            raise typer.Exit(code=2)
        conversations = load_platform_export(platform, input_path)

    if not conversations:
        typer.echo("Warning: Zero conversations parsed. Nothing to index.", err=True)
        raise typer.Exit(code=1)

    # Initialize store and save conversations once
    from chatmem.index.store import DEFAULT_DB_PATH, SQLiteMetadataStore

    target_db = Path(db) if db else DEFAULT_DB_PATH
    store = SQLiteMetadataStore(target_db)
    store.save_conversations(conversations)

    # Determine strategies
    strat_choice = strategy.lower().strip()
    if strat_choice == "all":
        strategies = ["pm", "qa", "ws"]
    elif strat_choice in ("pm", "qa", "ws"):
        strategies = [strat_choice]
    else:
        typer.echo(f"Error: Unknown strategy '{strategy}'. Supported: pm, qa, ws, all.", err=True)
        raise typer.Exit(code=2)

    from chatmem.chunking.per_message import chunk_per_message
    from chatmem.chunking.qa_pair import qa_chunks
    from chatmem.chunking.sliding_window import window_chunks

    chunker_map = {
        "pm": chunk_per_message,
        "qa": qa_chunks,
        "ws": window_chunks,
    }

    chunks_written: dict[str, int | str] = {}
    vectors_upserted: dict[str, int | str] = {}
    failed_strategies: list[str] = []
    skipped_conversations = 0

    start_time = time.perf_counter()

    for strat in strategies:
        chunker = chunker_map[strat]
        chunks = []
        try:
            chunks = chunker(conversations)
        except Exception as e:
            # Try per-conversation chunking to isolate and report single-conversation failures
            salvaged = []
            for conv in conversations:
                try:
                    c_chunks = chunker([conv])
                    salvaged.extend(c_chunks)
                except Exception as conv_err:
                    logger.warning(
                        "Skipping conversation '%s' due to chunking error: %s",
                        getattr(conv, "conversation_id", "unknown"),
                        conv_err,
                    )
                    skipped_conversations += 1
            if salvaged:
                chunks = salvaged
            else:
                logger.error("Failed to chunk for strategy '%s': %s", strat, e)
                chunks_written[strat] = "failed"
                vectors_upserted[strat] = "failed"
                failed_strategies.append(strat)
                continue

        try:
            saved_count = store.save_chunks(chunks, strategy=strat)
            chunks_written[strat] = saved_count
        except Exception as e:
            logger.error("Failed to save chunks to SQLite for strategy '%s': %s", strat, e)
            chunks_written[strat] = "failed"
            vectors_upserted[strat] = "failed"
            failed_strategies.append(strat)
            continue

        if skip_dense:
            vectors_upserted[strat] = "skipped"
        else:
            try:
                from chatmem.index.dense import DenseIndex
                from chatmem.index.store import DEFAULT_LOCAL_QDRANT_URL
                from chatmem.retrieval.pipeline import STRATEGY_COLLECTIONS

                collection = STRATEGY_COLLECTIONS[strat]
                dense_idx = DenseIndex(collection_name=collection)
                upserted_count = dense_idx.index_chunks(chunks)
                vectors_upserted[strat] = upserted_count
            except Exception as e:
                err_str = str(e).lower()
                qdrant_url = os.getenv("QDRANT_URL") or DEFAULT_LOCAL_QDRANT_URL
                if any(k in err_str for k in ("connection", "connect", "unreachable", "refused", "timeout", "socket")):
                    typer.echo(
                        f"Qdrant is not reachable at {qdrant_url}. Start it (docker) or re-run with --skip-dense.",
                        err=True,
                    )
                    raise typer.Exit(code=1)
                logger.error("Failed to upsert vectors into Qdrant for strategy '%s': %s", strat, e)
                vectors_upserted[strat] = "failed"
                failed_strategies.append(strat)

    elapsed = time.perf_counter() - start_time
    skipped_str = f"skipped={skipped_conversations}" if skipped_conversations > 0 else "skipped=unknown"

    typer.echo("Index Summary:")
    typer.echo(f"  Conversations loaded: {len(conversations)} ({skipped_str})")
    typer.echo(f"  Chunks written per strategy: {chunks_written}")
    typer.echo(f"  Vectors upserted per strategy: {vectors_upserted}")
    typer.echo(f"  Elapsed time: {elapsed:.2f}s")
    typer.echo(f"  Database path: {target_db}")

    if failed_strategies:
        raise typer.Exit(code=1)


# ============================================================================
# 2. Search Command
# ============================================================================

@app.command()
def search(
    query: str = typer.Argument(
        ...,
        help="Search query string.",
    ),
    config: Path = typer.Option(
        Path("configs/default.yaml"),
        "--config",
        "-c",
        help="Path to pipeline YAML configuration file.",
    ),
    top_k: Optional[int] = typer.Option(
        None,
        "--top-k",
        "-k",
        help="Optional override for final_top_k.",
    ),
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Print PipelineResult as JSON instead of formatted text.",
    ),
    db: Optional[Path] = typer.Option(
        None,
        "--db",
        help="Optional override of the SQLite path.",
    ),
) -> None:
    """Search indexed conversations using a multi-stage retrieval pipeline."""
    if not query or not query.strip():
        typer.echo("Error: Query cannot be empty or whitespace-only.", err=True)
        raise typer.Exit(code=2)

    config_path = Path(config)
    if not config_path.exists():
        typer.echo(f"Config file not found: {config_path}", err=True)
        raise typer.Exit(code=1)

    from chatmem.index.store import DEFAULT_DB_PATH

    target_db = Path(db) if db else DEFAULT_DB_PATH
    if not target_db.exists():
        typer.echo(f"No index found at {target_db}. Run `chatmem index` first.", err=True)
        raise typer.Exit(code=1)

    from chatmem.retrieval.pipeline import PipelineConfig, build_retriever

    try:
        cfg = PipelineConfig.from_yaml(config_path)
    except Exception as e:
        typer.echo(f"Invalid configuration file: {e}", err=True)
        raise typer.Exit(code=1)

    if top_k is not None:
        cfg.final_top_k = top_k
        if cfg.use_rerank and cfg.rerank_candidates < cfg.final_top_k:
            cfg.rerank_candidates = cfg.final_top_k

    retriever = build_retriever(cfg, db_path=target_db)
    try:
        with retriever as r:
            result = r.search(query)
    finally:
        if hasattr(retriever, "close"):
            retriever.close()

    if as_json:
        typer.echo(json.dumps(dataclasses.asdict(result), indent=2))
        return

    status_str = "INJECTED" if result.injected else "NOT INJECTED"
    if not result.injected:
        typer.echo("--- NOT INJECTED ---")

    typer.echo(f"Pipeline Result ({status_str}):")
    typer.echo(f"  Injected: {result.injected}")
    typer.echo(f"  Reason: {result.reason}")
    typer.echo(f"  Top score: {result.top_score}")
    if hasattr(result, "score_scale") and result.score_scale:
        typer.echo(f"  Score scale: {result.score_scale}")
    typer.echo(f"  Timings: {result.timings_ms}")

    typer.echo(f"\nRetrieved Chunks ({len(result.chunks)}):")
    for rank, chunk in enumerate(result.chunks, start=1):
        preview = chunk.text.replace("\n", " ")[:200]
        typer.echo(
            f"[{rank}] Score: {chunk.score:.4f} | Platform: {chunk.platform} | Title: {chunk.title or 'Untitled'} | Conv: {chunk.conversation_id}"
        )
        typer.echo(f"    Preview: {preview}")


# ============================================================================
# 3. Eval Command
# ============================================================================

@app.command()
def eval(
    queries: Path = typer.Option(
        Path("data/eval/queries.sample.jsonl"),
        "--queries",
        "-q",
        help="Path to labeled evaluation queries JSONL file.",
    ),
    configs: Optional[list[Path]] = typer.Option(
        None,
        "--configs",
        "-c",
        help="Pipeline configuration YAML files to evaluate (default: configs/eval/*.yaml).",
    ),
    out: Optional[Path] = typer.Option(
        None,
        "--out",
        "-o",
        help="Output directory path for evaluation artifacts.",
    ),
    ks: str = typer.Option(
        "1,3,5,10",
        "--ks",
        help="Comma-separated cutoffs for k (e.g. 1,3,5,10).",
    ),
    split: str = typer.Option(
        "all",
        "--split",
        help="Query split to evaluate: dev, test, or all (default: all).",
    ),
) -> None:
    """Evaluate retrieval pipelines against benchmark queries."""
    queries_path = Path(queries)
    if not queries_path.exists():
        typer.echo(f"Queries file not found: {queries_path}", err=True)
        raise typer.Exit(code=2)

    config_paths = configs
    if not config_paths:
        eval_configs_dir = Path("configs/eval")
        if not eval_configs_dir.exists():
            typer.echo("Default configs directory 'configs/eval' not found.", err=True)
            raise typer.Exit(code=2)
        config_paths = sorted(list(eval_configs_dir.glob("*.yaml")))

    if not config_paths:
        typer.echo("No configuration YAML files provided or found.", err=True)
        raise typer.Exit(code=2)

    for cp in config_paths:
        if not Path(cp).exists():
            typer.echo(f"Config file not found: {cp}", err=True)
            raise typer.Exit(code=2)

    try:
        k_values = tuple(int(x.strip()) for x in ks.split(",") if x.strip())
    except Exception:
        typer.echo(f"Invalid ks parameter: '{ks}'. Must be comma-separated integers.", err=True)
        raise typer.Exit(code=2)

    from chatmem.eval.run_eval import run_sweep

    if out is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_out_dir = Path("eval_results") / timestamp
    else:
        target_out_dir = Path(out)

    try:
        results = run_sweep(
            config_paths=config_paths,
            queries_path=queries_path,
            ks=k_values,
            out_dir=target_out_dir,
            split=split,
        )
    except ValueError as err:
        typer.echo(f"Invalid evaluation input or queries: {err}", err=True)
        raise typer.Exit(code=2)
    except Exception as err:
        typer.echo(f"Evaluation error: {err}", err=True)
        raise typer.Exit(code=2)

    summary_file = target_out_dir / "summary.md"
    if summary_file.exists():
        typer.echo(summary_file.read_text(encoding="utf-8"))

    gate_file = target_out_dir / "gate_report.md"
    if gate_file.exists():
        typer.echo(gate_file.read_text(encoding="utf-8"))

    typer.echo(f"\nArtifacts written to: {target_out_dir}")

    has_issues = any(res.errors or res.warnings for res in results)
    if has_issues:
        raise typer.Exit(code=1)


# ============================================================================
# 4. Serve Command (Stub)
# ============================================================================

@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Host address to bind to."),
    port: int = typer.Option(8000, "--port", help="Port number to bind to."),
) -> None:
    """Start local API server for memory retrieval (stub)."""
    try:
        from chatmem.api.main import app as fastapi_app  # type: ignore

        if fastapi_app is None or not hasattr(fastapi_app, "routes"):
            raise ImportError("FastAPI app is not defined")
        import uvicorn

        uvicorn.run(fastapi_app, host=host, port=port)
    except (ImportError, AttributeError):
        typer.echo("api not implemented yet")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()

