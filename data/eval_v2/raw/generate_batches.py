"""Batch and conversation generator for eval_v2."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from chatmem.ingest.normalize import load_all_exports


def get_sample_convs() -> dict[str, dict]:
    convs = load_all_exports("data/sample")
    facts_map = {
        "c01": ["pandas_groupby_keyerror"],
        "c02": ["pytorch_cuda_out_of_memory"],
        "c03": ["docker_qdrant_storage"],
        "c04": ["reciprocal_rank_fusion_k_constant", "rag_chat_retrieval_architecture"],
        "c05": ["paneer_butter_masala_recipe"],
        "c06": ["delhi_jaipur_itinerary"],
        "c07": ["resume_ml_bullets"],
        "c08": ["fastapi_cors_chrome_extension"],
        "c09": ["chunking_chat_transcripts", "rag_chat_retrieval_architecture"],
        "c10": ["binary_search_bounds"],
        "c11": ["git_rebase_vs_merge"],
        "c12": ["uv_python_virtualenv_setup"],
        "c13": ["guitar_beginner_chords"],
        "c14": ["embedding_model_bge_small_base", "rag_chat_retrieval_architecture"],
    }
    result = {}
    for c in convs:
        cid = c.conversation_id
        messages = [{"role": m.role, "text": m.text} for m in c.messages]
        result[cid] = {
            "id": cid,
            "platform": c.platform,
            "title": c.title,
            "created": c.creation_time or 1672531200.0,
            "facts": facts_map.get(cid, []),
            "messages": messages,
        }
    return result

