"""Dataset generator for chatmem eval_v2.

Assembles:
- data/eval_v2/raw/batch_01.json .. batch_10.json (100 conversations total)
- data/eval_v2/raw/queries_authored.jsonl (100 queries total, stratified dev/test with seed=1234)
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path

from chatmem.ingest.normalize import load_all_exports
import data.eval_v2.raw.gen_b2_b4 as b2_b4
import data.eval_v2.raw.gen_b5_b7 as b5_b7
import data.eval_v2.raw.gen_b8_b10 as b8_b10
from data.eval_v2.raw.queries_data import QUERIES


def get_base_timestamp(idx: int) -> float:
    """Generate created timestamps spread across 16 months (2023-01-15 to 2024-05-20)."""
    # Base timestamp: 2023-01-15 09:00:00 UTC = 1673773200
    # Step: ~5 days = 432000 seconds per conversation
    return 1673773200.0 + idx * 432000.0


def extract_sample_conversations() -> dict[str, dict]:
    convs = load_all_exports("data/sample")
    facts_map = {
        "c01": ["pandas_groupby_keyerror"],
        "c02": ["pytorch_cuda_out_of_memory", "gpu_out_of_memory_management"],
        "c03": ["docker_qdrant_storage"],
        "c04": ["reciprocal_rank_fusion_k_constant", "rag_chat_original_retrieval", "rag_chat_retrieval_architecture"],
        "c05": ["paneer_butter_masala_recipe"],
        "c06": ["delhi_jaipur_itinerary"],
        "c07": ["resume_ml_bullets", "tech_career_negotiation_growth"],
        "c08": ["fastapi_cors_chrome_extension", "fastapi_production_patterns"],
        "c09": ["chunking_chat_transcripts", "rag_chat_original_retrieval", "rag_chat_retrieval_architecture"],
        "c10": ["binary_search_bounds", "algorithmic_interview_patterns"],
        "c11": ["git_rebase_vs_merge", "git_history_manipulation"],
        "c12": ["uv_python_virtualenv_setup", "python_packaging_ecosystem"],
        "c13": ["guitar_beginner_chords", "guitar_fretboard_techniques"],
        "c14": ["embedding_model_bge_small_base", "rag_chat_original_retrieval", "rag_chat_retrieval_architecture"],
    }
    result = {}
    for c in convs:
        cid = c.conversation_id
        messages = [{"role": m.role, "text": m.text} for m in c.messages]
        created = c.creation_time
        if isinstance(created, datetime):
            created_val = created.isoformat()
        elif isinstance(created, (int, float)):
            created_val = float(created)
        else:
            created_val = str(created) if created else 1673773200.0
        result[cid] = {
            "id": cid,
            "platform": c.platform,
            "title": c.title,
            "created": created_val,
            "facts": facts_map.get(cid, []),
            "messages": messages,
        }
    return result


def build_all_conversations() -> list[dict]:
    samples = extract_sample_conversations()

    # Meta definitions for c15 to c100: (id, platform, title, facts, messages)
    defs = [
        # Batch 2: c15..c20
        ("c15", "chatgpt", "Poetry dependency management and lockfile", ["poetry_lockfile_resolution", "python_packaging_ecosystem"], b2_b4.c15_msgs),
        ("c16", "claude", "Conda environment export without builds", ["conda_env_export_no_builds"], b2_b4.c16_msgs),
        ("c17", "chatgpt", "uv cache clean and venv activation", ["uv_cache_clean_prune", "python_packaging_ecosystem"], b2_b4.c17_msgs),
        ("c18", "claude", "pyproject.toml build-system backend", ["pyproject_build_system_hatchling_flit", "python_packaging_ecosystem"], b2_b4.c18_msgs),
        ("c19", "chatgpt", "PyTorch device selection cuda vs mps", ["pytorch_device_cuda_mps"], b2_b4.c19_msgs),
        ("c20", "claude", "CUDA OOM gradient checkpointing and cloud costs", ["cuda_gradient_checkpointing_memory", "gpu_out_of_memory_management"], b2_b4.c20_msgs),

        # Batch 3: c21..c30
        ("c21", "chatgpt", "PyTorch mixed precision fp16 vs bf16", ["pytorch_mixed_precision_fp16_bf16", "gpu_out_of_memory_management"], b2_b4.c21_msgs),
        ("c22", "claude", "PyTorch DataLoader workers and pin_memory", ["pytorch_dataloader_workers_pin_memory"], b2_b4.c22_msgs),
        ("c23", "chatgpt", "Qdrant compound payload filtering", ["qdrant_payload_filter_conditions"], b2_b4.c23_msgs),
        ("c24", "claude", "Hybrid search with BM25 and Qdrant fusion", ["hybrid_search_bm25_qdrant_fusion", "rag_chat_retrieval_architecture"], b2_b4.c24_msgs),
        ("c25", "chatgpt", "Cross-encoder reranking with BAAI bge-reranker", ["cross_encoder_reranker_bge_base", "rag_chat_retrieval_architecture"], b2_b4.c25_msgs),
        ("c26", "claude", "Pandas merge vs join on multiple columns", ["pandas_merge_vs_join_columns"], b2_b4.c26_msgs),
        ("c27", "chatgpt", "Polars LazyFrame query optimization and posture", ["polars_lazyframe_predicate_pushdown"], b2_b4.c27_msgs),
        ("c28", "claude", "SQLite FTS5 bm25 full text search", ["sqlite_fts5_bm25_ranking", "sqlite_duckdb_tabular_analytics"], b2_b4.c28_msgs),
        ("c29", "chatgpt", "DuckDB querying partitioned parquet files", ["duckdb_partitioned_parquet_hive", "sqlite_duckdb_tabular_analytics"], b2_b4.c29_msgs),
        ("c30", "claude", "FastAPI BackgroundTasks vs Celery", ["fastapi_background_tasks_vs_celery", "fastapi_production_patterns"], b2_b4.c30_msgs),

        # Batch 4: c31..c40
        ("c31", "chatgpt", "Pydantic v2 migration and validators", ["pydantic_v2_field_model_validator"], b2_b4.c31_msgs),
        ("c32", "claude", "Django ORM select_related vs prefetch_related", ["django_orm_select_vs_prefetch_related"], b2_b4.c32_msgs),
        ("c33", "chatgpt", "FastAPI JWT Bearer auth header", ["fastapi_jwt_bearer_authentication"], b2_b4.c33_msgs),
        ("c34", "claude", "Git cherry-pick merge conflict resolution", ["git_cherry_pick_resolve_continue", "git_history_manipulation"], b2_b4.c34_msgs),
        ("c35", "chatgpt", "Git stash pop vs apply", ["git_stash_pop_vs_apply"], b2_b4.c35_msgs),
        ("c36", "claude", "Git reflog recover deleted branch", ["git_reflog_recover_branch", "git_history_manipulation"], b2_b4.c36_msgs),
        ("c37", "chatgpt", "Git submodule recursive init and update", ["git_submodule_update_recursive"], b2_b4.c37_msgs),
        ("c38", "claude", "React useEffect cleanup function", ["react_useeffect_cleanup_unmount", "react_lifecycle_state_pitfalls"], b2_b4.c38_msgs),
        ("c39", "chatgpt", "React useState stale closure in setInterval", ["react_usestate_stale_closure_interval", "react_lifecycle_state_pitfalls"], b2_b4.c39_msgs),
        ("c40", "claude", "Next.js App Router server vs client components", ["nextjs_app_router_server_vs_client", "react_lifecycle_state_pitfalls"], b2_b4.c40_msgs),

        # Batch 5: c41..c50
        ("c41", "chatgpt", "Tailwind CSS responsive grid and guitar calluses", ["tailwind_grid_responsive_columns"], b5_b7.c41_msgs),
        ("c42", "claude", "Authentic Dal Makhani slow cooker recipe", ["dal_makhani_slow_cooker_recipe"], b5_b7.c42_msgs),
        ("c43", "chatgpt", "Sourdough bread bulk fermentation troubleshooting", ["sourdough_bulk_fermentation_stretch_folds", "cooking_fermentation_techniques"], b5_b7.c43_msgs),
        ("c44", "claude", "Crispy baked tofu with cornstarch", ["crispy_baked_tofu_cornstarch"], b5_b7.c44_msgs),
        ("c45", "chatgpt", "Pizza dough cold fermentation and career switch", ["pizza_dough_cold_fermentation_enzymes", "cooking_fermentation_techniques"], b5_b7.c45_msgs),
        ("c46", "claude", "Manali to Leh road trip acclimatization", ["manali_leh_acclimatization_itinerary"], b5_b7.c46_msgs),
        ("c47", "chatgpt", "Tokyo 5-day subway pass and neighborhood plan", ["tokyo_subway_72_hour_itinerary", "japan_travel_transit_planning"], b5_b7.c47_msgs),
        ("c48", "claude", "Monsoon trekking packing checklist", ["monsoon_trekking_packing_checklist"], b5_b7.c48_msgs),
        ("c49", "chatgpt", "Budget trip to Vietnam and knapsack problem", ["vietnam_budget_hanoi_da_nang"], b5_b7.c49_msgs),
        ("c50", "claude", "STAR method behavioral interview answers", ["star_method_behavioral_interview", "tech_career_negotiation_growth"], b5_b7.c50_msgs),

        # Batch 6: c51..c60
        ("c51", "chatgpt", "Negotiating software engineer salary and equity", ["negotiate_software_engineer_equity_base", "tech_career_negotiation_growth"], b5_b7.c51_msgs),
        ("c52", "claude", "Cover letter transition to AI engineering", ["cover_letter_transition_ai_engineer"], b5_b7.c52_msgs),
        ("c53", "chatgpt", "System design interview preparation roadmap", ["system_design_interview_preparation_roadmap"], b5_b7.c53_msgs),
        ("c54", "claude", "Emergency fund in High-Yield Savings Account", ["emergency_fund_hysa_liquidity"], b5_b7.c54_msgs),
        ("c55", "chatgpt", "Index fund ETF vs mutual fund dollar cost averaging", ["index_fund_etf_voo_vs_vfiax"], b5_b7.c55_msgs),
        ("c56", "claude", "Backdoor Roth IRA and pro-rata rule", ["backdoor_roth_ira_pro_rata_rule", "retirement_accounts_tax_optimization"], b5_b7.c56_msgs),
        ("c57", "chatgpt", "Credit score utilization ratio rules", ["credit_score_utilization_ratio"], b5_b7.c57_msgs),
        ("c58", "claude", "Term vs whole life insurance and garden beds", ["term_vs_whole_life_insurance"], b5_b7.c58_msgs),
        ("c59", "chatgpt", "Couch to 5k running and shin splints prevention", ["couch_to_5k_shin_splints_cadence"], b5_b7.c59_msgs),
        ("c60", "claude", "Barbell bench press shoulder pain and scapula retraction", ["bench_press_shoulder_scapular_retraction", "strength_training_form_cues"], b5_b7.c60_msgs),

        # Batch 7: c61..c70
        ("c61", "chatgpt", "Daily protein intake for muscle building", ["daily_protein_intake_muscle_hypertrophy"], b5_b7.c61_msgs),
        ("c62", "claude", "Intermittent fasting 16:8 schedule", ["intermittent_fasting_16_8_morning_workout"], b5_b7.c62_msgs),
        ("c63", "chatgpt", "Stretching tight hip flexors from desk work", ["tight_hip_flexor_stretches_sitting"], b5_b7.c63_msgs),
        ("c64", "claude", "Guitar barre chords F chord technique", ["guitar_f_major_barre_chord_technique", "guitar_fretboard_techniques"], b5_b7.c64_msgs),
        ("c65", "chatgpt", "Minor pentatonic box 1 blues solo phrasing", ["minor_pentatonic_box_1_solo_phrasing", "guitar_fretboard_techniques"], b5_b7.c65_msgs),
        ("c66", "claude", "Acoustic guitar Travis picking pattern", ["acoustic_guitar_travis_picking_pattern"], b5_b7.c66_msgs),
        ("c67", "chatgpt", "Guitar Drop D tuning and power chords", ["guitar_drop_d_tuning_benefits", "guitar_fretboard_techniques"], b5_b7.c67_msgs),
        ("c68", "claude", "Breadth-first search vs depth-first search", ["bfs_vs_dfs_tree_graph_use_cases", "algorithmic_interview_patterns"], b5_b7.c68_msgs),
        ("c69", "chatgpt", "Two-pointer technique for sorted two sum", ["two_pointer_two_sum_sorted_array", "algorithmic_interview_patterns"], b5_b7.c69_msgs),
        ("c70", "claude", "LRU Cache with doubly linked list and hash map", ["lru_cache_doubly_linked_list_hashmap", "algorithmic_interview_patterns"], b5_b7.c70_msgs),

        # Batch 8: c71..c80
        ("c71", "chatgpt", "Dynamic programming coin change problem", ["dynamic_programming_coin_change_minimum", "algorithmic_interview_patterns"], b8_b10.c71_msgs),
        ("c72", "claude", "Kitchen faucet cartridge replacement", ["kitchen_faucet_cartridge_replacement", "plumbing_drain_faucet_fixes"], b8_b10.c72_msgs),
        ("c73", "chatgpt", "Unclogging sink with baking soda and vinegar", ["unclog_sink_baking_soda_vinegar", "plumbing_drain_faucet_fixes"], b8_b10.c73_msgs),
        ("c74", "claude", "Drywall patching small hole with spackle", ["patch_drywall_doorknob_hole_spackle"], b8_b10.c74_msgs),
        ("c75", "chatgpt", "Indoor snake plant watering and light", ["snake_plant_watering_neglect_light"], b8_b10.c75_msgs),
        ("c76", "claude", "Succulent propagation from stem cuttings", ["succulent_stem_cutting_callous_propagation"], b8_b10.c76_msgs),
        ("c77", "chatgpt", "Circuit breaker tripping and smart plug python script", ["circuit_breaker_tripping_space_heater"], b8_b10.c77_msgs),
        ("c78", "claude", "Pip build failure missing Python C headers", ["pip_wheel_build_missing_python_h_headers"], b8_b10.c78_msgs),
        ("c79", "chatgpt", "Hatchling vs Flit build backends", ["hatchling_vs_flit_build_backend", "python_packaging_ecosystem"], b8_b10.c79_msgs),
        ("c80", "claude", "PyTorch DistributedDataParallel single-node multi-GPU", ["pytorch_ddp_torchrun_process_group"], b8_b10.c80_msgs),

        # Batch 9: c81..c90
        ("c81", "chatgpt", "vLLM PagedAttention and continuous batching", ["vllm_paged_attention_continuous_batching"], b8_b10.c81_msgs),
        ("c82", "claude", "Markdown chunking and overlap for technical documentation", ["rag_chunk_overlap_markdown_documentation"], b8_b10.c82_msgs),
        ("c83", "chatgpt", "Qdrant snapshot creation and recovery", ["qdrant_collection_snapshot_recovery"], b8_b10.c83_msgs),
        ("c84", "claude", "Pandas categorical dtype memory optimization", ["pandas_categorical_dtype_memory_reduction"], b8_b10.c84_msgs),
        ("c85", "chatgpt", "DuckDB window functions running total", ["duckdb_window_functions_running_total", "sqlite_duckdb_tabular_analytics"], b8_b10.c85_msgs),
        ("c86", "claude", "FastAPI request execution timing middleware", ["fastapi_timing_middleware_process_time", "fastapi_production_patterns"], b8_b10.c86_msgs),
        ("c87", "chatgpt", "Pydantic v2 Settings reading .env file", ["pydantic_settings_env_file_configuration"], b8_b10.c87_msgs),
        ("c88", "claude", "Git worktree add for urgent hotfixes", ["git_worktree_hotfix_without_stashing"], b8_b10.c88_msgs),
        ("c89", "chatgpt", "Git soft reset squash last N commits", ["git_reset_soft_squash_commits", "git_history_manipulation"], b8_b10.c89_msgs),
        ("c90", "claude", "Creamy tomato basil pasta sauce recipe", ["creamy_tomato_basil_pasta_sauce"], b8_b10.c90_msgs),

        # Batch 10: c91..c100
        ("c91", "chatgpt", "French press cold brew coffee ratio", ["french_press_cold_brew_ratio_steep_time"], b8_b10.c91_msgs),
        ("c92", "claude", "Kyoto 2-day temple walking itinerary", ["kyoto_2_day_walking_itinerary", "japan_travel_transit_planning"], b8_b10.c92_msgs),
        ("c93", "chatgpt", "One bag travel packing for Europe", ["one_bag_travel_europe_packing_system"], b8_b10.c93_msgs),
        ("c94", "claude", "Engineering manager 1-on-1 meeting questions", ["engineering_manager_1_on_1_questions"], b8_b10.c94_msgs),
        ("c95", "chatgpt", "Promotion packet self-evaluation impact statement", ["promotion_packet_impact_statement_metrics", "tech_career_negotiation_growth"], b8_b10.c95_msgs),
        ("c96", "claude", "401k employer match vesting cliff vs graded", ["401k_employer_match_cliff_graded_vesting", "retirement_accounts_tax_optimization"], b8_b10.c96_msgs),
        ("c97", "chatgpt", "HSA triple tax advantage stealth IRA", ["hsa_triple_tax_advantage_stealth_ira", "retirement_accounts_tax_optimization"], b8_b10.c97_msgs),
        ("c98", "claude", "Barbell deadlift back rounding cues", ["barbell_deadlift_back_rounding_armpit_cue", "strength_training_form_cues"], b8_b10.c98_msgs),
        ("c99", "chatgpt", "Zone 2 cardio heart rate calculation", ["zone_2_cardio_heart_rate_talk_test"], b8_b10.c99_msgs),
        ("c100", "claude", "Bathtub silicone caulk removal and replacement", ["bathtub_caulking_silicone_sealant_removal"], b8_b10.c100_msgs),
    ]

    all_convs = []
    # c01 to c10
    for i in range(1, 11):
        cid = f"c{i:02d}"
        all_convs.append(samples[cid])

    # c11 to c14
    for i in range(11, 15):
        cid = f"c{i:02d}"
        all_convs.append(samples[cid])

    # c15 to c100
    for idx, (cid, platform, title, facts, turns) in enumerate(defs, start=15):
        created_ts = get_base_timestamp(idx)
        msgs = [{"role": r, "text": t} for r, t in turns]
        all_convs.append({
            "id": cid,
            "platform": platform,
            "title": title,
            "created": created_ts,
            "facts": facts,
            "messages": msgs,
        })

    return all_convs


def assign_splits(queries: list[dict], seed: int = 1234) -> list[dict]:
    """Stratified 50/50 split assignment per query type with fixed seed."""
    rng = random.Random(seed)
    by_type: dict[str, list[dict]] = {}
    for q in queries:
        by_type.setdefault(q["type"], []).append(q)

    result = []
    for qtype, group in sorted(by_type.items()):
        # Deterministic shuffle
        indices = list(range(len(group)))
        rng.shuffle(indices)
        half = len(group) // 2
        for rank, idx in enumerate(indices):
            q_copy = dict(group[idx])
            q_copy["split"] = "dev" if rank < half else "test"
            result.append(q_copy)

    # Sort back by qid
    result.sort(key=lambda x: x["qid"])
    return result


def main():
    raw_dir = Path("data/eval_v2/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    conversations = build_all_conversations()
    assert len(conversations) == 100, f"Expected 100 conversations, got {len(conversations)}"

    # Write 10 batches of 10 conversations
    for b_idx in range(10):
        batch_num = b_idx + 1
        batch_convs = conversations[b_idx * 10 : (b_idx + 1) * 10]
        batch_file = raw_dir / f"batch_{batch_num:02d}.json"
        with batch_file.open("w", encoding="utf-8") as f:
            json.dump(batch_convs, f, indent=2)
        print(f"Wrote {len(batch_convs)} conversations to {batch_file}")

    # Process and write queries
    queries_with_splits = assign_splits(QUERIES, seed=1234)
    assert len(queries_with_splits) == 100, f"Expected 100 queries, got {len(queries_with_splits)}"

    queries_file = raw_dir / "queries_authored.jsonl"
    with queries_file.open("w", encoding="utf-8") as f:
        for q in queries_with_splits:
            f.write(json.dumps(q) + "\n")
    print(f"Wrote {len(queries_with_splits)} queries to {queries_file}")


if __name__ == "__main__":
    main()
