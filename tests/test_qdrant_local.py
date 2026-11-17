"""Tests for embedded local Qdrant mode, locking, and incremental sync."""
from __future__ import annotations

from pathlib import Path
import pytest

from chatmem.chunking.qa_pair import qa_chunks
from chatmem.index.dense import DenseIndex, get_qdrant_client
from chatmem.index.store import QdrantStore, SQLiteMetadataStore
from chatmem.ingest.normalize import load_all_exports
from chatmem.schema import Chunk


def test_local_mode_lock_error_and_reopen(tmp_path: Path) -> None:
    """Tests single-process locking error message and clean reopen after close."""
    storage_path = tmp_path / "qdrant_lock_test"

    # 1. Open first client
    client1 = get_qdrant_client(mode="local", path=storage_path)
    assert client1 is not None

    # 2. Attempting to open second client concurrently must raise RuntimeError with descriptive message
    with pytest.raises(RuntimeError) as exc_info:
        get_qdrant_client(mode="local", path=storage_path)
    err_msg = str(exc_info.value)
    assert "locked by another process" in err_msg or "already accessed" in err_msg

    # 3. Close first client
    client1.close()

    # 4. Opening again must succeed cleanly
    client2 = get_qdrant_client(mode="local", path=storage_path)
    assert client2 is not None
    client2.close()


def test_local_mode_lifecycle_and_filtering(tmp_path: Path) -> None:
    """Verifies collection creation, upsert, search, filtering, idempotent reindexing, and deletion."""
    storage_path = tmp_path / "qdrant_lifecycle"
    sample_convs = load_all_exports("data/sample")
    assert len(sample_convs) > 0

    all_chunks = qa_chunks(sample_convs)
    assert len(all_chunks) > 0

    with DenseIndex(
        path=storage_path,
        collection_name="test_lifecycle",
        mode="local",
    ) as dense_idx:
        # Create & Upsert
        initial_count = dense_idx.index_chunks(all_chunks)
        assert initial_count == len(all_chunks)
        assert dense_idx.count_points() == len(all_chunks)

        # Idempotent re-indexing: indexing again must not duplicate points
        second_count = dense_idx.index_chunks(all_chunks)
        assert second_count == len(all_chunks)
        assert dense_idx.count_points() == len(all_chunks)

        # Search without filter
        results_unfiltered = dense_idx.search("python programming", top_k=5)
        assert len(results_unfiltered) > 0

        # Search with platform filter
        results_claude = dense_idx.search("python programming", top_k=5, platform_filter="claude")
        assert len(results_claude) > 0
        assert all(c.platform == "claude" for c, _ in results_claude)

        results_chatgpt = dense_idx.search("python programming", top_k=5, platform_filter="chatgpt")
        assert len(results_chatgpt) > 0
        assert all(c.platform == "chatgpt" for c, _ in results_chatgpt)

        # Delete by conversation id
        target_cid = all_chunks[0].conversation_id
        chunks_for_cid = [c for c in all_chunks if c.conversation_id == target_cid]
        expected_remaining = len(all_chunks) - len(chunks_for_cid)

        dense_idx.delete_conversation(target_cid)
        assert dense_idx.count_points() == expected_remaining


def test_sqlite_and_qdrant_incremental_consistency(tmp_path: Path) -> None:
    """Tests consistency between SQLiteMetadataStore and QdrantStore across add, update, and delete."""
    db_path = tmp_path / "test_metadata.db"
    qdrant_path = tmp_path / "test_qdrant_sync"

    sqlite_store = SQLiteMetadataStore(db_path=db_path)
    sqlite_store.init_db()

    with QdrantStore(path=qdrant_path, mode="local") as qdrant_store:
        qdrant_store.ensure_collection("sync_col", vector_size=384, force_recreate=True)

        with DenseIndex(
            store=qdrant_store,
            collection_name="sync_col",
        ) as dense_idx:
            sample_convs = load_all_exports("data/sample")
            conv1 = sample_convs[0]
            chunks_v1 = qa_chunks([conv1])
            assert len(chunks_v1) >= 1

            # 1. Add chunks (Incremental upsert)
            saved_sqlite = sqlite_store.save_chunks(chunks_v1, strategy="qa")
            saved_qdrant = dense_idx.upsert_chunks(chunks_v1)
            assert saved_sqlite == len(chunks_v1)
            assert saved_qdrant == len(chunks_v1)
            assert qdrant_store.count_points("sync_col") == len(chunks_v1)

            with sqlite_store._get_connection() as conn:
                count_sq = conn.execute(
                    "SELECT COUNT(*) as cnt FROM chunks WHERE conversation_id = ? AND strategy = 'qa';",
                    (conv1.conversation_id,),
                ).fetchone()["cnt"]
            assert count_sq == len(chunks_v1)
            assert count_sq == qdrant_store.count_points("sync_col")

            # 2. Update chunks (more chunks added or updated)
            # Simulate updated conversation with duplicate chunk plus an additional chunk
            extra_chunk = Chunk(
                chunk_id=f"{conv1.conversation_id}_extra",
                conversation_id=conv1.conversation_id,
                platform=conv1.platform,
                title=conv1.title,
                text="Extra follow up answer",
                message_ids=["extra_1"],
                time=chunks_v1[0].time,
            )
            chunks_v2 = chunks_v1 + [extra_chunk]

            sqlite_store.save_chunks(chunks_v2, strategy="qa")
            dense_idx.upsert_chunks(chunks_v2)

            with sqlite_store._get_connection() as conn:
                count_sq_v2 = conn.execute(
                    "SELECT COUNT(*) as cnt FROM chunks WHERE conversation_id = ? AND strategy = 'qa';",
                    (conv1.conversation_id,),
                ).fetchone()["cnt"]
            count_qd_v2 = qdrant_store.count_points("sync_col")
            assert count_sq_v2 == len(chunks_v2)
            assert count_qd_v2 == len(chunks_v2)
            assert count_sq_v2 == count_qd_v2

            # 3. Delete by conversation id
            sqlite_store.delete_chunks_by_conversation(conv1.conversation_id, strategy="qa")
            dense_idx.delete_conversation(conv1.conversation_id)

            with sqlite_store._get_connection() as conn:
                count_sq_after = conn.execute(
                    "SELECT COUNT(*) as cnt FROM chunks WHERE conversation_id = ? AND strategy = 'qa';",
                    (conv1.conversation_id,),
                ).fetchone()["cnt"]
            count_qd_after = qdrant_store.count_points("sync_col")
            assert count_sq_after == 0
            assert count_qd_after == 0
