"""Unit tests for chunking strategies: per_message, sliding_window, and qa_pair."""

import pytest
from chatmem.chunking.per_message import chunk_per_message
from chatmem.chunking.qa_pair import qa_chunks
from chatmem.chunking.sliding_window import window_chunks
from chatmem.schema import Conversation, Message


@pytest.fixture
def sample_6msg_conversation() -> Conversation:
    """Fixture providing a hand-made 6-message conversation (3 user/assistant turns)."""
    messages = [
        Message(
            message_id="msg_0",
            conversation_id="conv_1",
            platform="chatgpt",
            role="user",
            title="Python Testing",
            text="How do I write a pytest fixture?",
            timestamp=100.0,
        ),
        Message(
            message_id="msg_1",
            conversation_id="conv_1",
            platform="chatgpt",
            role="assistant",
            title="Python Testing",
            text="Use the @pytest.fixture decorator above your function.",
            timestamp=105.0,
        ),
        Message(
            message_id="msg_2",
            conversation_id="conv_1",
            platform="chatgpt",
            role="user",
            title="Python Testing",
            text="Can fixtures have parameters or scope?",
            timestamp=120.0,
        ),
        Message(
            message_id="msg_3",
            conversation_id="conv_1",
            platform="chatgpt",
            role="assistant",
            title="Python Testing",
            text="Yes, pass scope='module' or params=[...] to the fixture decorator.",
            timestamp=125.0,
        ),
        Message(
            message_id="msg_4",
            conversation_id="conv_1",
            platform="chatgpt",
            role="user",
            title="Python Testing",
            text="How do I clean up resources after tests finish?",
            timestamp=140.0,
        ),
        Message(
            message_id="msg_5",
            conversation_id="conv_1",
            platform="chatgpt",
            role="assistant",
            title="Python Testing",
            text="Use a yield statement inside the fixture; code after yield runs teardown.",
            timestamp=145.0,
        ),
    ]
    return Conversation(
        conversation_id="conv_1",
        platform="chatgpt",
        title="Python Testing",
        creation_time=100.0,
        updation_time=145.0,
        messages=messages,
    )


def test_chunk_per_message_6msg(sample_6msg_conversation: Conversation) -> None:
    chunks = chunk_per_message([sample_6msg_conversation])
    assert len(chunks) == 6
    for i, chunk in enumerate(chunks):
        assert chunk.message_ids == [f"msg_{i}"]
        assert chunk.conversation_id == "conv_1"
        assert chunk.platform == "chatgpt"
        assert chunk.title == "Python Testing"


def test_qa_pair_6msg(sample_6msg_conversation: Conversation) -> None:
    chunks = qa_chunks([sample_6msg_conversation])
    assert len(chunks) == 3

    assert chunks[0].message_ids == ["msg_0", "msg_1"]
    assert "User: How do I write a pytest fixture?" in chunks[0].text
    assert "Assistant: Use the @pytest.fixture" in chunks[0].text

    assert chunks[1].message_ids == ["msg_2", "msg_3"]
    assert "User: Can fixtures have parameters" in chunks[1].text
    assert "Assistant: Yes, pass scope='module'" in chunks[1].text

    assert chunks[2].message_ids == ["msg_4", "msg_5"]
    assert "User: How do I clean up" in chunks[2].text
    assert "Assistant: Use a yield statement" in chunks[2].text


def test_sliding_window_6msg(sample_6msg_conversation: Conversation) -> None:
    chunks = window_chunks([sample_6msg_conversation], window_size=3, overlap=1)
    # Stride is window_size - overlap = 2
    # Window 0: msgs 0, 1, 2
    # Window 1: msgs 2, 3, 4 (overlaps with window 0 on msg_2)
    # Window 2: msgs 4, 5 (overlaps with window 1 on msg_4)
    assert len(chunks) == 3

    assert chunks[0].message_ids == ["msg_0", "msg_1", "msg_2"]
    assert chunks[1].message_ids == ["msg_2", "msg_3", "msg_4"]
    assert chunks[2].message_ids == ["msg_4", "msg_5"]

    # Verify that msg_2 is in both chunk 0 and chunk 1
    assert "msg_2" in chunks[0].message_ids
    assert "msg_2" in chunks[1].message_ids

    # Verify that msg_4 is in both chunk 1 and chunk 2
    assert "msg_4" in chunks[1].message_ids
    assert "msg_4" in chunks[2].message_ids


def test_sliding_window_and_qa_pair_produce_different_chunks(sample_6msg_conversation: Conversation) -> None:
    qa = qa_chunks([sample_6msg_conversation])
    sw = window_chunks([sample_6msg_conversation], window_size=3, overlap=1)

    # Even though both produce 3 chunks for a 6-message conversation,
    # their message allocations and overlapping behaviors are fundamentally different:
    assert len(qa) == len(sw) == 3

    # Chunk 0: QA has 2 messages (user, assistant); SW has 3 messages (user, assistant, user)
    assert qa[0].message_ids == ["msg_0", "msg_1"]
    assert sw[0].message_ids == ["msg_0", "msg_1", "msg_2"]
    assert qa[0].message_ids != sw[0].message_ids

    # Chunk 1: QA has 2 messages; SW has 3 messages, overlapping on msg_2
    assert qa[1].message_ids == ["msg_2", "msg_3"]
    assert sw[1].message_ids == ["msg_2", "msg_3", "msg_4"]
    assert qa[1].message_ids != sw[1].message_ids

    # Chunk 2: QA has the final 2 messages; SW reaches the conversation tail (min(4+3, 6)=6)
    assert qa[2].message_ids == ["msg_4", "msg_5"]
    assert sw[2].message_ids == ["msg_4", "msg_5"]

    # In SW, the sum of message lengths across chunks is 3 + 3 + 2 = 8 due to overlap.
    # In QA, the sum of message lengths across chunks is 2 + 2 + 2 = 6 (partitioned without overlap).
    assert sum(len(c.message_ids) for c in sw) == 8
    assert sum(len(c.message_ids) for c in qa) == 6


def test_chunking_odd_message_count() -> None:
    """When message count is odd (5 messages), qa_pair and sliding_window differ in count."""
    msgs = [
        Message(
            message_id=f"m_{i}",
            conversation_id="conv_odd",
            platform="chatgpt",
            role="user" if i % 2 == 0 else "assistant",
            title="Odd Conv",
            text=f"Text {i}",
            timestamp=float(i),
        )
        for i in range(5)
    ]
    conv = Conversation(
        conversation_id="conv_odd",
        platform="chatgpt",
        title="Odd Conv",
        creation_time=0.0,
        updation_time=4.0,
        messages=msgs,
    )

    # qa_pair creates 3 chunks: [m0, m1], [m2, m3], [m4]
    qa = qa_chunks([conv])
    assert len(qa) == 3
    assert qa[2].message_ids == ["m_4"]

    # sliding_window creates 2 chunks: [m0, m1, m2] (start 0, end 3), [m2, m3, m4] (start 2, end 5 >= 5)
    sw = window_chunks([conv], window_size=3, overlap=1)
    assert len(sw) == 2
    assert sw[0].message_ids == ["m_0", "m_1", "m_2"]
    assert sw[1].message_ids == ["m_2", "m_3", "m_4"]
