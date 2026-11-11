"""Unit tests for Claude and ChatGPT export parsers."""

import json
from pathlib import Path

from chatmem.ingest.chatgpt import parse_chatgpt_export
from chatmem.ingest.claude import parse_claude_export


def test_parse_claude_flat_list(tmp_path: Path) -> None:
    data = [
        {
            "uuid": "claude_conv_1",
            "name": "Testing Claude Parser",
            "created_at": "2024-01-01T12:00:00Z",
            "updated_at": "2024-01-01T12:05:00Z",
            "chat_messages": [
                {
                    "uuid": "m1",
                    "sender": "human",
                    "text": "Hello Claude!",
                    "created_at": "2024-01-01T12:00:00Z",
                },
                {
                    "uuid": "m2",
                    "sender": "assistant",
                    "text": "Hello! How can I help you today?",
                    "created_at": "2024-01-01T12:01:00Z",
                },
            ],
        }
    ]
    file_path = tmp_path / "conversations.json"
    file_path.write_text(json.dumps(data), encoding="utf-8")

    convs = parse_claude_export(file_path)
    assert len(convs) == 1
    assert convs[0].conversation_id == "claude_conv_1"
    assert convs[0].title == "Testing Claude Parser"
    assert len(convs[0].messages) == 2
    assert convs[0].messages[0].role == "user"
    assert convs[0].messages[0].text == "Hello Claude!"
    assert convs[0].messages[1].role == "assistant"


def test_parse_claude_wrapped_dict(tmp_path: Path) -> None:
    """Tests the real-world export format where data is wrapped in {'Count': ..., 'value': [...]}."""
    data = {
        "Count": 1,
        "value": [
            {
                "uuid": "claude_wrapped_1",
                "name": "Wrapped Export",
                "created_at": "2024-02-01T10:00:00Z",
                "updated_at": "2024-02-01T10:10:00Z",
                "chat_messages": [
                    {
                        "uuid": "msg_w1",
                        "sender": "human",
                        "text": "What is Python?",
                        "created_at": "2024-02-01T10:00:00Z",
                    },
                    {
                        "uuid": "msg_w2",
                        "sender": "assistant",
                        "text": "Python is a programming language.",
                        "created_at": "2024-02-01T10:01:00Z",
                    },
                ],
            }
        ],
    }
    file_path = tmp_path / "conversations.json"
    file_path.write_text(json.dumps(data), encoding="utf-8")

    convs = parse_claude_export(file_path)
    assert len(convs) == 1
    assert convs[0].conversation_id == "claude_wrapped_1"
    assert len(convs[0].messages) == 2


def test_parse_claude_content_blocks_when_text_empty(tmp_path: Path) -> None:
    """Tests fallback to structured content blocks when the 'text' field is empty."""
    data = [
        {
            "uuid": "claude_content_blocks",
            "name": "Content Blocks Test",
            "created_at": "2024-03-01T08:00:00Z",
            "chat_messages": [
                {
                    "uuid": "msg_cb1",
                    "sender": "human",
                    "text": "",
                    "content": [
                        {"type": "text", "text": "Extracted from content block"},
                    ],
                    "created_at": "2024-03-01T08:00:00Z",
                },
                {
                    "uuid": "msg_cb2",
                    "sender": "assistant",
                    "text": "Response from assistant",
                    "created_at": "2024-03-01T08:01:00Z",
                },
            ],
        }
    ]
    file_path = tmp_path / "conversations.json"
    file_path.write_text(json.dumps(data), encoding="utf-8")

    convs = parse_claude_export(file_path)
    assert len(convs) == 1
    assert len(convs[0].messages) == 2
    assert convs[0].messages[0].text == "Extracted from content block"


def test_parse_claude_attachment_fallback(tmp_path: Path) -> None:
    """Tests fallback to attachment info when text is empty."""
    data = [
        {
            "uuid": "claude_attachments",
            "name": "Attachments Test",
            "created_at": "2024-04-01T09:00:00Z",
            "chat_messages": [
                {
                    "uuid": "msg_att1",
                    "sender": "human",
                    "text": "",
                    "attachments": [
                        {
                            "file_name": "report.pdf",
                            "extracted_content": "Annual report details here",
                        }
                    ],
                    "created_at": "2024-04-01T09:00:00Z",
                },
                {
                    "uuid": "msg_att2",
                    "sender": "assistant",
                    "text": "I analyzed your annual report.",
                    "created_at": "2024-04-01T09:01:00Z",
                },
            ],
        }
    ]
    file_path = tmp_path / "conversations.json"
    file_path.write_text(json.dumps(data), encoding="utf-8")

    convs = parse_claude_export(file_path)
    assert len(convs) == 1
    assert len(convs[0].messages) == 2
    assert "report.pdf" in convs[0].messages[0].text
    assert "Annual report details here" in convs[0].messages[0].text


def test_parse_sample_claude_export() -> None:
    """Verifies that the synthetic sample export data/sample/claude/conversations.json parses completely."""
    sample_path = Path("data/sample/claude/conversations.json")
    assert sample_path.exists(), "Sample claude export should exist"
    convs = parse_claude_export(sample_path)
    assert len(convs) > 0
    assert all(len(c.messages) > 0 for c in convs)


def test_parse_chatgpt_export_formats(tmp_path: Path) -> None:
    """Tests ChatGPT export format parsing for both list and wrapped dict."""
    sample_chatgpt = [
        {
            "id": "cg_1",
            "title": "ChatGPT Conv",
            "create_time": 1700000000.0,
            "current_node": "node_2",
            "mapping": {
                "node_1": {
                    "id": "node_1",
                    "parent": None,
                    "children": ["node_2"],
                    "message": {
                        "id": "msg_cg1",
                        "author": {"role": "user"},
                        "content": {"parts": ["Hello ChatGPT"]},
                        "create_time": 1700000000.0,
                    },
                },
                "node_2": {
                    "id": "node_2",
                    "parent": "node_1",
                    "children": [],
                    "message": {
                        "id": "msg_cg2",
                        "author": {"role": "assistant"},
                        "content": {"parts": ["Hello human!"]},
                        "create_time": 1700000010.0,
                    },
                },
            },
        }
    ]
    fpath = tmp_path / "chatgpt.json"
    fpath.write_text(json.dumps(sample_chatgpt), encoding="utf-8")
    convs = parse_chatgpt_export(fpath)
    assert len(convs) == 1
    assert convs[0].conversation_id == "cg_1"
    assert len(convs[0].messages) == 2


def test_parse_claude_missing_or_invalid_timestamps(tmp_path: Path) -> None:
    """Verifies that missing or malformed timestamps in Claude exports do not crash."""
    data = [
        {
            "uuid": "claude_no_ts",
            "name": "No Timestamp Conv",
            "created_at": None,
            "updated_at": "invalid-timestamp",
            "chat_messages": [
                {
                    "uuid": "m_no_ts",
                    "sender": "human",
                    "text": "Message with missing ts",
                    "created_at": None,
                }
            ],
        }
    ]
    fpath = tmp_path / "claude_no_ts.json"
    fpath.write_text(json.dumps(data), encoding="utf-8")
    convs = parse_claude_export(fpath)
    assert len(convs) == 1
    assert len(convs[0].messages) == 1
    assert convs[0].messages[0].timestamp is not None


def test_parse_claude_summary_fallback_title(tmp_path: Path) -> None:
    """Verifies that Claude summary is used when title/name is missing."""
    data = [
        {
            "uuid": "claude_summary_title",
            "name": "",
            "summary": "Summary As Title",
            "created_at": "2024-01-01T00:00:00Z",
            "chat_messages": [
                {
                    "uuid": "m_sum",
                    "sender": "human",
                    "text": "Hello with summary title",
                    "created_at": "2024-01-01T00:00:00Z",
                }
            ],
        }
    ]
    fpath = tmp_path / "claude_sum.json"
    fpath.write_text(json.dumps(data), encoding="utf-8")
    convs = parse_claude_export(fpath)
    assert len(convs) == 1
    assert convs[0].title == "Summary As Title"


def test_parse_chatgpt_current_node_fallback(tmp_path: Path) -> None:
    """Verifies that ChatGPT export parses even when current_node is missing/null."""
    data = [
        {
            "id": "cg_no_current_node",
            "title": "Missing Current Node",
            "current_node": None,
            "mapping": {
                "root": {
                    "id": "root",
                    "parent": None,
                    "children": ["n1"],
                    "message": None,
                },
                "n1": {
                    "id": "n1",
                    "parent": "root",
                    "children": [],
                    "message": {
                        "id": "m1",
                        "author": {"role": "user"},
                        "content": {"parts": ["Found from leaf!"]},
                        "create_time": 1700000000.0,
                    },
                },
            },
        }
    ]
    fpath = tmp_path / "cg_no_cn.json"
    fpath.write_text(json.dumps(data), encoding="utf-8")
    convs = parse_chatgpt_export(fpath)
    assert len(convs) == 1
    assert len(convs[0].messages) == 1
    assert convs[0].messages[0].text == "Found from leaf!"


def test_parse_chatgpt_multimodal_and_code_parts(tmp_path: Path) -> None:
    """Verifies handling of dict parts (multimodal) and direct content.text (code interpreter)."""
    data = [
        {
            "id": "cg_multimodal_and_code",
            "title": "Multimodal and Code",
            "current_node": "n2",
            "mapping": {
                "n1": {
                    "id": "n1",
                    "parent": None,
                    "children": ["n2"],
                    "message": {
                        "id": "m1",
                        "author": {"role": "user"},
                        "content": {
                            "content_type": "multimodal_text",
                            "parts": [
                                "What is in this chart?",
                                {"content_type": "image_asset_pointer", "asset_pointer": "file-123"},
                                {"text": "And summarize trends."},
                            ],
                        },
                        "create_time": 1700000000.0,
                    },
                },
                "n2": {
                    "id": "n2",
                    "parent": "n1",
                    "children": [],
                    "message": {
                        "id": "m2",
                        "author": {"role": "tool"},
                        "content": {
                            "content_type": "code",
                            "text": "import pandas as pd\ndf = pd.read_csv('chart.csv')",
                        },
                        "create_time": 1700000010.0,
                    },
                },
            },
        }
    ]
    fpath = tmp_path / "cg_mm.json"
    fpath.write_text(json.dumps(data), encoding="utf-8")
    convs = parse_chatgpt_export(fpath)
    assert len(convs) == 1
    assert len(convs[0].messages) == 2
    assert "What is in this chart?" in convs[0].messages[0].text
    assert "And summarize trends." in convs[0].messages[0].text
    assert convs[0].messages[1].role == "system"
    assert "import pandas as pd" in convs[0].messages[1].text


def test_parse_very_long_message(tmp_path: Path) -> None:
    """Verifies that very long messages (>20,000 characters) parse without error or truncation."""
    long_text = "Log line entry: " + ("x" * 25000)
    data = [
        {
            "uuid": "claude_long_msg",
            "name": "Long Message Test",
            "chat_messages": [
                {
                    "uuid": "m_long",
                    "sender": "human",
                    "text": long_text,
                    "created_at": "2024-01-01T00:00:00Z",
                }
            ],
        }
    ]
    fpath = tmp_path / "claude_long.json"
    fpath.write_text(json.dumps(data), encoding="utf-8")
    convs = parse_claude_export(fpath)
    assert len(convs) == 1
    assert len(convs[0].messages[0].text) == len(long_text)

