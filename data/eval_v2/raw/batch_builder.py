"""Detailed conversation generator for batches 2 to 10."""
from __future__ import annotations

def make_turn(role: str, text: str) -> dict:
    return {"role": role, "text": text}

def make_dialogue(turns: list[tuple[str, str]]) -> list[dict]:
    return [make_turn(r, t) for r, t in turns]

