"""Shared helpers for the clustering robustness pilot."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def resolve_from_root(path: str | Path) -> Path:
    """Resolve a relative path from the repository root; preserve absolute paths."""
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()


def read_json(path: str | Path) -> Any:
    resolved = resolve_from_root(path)
    with resolved.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(data: Any, path: str | Path) -> Path:
    resolved = resolve_from_root(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    return resolved
