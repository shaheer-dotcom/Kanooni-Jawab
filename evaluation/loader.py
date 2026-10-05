from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import Config
from jurisdiction_registry import get_jurisdiction_key, normalize_jurisdiction


@dataclass
class GoldenItem:
    id: str
    query: str
    jurisdiction: str
    expected_docs: list[str] = field(default_factory=list)
    expected_parent_ids: list[str] = field(default_factory=list)
    doc_type: str | None = None
    must_cite: bool = False
    tags: list[str] = field(default_factory=list)


@dataclass
class GoldenSet:
    jurisdiction: str
    registry_key: str
    items: list[GoldenItem]
    source_file: str


def _parse_item(raw: dict[str, Any], default_jurisdiction: str) -> GoldenItem:
    jurisdiction = raw.get("jurisdiction") or default_jurisdiction
    return GoldenItem(
        id=str(raw.get("id", "")).strip(),
        query=str(raw.get("query", "")).strip(),
        jurisdiction=normalize_jurisdiction(jurisdiction),
        expected_docs=[str(d).strip() for d in raw.get("expected_docs", []) if str(d).strip()],
        expected_parent_ids=[
            str(p).strip() for p in raw.get("expected_parent_ids", []) if str(p).strip()
        ],
        doc_type=raw.get("doc_type"),
        must_cite=bool(raw.get("must_cite", False)),
        tags=[str(t).strip() for t in raw.get("tags", []) if str(t).strip()],
    )


def load_golden_file(path: Path) -> GoldenSet:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)

    default_jurisdiction = normalize_jurisdiction(
        str(data.get("jurisdiction", path.stem.replace("_", " ").title()))
    )
    jkey = get_jurisdiction_key(default_jurisdiction) or path.stem.lower()
    items = [
        _parse_item(item, default_jurisdiction)
        for item in data.get("items", [])
        if isinstance(item, dict) and item.get("query")
    ]
    return GoldenSet(
        jurisdiction=default_jurisdiction,
        registry_key=jkey,
        items=items,
        source_file=str(path),
    )


def discover_golden_sets(golden_dir: str | None = None) -> list[GoldenSet]:
    root = Path(golden_dir or Config.EVAL_GOLDEN_DIR)
    if not root.is_dir():
        return []
    sets: list[GoldenSet] = []
    for path in sorted(root.glob("*.json")):
        sets.append(load_golden_file(path))
    return sets


def load_golden_sets(
  jurisdiction_keys: list[str] | None = None,
  golden_dir: str | None = None,
) -> list[GoldenSet]:
    sets = discover_golden_sets(golden_dir)
    if not jurisdiction_keys:
        return sets
    wanted = {k.lower().replace(" ", "_") for k in jurisdiction_keys}
    return [s for s in sets if s.registry_key.lower() in wanted]
