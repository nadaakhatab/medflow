"""Non-destructive audit of the persisted Chroma index against frozen Day 2 config."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict

if __package__ in (None, ""):
    import sys
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from day4 import config
else:
    from . import config


def audit_index(persist_dir: Path | None = None, collection_name: str | None = None) -> Dict[str, Any]:
    persist_dir = Path(persist_dir or config.LIVE_PERSIST_DIR)
    collection_name = collection_name or config.LIVE_COLLECTION_NAME
    db_path = persist_dir / "chroma.sqlite3"
    manifest_path = persist_dir / "frozen_index_manifest.json"
    try:
        display_persist_dir = str(persist_dir.resolve().relative_to(config.PROJECT_ROOT.resolve()))
    except ValueError:
        display_persist_dir = str(persist_dir)
    result: Dict[str, Any] = {
        "persist_directory": display_persist_dir,
        "collection_name": collection_name,
        "expected_indexed_chunks": config.EXPECTED_INDEXED_CHUNKS,
        "database_exists": db_path.exists(),
        "actual_indexed_chunks": None,
        "manifest_exists": manifest_path.exists(),
        "manifest_matches_frozen_day2": None,
        "index_matches_frozen_day2": False,
        "notes": [],
    }
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected_manifest = {
                "collection_name": collection_name,
                "embedding_model": config.EMBEDDING_MODEL_NAME,
                "chunk_size_tokens": config.CHUNK_SIZE_TOKENS,
                "chunk_overlap_tokens": config.CHUNK_OVERLAP_TOKENS,
                "indexed_chunks": config.EXPECTED_INDEXED_CHUNKS,
                "top_k": config.TOP_K,
            }
            result["manifest"] = manifest
            result["manifest_matches_frozen_day2"] = all(manifest.get(k) == v for k, v in expected_manifest.items())
        except Exception as exc:
            result["manifest_matches_frozen_day2"] = False
            result["notes"].append(f"Frozen-index manifest could not be validated: {exc}")
    if not db_path.exists():
        result["notes"].append("No persisted Chroma SQLite database found.")
        return result

    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            """
            SELECT c.name, COUNT(e.id)
            FROM collections c
            LEFT JOIN segments s ON s.collection = c.id
            LEFT JOIN embeddings e ON e.segment_id = s.id
            WHERE c.name = ?
            GROUP BY c.name
            """,
            (collection_name,),
        ).fetchone()
        if row is None:
            # Chroma schemas vary by version; fall back to the common collection->segment path.
            collections = conn.execute("SELECT id, name FROM collections").fetchall()
            match = next((x for x in collections if x[1] == collection_name), None)
            if match:
                segment_ids = [x[0] for x in conn.execute("SELECT id FROM segments WHERE collection = ?", (match[0],)).fetchall()]
                count = 0
                for sid in segment_ids:
                    count += conn.execute("SELECT COUNT(*) FROM embeddings WHERE segment_id = ?", (sid,)).fetchone()[0]
                row = (collection_name, count)
        if row is None:
            result["notes"].append("Requested collection was not found in the persisted database.")
            return result
        actual = int(row[1])
        result["actual_indexed_chunks"] = actual
        count_matches = actual == config.EXPECTED_INDEXED_CHUNKS
        manifest_ok = result["manifest_matches_frozen_day2"]
        # Legacy stores may have no manifest; count-only is then reported as a weaker
        # audit. Separate Day 4 frozen builds include a manifest and require both.
        result["index_matches_frozen_day2"] = count_matches and (manifest_ok is not False)
        if result["index_matches_frozen_day2"]:
            if manifest_ok is True:
                result["notes"].append("Persisted collection count and frozen-index manifest match the Day 2 configuration.")
            else:
                result["notes"].append("Persisted collection count matches Day 2, but no frozen-index manifest is available for stronger provenance verification.")
        else:
            result["notes"].append(
                "Persisted collection does not match the frozen Day 2 chunk count. "
                "Do not claim frozen Day 2 metrics from this live collection without rebuilding/auditing it."
            )
    finally:
        conn.close()
    return result


def save_audit(result: Dict[str, Any], path: Path | None = None) -> Path:
    path = Path(path or (config.RESULTS_DIR / "index_audit.json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    r = audit_index()
    p = save_audit(r)
    print(json.dumps(r, indent=2))
    print(f"Saved: {p}")
