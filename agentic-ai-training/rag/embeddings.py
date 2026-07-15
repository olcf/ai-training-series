"""Embedding helpers shared by indexing and retrieval scripts."""

import json
import re
from pathlib import Path

from rag.logging_utils import verbose_step


@verbose_step(
    "Builds a model-specific Chroma collection name to avoid embedding "
    "collisions."
)
def build_model_collection_name(base: str, model_name: str) -> str:
    model_slug = re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_").lower()
    return f"{base}__{model_slug}"


@verbose_step("Loads the sentence-transformer used to embed chunks and queries.")
def load_embedder(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "This script requires 'sentence-transformers'. "
            "Install it with: pip install sentence-transformers"
        ) from exc

    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        try:
            return SentenceTransformer(model_name)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load embedding model '{model_name}'. "
                "If you're offline, make sure the model has already been "
                "cached locally."
            ) from exc


@verbose_step("Streams chunk records from JSONL so they can be embedded in batches.")
def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_num}: {exc}") from exc
