"""Shared Chroma helpers for storing and retrieving vectors."""

import json
import re
from pathlib import Path
from typing import Any

from rag.logging_utils import verbose_step


@verbose_step("Converts metadata into Chroma-safe scalar values before storage.")
def sanitize_for_chroma(meta: dict[str, Any]) -> dict[str, str | int | float | bool]:
    clean: dict[str, str | int | float | bool] = {}
    for key, value in meta.items():
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        elif value is None:
            continue
        else:
            clean[key] = json.dumps(value, ensure_ascii=False)
    return clean


@verbose_step("Creates or opens the target Chroma collection for storing vectors.")
def get_chroma_collection(
    chroma_path: Path,
    collection_name: str,
    create: bool = True,
):
    try:
        import chromadb
        from chromadb.errors import NotFoundError
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "This script requires 'chromadb'. Install it with: pip install chromadb"
        ) from exc

    client = chromadb.PersistentClient(path=str(chroma_path))
    if create:
        return client.get_or_create_collection(name=collection_name)

    try:
        return client.get_collection(name=collection_name)
    except NotFoundError:
        raise


def normalize_collection_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", name).lower()


def simplified_model_slug(model_name: str) -> str:
    tail = model_name.split("/")[-1].lower()
    tail = tail.replace("embedding", "")
    return re.sub(r"[^a-z0-9]+", "", tail)


@verbose_step("Checks Chroma collections to find one that matches the embedding model.")
def find_collection_for_embedding_model(
    client,
    available_names: list[str],
    embedding_model: str,
) -> str | None:
    normalized_model = normalize_collection_name(embedding_model)
    slug_hint = simplified_model_slug(embedding_model)

    for name in available_names:
        try:
            collection = client.get_collection(name)
            sample = collection.get(limit=1, include=["metadatas"])
            metadatas = sample.get("metadatas") or []
            first_meta = metadatas[0] if metadatas else None
            stored_model = (first_meta or {}).get("embedding_model")
            if stored_model == embedding_model:
                return name
        except Exception:
            continue

    for name in available_names:
        normalized_name = normalize_collection_name(name)
        if slug_hint and slug_hint in normalized_name:
            return name
        if normalized_model and normalized_model in normalized_name:
            return name

    return None


@verbose_step(
    "Resolves a fallback collection name when the expected Chroma collection is "
    "missing."
)
def resolve_collection_fallback(
    client,
    expected_name: str,
    available_names: list[str],
    embedding_model: str | None,
) -> str | None:
    if expected_name in available_names:
        return expected_name

    if embedding_model:
        metadata_match = find_collection_for_embedding_model(
            client, available_names, embedding_model
        )
        if metadata_match:
            return metadata_match

    if len(available_names) == 1:
        return available_names[0]

    normalized_expected = normalize_collection_name(expected_name)
    normalized_map = {
        name: normalize_collection_name(name) for name in available_names
    }

    for name, normalized_name in normalized_map.items():
        if normalized_name == normalized_expected:
            return name

    prefix_matches = [
        name
        for name, normalized_name in normalized_map.items()
        if normalized_name.startswith(normalized_expected)
        or normalized_expected.startswith(normalized_name)
    ]
    if len(prefix_matches) == 1:
        return prefix_matches[0]

    base_name = expected_name.split("__", 1)[0]
    if base_name in available_names:
        return base_name

    return None


@verbose_step(
    "Opens an existing Chroma collection for retrieval, with fallback matching "
    "when needed."
)
def get_existing_collection(
    chroma_path: Path, collection_name: str, embedding_model: str | None = None
):
    try:
        import chromadb
        from chromadb.errors import NotFoundError
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "This script requires 'chromadb'. Install it with: pip install chromadb"
        ) from exc

    client = chromadb.PersistentClient(path=str(chroma_path))
    try:
        return client.get_collection(name=collection_name)
    except NotFoundError as exc:
        available_names = [
            collection.name for collection in client.list_collections()
        ]
        fallback_name = resolve_collection_fallback(
            client, collection_name, available_names, embedding_model
        )
        if fallback_name:
            print(
                f"Collection '{collection_name}' was not found. "
                f"Using '{fallback_name}' instead."
            )
            return client.get_collection(name=fallback_name)

        available_display = ", ".join(sorted(available_names)) or "none"
        raise RuntimeError(
            f"Collection '{collection_name}' does not exist at {chroma_path}. "
            f"Available collections: {available_display}. "
            "Pass --chroma-collection explicitly or regenerate embeddings."
        ) from exc
