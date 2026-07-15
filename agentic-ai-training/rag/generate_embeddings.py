#!/usr/bin/env python3
"""Tutorial step 2: embed chunk records and store them in ChromaDB."""

import argparse
import json
import re
from pathlib import Path
from typing import Any

from config import (
    DEFAULT_CHROMA_BATCH_SIZE,
    DEFAULT_CHROMA_COLLECTION,
    DEFAULT_CHROMA_PATH,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OUTPUT_FILE,
)
from rag.embeddings import (
    build_model_collection_name,
    iter_jsonl,
    load_embedder,
)
from rag.logging_utils import log_stage, log_substep, set_verbose
from rag.vector_store import get_chroma_collection, sanitize_for_chroma


## Build CLI arguments.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate embeddings from chunks JSONL and upsert to Chroma."
    )
    parser.add_argument(
        "--chunks-file",
        default=str(DEFAULT_OUTPUT_FILE),
        help="Input chunks JSONL path",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Hugging Face sentence-transformers model name",
    )
    parser.add_argument(
        "--chroma-path",
        default=str(DEFAULT_CHROMA_PATH),
        help="Persistent directory for ChromaDB",
    )
    parser.add_argument(
        "--chroma-collection",
        default=None,
        help=(
            "Target Chroma collection name. If omitted, a model-specific "
            "collection is used automatically."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_CHROMA_BATCH_SIZE,
        help="Batch size for embedding/upsert",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print function-level tracing with a one-line summary for each step",
    )
    parser.add_argument(
        "--odo",
        default=False,
        action='store_true',
        help="Pass if running on Odo, sets default embedding model location."
    )

    parser.add_argument(
        "--frontier",
        default=False,
        action='store_true',
        help="Pass if running on Frontier, sets default embedding model location."
    )
    return parser


def run(args: argparse.Namespace) -> None:
    set_verbose(getattr(args, "verbose", False))
    chunks_file = Path(args.chunks_file)
    if not chunks_file.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_file}")

    if args.odo:
        args.embedding_model = "/gpfs/wolf2/olcf/stf007/world-shared/agentic-ai-training/nomic-embed-text-v2-moe"

    if args.frontier:
        args.embedding_model = "/lustre/orion/stf007/world-shared/agentic-ai-training/nomic-embed-text-v2-moe"

    log_stage("2/3", "Starting embedding pipeline")
    log_substep("2/3", f"Chunks file: {chunks_file}")
    log_substep("2/3", f"Models used in this step: embedding={args.embedding_model}")

    embedder = load_embedder(args.embedding_model)
    collection_name = args.chroma_collection or build_model_collection_name(
        DEFAULT_CHROMA_COLLECTION, args.embedding_model
    )
    log_substep("2/3", f"Target Chroma collection: {collection_name}")
    collection = get_chroma_collection(Path(args.chroma_path), collection_name)

    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict[str, str | int | float | bool]] = []
    total = 0
    batch_size = max(1, args.batch_size)

    def flush_batch() -> None:
        nonlocal total
        if not ids:
            return
        batch_count = len(ids)
        vectors = embedder.encode(docs, convert_to_numpy=True)
        embeddings = vectors.tolist()
        collection.upsert(
            ids=ids,
            documents=docs,
            metadatas=metas,
            embeddings=embeddings,
        )
        total += batch_count
        log_substep("2/3", f"Upserted batch of {batch_count} chunk embeddings")
        ids.clear()
        docs.clear()
        metas.clear()

    for rec in iter_jsonl(chunks_file):
        chunk_id = rec.get("id")
        chunk_text = rec.get("chunk_text")
        if not isinstance(chunk_id, str) or not chunk_id:
            continue
        if not isinstance(chunk_text, str) or not chunk_text.strip():
            continue

        meta: dict[str, Any] = {
            "id": chunk_id,
            "source": rec.get("source"),
            "relative_source": rec.get("relative_source"),
            "chunk_index": rec.get("chunk_index"),
            "embedding_model": args.embedding_model,
        }
        if isinstance(rec.get("metadata"), dict):
            for mk, mv in rec["metadata"].items():
                meta[f"meta_{mk}"] = mv

        ids.append(chunk_id)
        docs.append(chunk_text)
        metas.append(sanitize_for_chroma(meta))

        if len(ids) >= batch_size:
            flush_batch()

    flush_batch()
    log_stage(
        "2/3",
        (
            f"Finished embedding {total} chunks into '{collection_name}' at "
            f"{args.chroma_path}"
        ),
    )


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
