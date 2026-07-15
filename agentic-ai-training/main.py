#!/usr/bin/env python3
"""Tutorial entrypoint for chunking, embedding, and retrieval/chat steps."""

import argparse

from config import (
    DEFAULT_CHROMA_BATCH_SIZE,
    DEFAULT_CHROMA_COLLECTION,
    DEFAULT_CHROMA_PATH,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MANIFEST_FILE,
    DEFAULT_OUTPUT_FILE,
    DEFAULT_SOURCE_DIR,
    DEFAULT_STORE_MODE,
)
from rag.create_contextual_chunks import run as run_chunking
from rag.generate_embeddings import run as run_embedding
from rag.query_chroma import run as run_query
from rag.logging_utils import log_stage, log_function, set_verbose


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one tutorial step at a time or use a single entrypoint for the full "
            "RAG pipeline."
        )
    )
    parser.add_argument(
        "--step",
        choices=("chunk", "embed", "query", "all"),
        required=True,
        help="Which tutorial step to run",
    )

    parser.add_argument(
        "--source-dir",
        default=str(DEFAULT_SOURCE_DIR),
        help="Directory containing source documents",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_FILE),
        help="Output JSONL file path for chunks",
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST_FILE),
        help="Optional JSONL metadata manifest path",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Chunk size in characters",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Chunk overlap in characters",
    )
    parser.add_argument(
        "--store",
        choices=("jsonl", "chroma", "both"),
        default=DEFAULT_STORE_MODE,
        help="Where chunking output should be stored",
    )
    parser.add_argument(
        "--chroma-path",
        default=str(DEFAULT_CHROMA_PATH),
        help="Persistent directory for ChromaDB",
    )
    parser.add_argument(
        "--chroma-collection",
        default=None,
        help="Optional explicit Chroma collection name",
    )
    parser.add_argument(
        "--chroma-batch-size",
        type=int,
        default=DEFAULT_CHROMA_BATCH_SIZE,
        help="Batch size for chunk upserts into Chroma",
    )

    parser.add_argument(
        "--chunks-file",
        default=str(DEFAULT_OUTPUT_FILE),
        help="Input chunks JSONL path for embedding",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Embedding model used for indexing and retrieval",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_CHROMA_BATCH_SIZE,
        help="Batch size for embedding/upsert",
    )

    parser.add_argument("query", nargs="?", help="Natural-language query text")
    parser.add_argument(
        "--query",
        dest="query_flag",
        default=None,
        help="Natural-language query text",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieved chunks to include in query step",
    )
    parser.add_argument(
        "--show-chars",
        type=int,
        default=400,
        help="Characters to show per retrieved chunk preview",
    )
    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Print retrieved chunk previews before each answer",
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


def main() -> None:
    args = build_parser().parse_args()
    set_verbose(args.verbose)

    if args.step == "chunk":
        log_function(
            "run_chunking",
            "Runs the document preparation stage that creates retrieval-ready chunks.",
        )
        run_chunking(args)
        return

    if args.step == "embed":
        log_function(
            "run_embedding",
            "Runs the vectorization stage that turns chunks into "
            "searchable embeddings.",
        )
        run_embedding(args)
        return

    if args.step == "query":
        log_function(
            "run_query",
            "Runs the retrieval and grounded answer stage for user questions.",
        )
        run_query(args)
        return

    log_stage("main", "Running full tutorial pipeline: chunk -> embed -> query")
    log_function(
        "run_chunking",
        "Runs the document preparation stage that creates retrieval-ready chunks.",
    )
    run_chunking(args)
    log_function(
        "run_embedding",
        "Runs the vectorization stage that turns chunks into searchable embeddings.",
    )
    run_embedding(args)
    log_function(
        "run_query",
        "Runs the retrieval and grounded answer stage for user questions.",
    )
    run_query(args)


if __name__ == "__main__":
    main()
