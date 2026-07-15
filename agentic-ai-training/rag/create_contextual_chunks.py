#!/usr/bin/env python3
"""Tutorial step 1: chunk source documents into JSONL records."""

import argparse
from pathlib import Path

from config import (
    DEFAULT_CHROMA_BATCH_SIZE,
    DEFAULT_CHROMA_COLLECTION,
    DEFAULT_CHROMA_PATH,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MANIFEST_FILE,
    DEFAULT_OUTPUT_FILE,
    DEFAULT_SOURCE_DIR,
    DEFAULT_STORE_MODE,
)
from rag.chunking import process_sources
from rag.logging_utils import log_stage, log_substep, set_verbose


## Define CLI arguments for input/output paths and chunking options.
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Create document chunks from source files.")
    p.add_argument(
        "--source-dir",
        default=str(DEFAULT_SOURCE_DIR),
        help="Directory containing source documents",
    )
    p.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_FILE),
        help="Output JSONL file path",
    )
    p.add_argument(
        "--manifest",
        default=None,
        help=(
            "Optional JSONL metadata file "
            "(default: config.DEFAULT_MANIFEST_FILE or <source-dir>/manifest.jsonl)"
        ),
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Chunk size in characters",
    )
    p.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Chunk overlap in characters",
    )
    p.add_argument(
        "--store",
        choices=("jsonl", "chroma", "both"),
        default=DEFAULT_STORE_MODE,
        help="Where to store output chunks",
    )
    p.add_argument(
        "--chroma-path",
        default=str(DEFAULT_CHROMA_PATH),
        help="Persistent directory for ChromaDB",
    )
    p.add_argument(
        "--chroma-collection",
        default=DEFAULT_CHROMA_COLLECTION,
        help="Chroma collection name for chunks",
    )
    p.add_argument(
        "--chroma-batch-size",
        type=int,
        default=DEFAULT_CHROMA_BATCH_SIZE,
        help="Batch size for Chroma upserts",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print function-level tracing with a one-line summary for each step",
    )
    return p


def run(args: argparse.Namespace) -> None:
    set_verbose(getattr(args, "verbose", False))
    source_dir = Path(args.source_dir)
    output_file = Path(args.output)

    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"Invalid source directory: {source_dir}")

    if args.manifest:
        manifest_path = Path(args.manifest)
    elif DEFAULT_MANIFEST_FILE.exists():
        manifest_path = DEFAULT_MANIFEST_FILE
    else:
        manifest_path = source_dir / "manifest.jsonl"

    log_stage("1/3", "Starting chunking pipeline")
    log_substep("1/3", f"Source directory: {source_dir}")
    log_substep("1/3", f"Manifest path: {manifest_path}")
    log_substep(
        "1/3",
        f"Chunk settings: size={args.chunk_size}, overlap={args.chunk_overlap}",
    )
    log_substep("1/3", "Models used in this step: none")
    log_substep("1/3", f"Output mode: {args.store}")

    stats = process_sources(
        source_dir=source_dir,
        output_file=output_file,
        manifest_path=manifest_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        store_mode=args.store,
        chroma_path=Path(args.chroma_path),
        chroma_collection_name=args.chroma_collection,
        chroma_batch_size=args.chroma_batch_size,
    )

    log_substep("1/3", f"Processed {stats['files']} source documents")
    log_substep("1/3", f"Created {stats['chunks']} chunks")
    if args.store == "jsonl":
        log_stage("1/3", f"Wrote chunk records to {output_file}")
    elif args.store == "chroma":
        log_stage(
            "1/3",
            f"Wrote chunk documents to Chroma collection '{args.chroma_collection}'",
        )
    else:
        log_stage(
            "1/3",
            f"Wrote chunks to {output_file} and collection '{args.chroma_collection}'",
        )


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
