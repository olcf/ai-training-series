#!/usr/bin/env python3
"""Chat with a SambaNova model using Chroma retrieval as context."""

import argparse
from pathlib import Path

from config import (
    DEFAULT_CHROMA_COLLECTION,
    DEFAULT_CHROMA_PATH,
    DEFAULT_EMBEDDING_MODEL,
    SAMBANOVA_API_KEY,
    SAMBANOVA_BASE_URL,
    SAMBANOVA_MODEL_NAME,
)
from rag.chat import (
    SYSTEM_PROMPT,
    build_grounded_user_message,
    create_sambanova_client,
    request_grounded_answer,
)
from rag.embeddings import build_model_collection_name, load_embedder
from rag.logging_utils import log_stage, log_substep, set_verbose
from rag.retrieval import (
    build_context_block,
    build_source_guide,
    print_retrieved_chunks,
    retrieve_context,
)
from rag.vector_store import get_existing_collection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chat with a SambaNova model using Chroma retrieval."
    )
    parser.add_argument("query", nargs="?", help="Natural-language query text")
    parser.add_argument(
        "--query",
        dest="query_flag",
        default=None,
        help="Natural-language query text",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Embedding model used for retrieval vectorization",
    )
    parser.add_argument(
        "--chroma-path",
        default=str(DEFAULT_CHROMA_PATH),
        help="Persistent directory for ChromaDB",
    )
    parser.add_argument(
        "--chroma-collection",
        default=None,
        help="Collection name. If omitted, resolves from base+model.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieved chunks to include",
    )
    parser.add_argument(
        "--show-chars",
        type=int,
        default=400,
        help="Number of characters to show per retrieved chunk preview",
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
    return parser


def run(args: argparse.Namespace) -> None:
    set_verbose(getattr(args, "verbose", False))
    top_k = max(1, args.top_k)
    first_query = args.query_flag or args.query

    collection_name = args.chroma_collection or build_model_collection_name(
        DEFAULT_CHROMA_COLLECTION, args.embedding_model
    )

    log_stage("3/3", "Starting retrieval and grounded chat pipeline")
    log_substep(
        "3/3",
        "Models used in this step: "
        f"embedding={args.embedding_model}, chat={SAMBANOVA_MODEL_NAME}",
    )
    log_substep("3/3", f"Requested collection: {collection_name}")

    log_substep("3/3", "Loading query embedder")
    embedder = load_embedder(args.embedding_model)
    log_substep("3/3", "Connecting to Chroma collection")
    collection = get_existing_collection(
        Path(args.chroma_path), collection_name, args.embedding_model
    )
    active_collection_name = getattr(collection, "name", collection_name)
    log_substep("3/3", f"Active collection: {active_collection_name}")
    client = create_sambanova_client(
        api_key=SAMBANOVA_API_KEY,
        base_url=SAMBANOVA_BASE_URL,
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("Chroma chat is ready. Type 'exit' or 'quit' to leave.")
    print(f"Collection: {active_collection_name}")
    print(f"Agent model: {SAMBANOVA_MODEL_NAME}")

    pending_query = first_query

    while True:
        query_text = pending_query
        pending_query = None

        if not query_text:
            query_text = input("\nYou: ").strip()
        if not query_text:
            continue
        if query_text.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        log_substep("3/3", f"Retrieving top {top_k} chunks for query")
        docs, metas, dists = retrieve_context(query_text, embedder, collection, top_k)
        if not docs:
            print("\nNo matching context found.")
            continue

        if args.show_sources:
            log_substep("3/3", "Displaying retrieved chunk previews")
            print_retrieved_chunks(docs, metas, dists, args.show_chars)

        log_substep("3/3", "Building grounded prompt from retrieved context")
        context_block = build_context_block(docs, metas, dists)
        source_guide = build_source_guide(metas)
        messages.append(
            {
                "role": "user",
                "content": build_grounded_user_message(
                    query_text, source_guide, context_block
                ),
            }
        )

        log_substep("3/3", "Requesting grounded answer from SambaNova")
        assistant_reply = request_grounded_answer(
            client=client,
            model_name=SAMBANOVA_MODEL_NAME,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": assistant_reply})
        print(f"\nAssistant: {assistant_reply}")


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
