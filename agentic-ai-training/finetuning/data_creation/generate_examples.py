#!/usr/bin/env python3
"""Generate fine-tuning examples with RAG-grounded SambaNova prompts."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DEFAULT_CHROMA_COLLECTION,
    DEFAULT_CHROMA_PATH,
    DEFAULT_EMBEDDING_MODEL,
    SAMBANOVA_API_KEY,
    SAMBANOVA_BASE_URL,
    SAMBANOVA_MODEL_NAME,
)
from rag.chat import create_sambanova_client, request_grounded_answer
from rag.embeddings import build_model_collection_name, load_embedder
from rag.logging_utils import log_stage, log_substep, set_verbose
from rag.retrieval import build_context_block, build_source_guide, retrieve_context
from rag.vector_store import get_existing_collection


DEFAULT_SEED_FILE = PROJECT_ROOT / "finetuning" / "data_creation" / "seed_topics.jsonl"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "finetuning" / "generated_train.jsonl"
SYSTEM_MESSAGE = (
    "You are a helpful assistant for autonomous systems and AI security "
    "training. Answer clearly, briefly, and in a structured way for learners."
)
GENERATION_SYSTEM_PROMPT = (
    "You are generating training data for a fine-tuning dataset. Use the "
    "retrieved context to create one high-quality chat example for the given "
    "topic and question style. The example should stay high-level, "
    "beginner-friendly, and domain-appropriate.\n\n"
    "Return only valid JSON with this schema:\n"
    "{\"user\": \"...\", \"assistant\": \"...\"}\n\n"
    "Do not include markdown fences. Do not mention that you used retrieved "
    "context. Keep the answer concise and educational."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate fine-tuning examples using retrieval-grounded prompts."
    )
    parser.add_argument(
        "--seed-file",
        default=str(DEFAULT_SEED_FILE),
        help="Input JSONL file containing seed topics",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_FILE),
        help="Output JSONL file for generated training examples",
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
        default=4,
        help="Number of retrieved chunks to include when generating each example",
    )
    parser.add_argument(
        "--examples-per-seed",
        type=int,
        default=5,
        help="Number of training examples to generate for each seed topic",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print function-level tracing with a one-line summary for each step",
    )
    return parser


def iter_seed_records(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                yield json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in seed file at {path}:{line_num}: {exc}"
                ) from exc


def extract_json_object(text: str) -> dict[str, str]:
    raw = (text or "").strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("Model response did not contain a JSON object.")

    payload = json.loads(raw[start : end + 1])
    user_text = payload.get("user")
    assistant_text = payload.get("assistant")
    if not isinstance(user_text, str) or not user_text.strip():
        raise ValueError("Generated example is missing a valid 'user' field.")
    if not isinstance(assistant_text, str) or not assistant_text.strip():
        raise ValueError("Generated example is missing a valid 'assistant' field.")
    return {"user": user_text.strip(), "assistant": assistant_text.strip()}


def build_generation_prompt(seed: dict, source_guide: str, context_block: str) -> str:
    topic = seed.get("topic", "autonomous systems security")
    question_style = seed.get("question_style", "beginner explanation")
    retrieval_query = seed.get("retrieval_query", topic)

    return (
        f"Topic: {topic}\n"
        f"Question style: {question_style}\n"
        f"Retrieval query: {retrieval_query}\n\n"
        "Create one user question and one assistant answer suitable for a "
        "fine-tuning dataset.\n\n"
        f"Source guide:\n{source_guide}\n\n"
        f"Retrieved context:\n{context_block}"
    )


def generate_example_for_seed(
    seed: dict,
    embedder,
    collection,
    client,
    top_k: int,
) -> dict[str, object]:
    retrieval_query = seed.get("retrieval_query") or seed.get("topic") or ""
    docs, metas, dists = retrieve_context(retrieval_query, embedder, collection, top_k)
    if not docs:
        raise ValueError(
            f"No retrieval results found for seed query: {retrieval_query}"
        )

    context_block = build_context_block(docs, metas, dists)
    source_guide = build_source_guide(metas)
    messages = [
        {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_generation_prompt(seed, source_guide, context_block),
        },
    ]
    model_reply = request_grounded_answer(
        client=client,
        model_name=SAMBANOVA_MODEL_NAME,
        messages=messages,
    )
    example = extract_json_object(model_reply)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": example["user"]},
            {"role": "assistant", "content": example["assistant"]},
        ],
        "metadata": {
            "topic": seed.get("topic"),
            "question_style": seed.get("question_style"),
            "retrieval_query": retrieval_query,
        },
    }


def build_seed_variant(seed: dict, variant_index: int) -> dict:
    seed_copy = dict(seed)
    base_style = seed_copy.get("question_style", "beginner explanation")
    seed_copy["question_style"] = f"{base_style} variant {variant_index + 1}"
    return seed_copy


def run(args: argparse.Namespace) -> None:
    set_verbose(getattr(args, "verbose", False))
    seed_file = Path(args.seed_file)
    output_file = Path(args.output)

    if not seed_file.exists():
        raise FileNotFoundError(f"Seed file not found: {seed_file}")

    collection_name = args.chroma_collection or build_model_collection_name(
        DEFAULT_CHROMA_COLLECTION, args.embedding_model
    )

    log_stage("ft", "Starting RAG-assisted data creation")
    log_substep("ft", f"Seed file: {seed_file}")
    log_substep(
        "ft",
        f"Models used: embedding={args.embedding_model}, chat={SAMBANOVA_MODEL_NAME}",
    )
    log_substep("ft", f"Target output: {output_file}")
    log_substep("ft", f"Examples per seed: {args.examples_per_seed}")

    embedder = load_embedder(args.embedding_model)
    collection = get_existing_collection(
        Path(args.chroma_path), collection_name, args.embedding_model
    )
    client = create_sambanova_client(
        api_key=SAMBANOVA_API_KEY,
        base_url=SAMBANOVA_BASE_URL,
    )

    seeds = list(iter_seed_records(seed_file))
    generated_count = 0
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        for index, seed in enumerate(seeds, 1):
            topic = seed.get("topic", f"seed-{index}")
            for variant_index in range(max(1, args.examples_per_seed)):
                log_substep(
                    "ft",
                    (
                        f"Generating example {variant_index + 1}/"
                        f"{args.examples_per_seed} for seed {index}/{len(seeds)} "
                        f"('{topic}')"
                    ),
                )
                example = generate_example_for_seed(
                    seed=build_seed_variant(seed, variant_index),
                    embedder=embedder,
                    collection=collection,
                    client=client,
                    top_k=max(1, args.top_k),
                )
                handle.write(json.dumps(example, ensure_ascii=False) + "\n")
                generated_count += 1

    log_stage("ft", f"Wrote {generated_count} generated examples to {output_file}")


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
