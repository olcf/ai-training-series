"""Retrieval helpers for query-time grounding."""


from rag.logging_utils import verbose_step


@verbose_step(
    "Formats retrieved chunks into the grounded context block sent to the model."
)
def build_context_block(docs, metas, dists) -> str:
    context_parts = []
    for idx, (doc, meta, dist) in enumerate(zip(docs, metas, dists), 1):
        meta = meta or {}
        source = meta.get("relative_source") or meta.get("source") or "unknown"
        chunk_index = meta.get("chunk_index", "?")
        distance = f"{dist:.6f}" if isinstance(dist, (float, int)) else str(dist)
        context_parts.append(
            f"[Chunk {idx}]\n"
            f"source: {source}\n"
            f"chunk_index: {chunk_index}\n"
            f"distance: {distance}\n"
            f"content:\n{doc or ''}"
        )
    return "\n\n".join(context_parts)


@verbose_step(
    "Builds a compact source guide so citations map back to retrieved chunks."
)
def build_source_guide(metas) -> str:
    source_lines = []
    for idx, meta in enumerate(metas, 1):
        meta = meta or {}
        source = meta.get("relative_source") or meta.get("source") or "unknown"
        chunk_index = meta.get("chunk_index", "?")
        source_lines.append(
            f"Chunk {idx}: source={source}, chunk_index={chunk_index}"
        )
    return "\n".join(source_lines)


@verbose_step("Prints short previews of retrieved chunks for tutorial inspection.")
def print_retrieved_chunks(docs, metas, dists, show_chars: int) -> None:
    print("\nRetrieved context:")
    for idx, (doc, meta, dist) in enumerate(zip(docs, metas, dists), 1):
        meta = meta or {}
        source = meta.get("relative_source") or meta.get("source") or "unknown"
        chunk_index = meta.get("chunk_index", "?")
        distance = f"{dist:.6f}" if isinstance(dist, (float, int)) else str(dist)
        preview = (doc or "")[: max(50, show_chars)].replace("\n", " ").strip()
        print(f"\n[{idx}] distance={distance}")
        print(f"source={source} chunk_index={chunk_index}")
        print(preview)


@verbose_step(
    "Embeds the user query and fetches the closest matching chunks from Chroma."
)
def retrieve_context(query_text: str, embedder, collection, top_k: int):
    try:
        from chromadb.errors import InvalidArgumentError
    except Exception:  # pragma: no cover
        InvalidArgumentError = Exception

    query_vector = embedder.encode([query_text], convert_to_numpy=True).tolist()[0]
    try:
        result = collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
    except InvalidArgumentError as exc:
        raise RuntimeError(
            f"Embedding dimension mismatch for collection '{collection.name}'. "
            f"The current query model produced dimension {len(query_vector)}. "
            "Choose the matching --chroma-collection / "
            "--embedding-model pair or regenerate embeddings."
        ) from exc

    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    dists = result.get("distances", [[]])[0]
    return docs, metas, dists
