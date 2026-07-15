"""SambaNova chat helpers for grounded RAG responses."""

from sambanova import SambaNova

from rag.logging_utils import verbose_step


SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions using the retrieved context. "
    "Prefer the provided context when it is relevant. If the context is incomplete, "
    "say what is missing instead of making up details. Cite the supporting sources in "
    "your answer using the provided chunk labels and source paths, for example "
    "[Chunk 1, source_docs/paper.pdf]. If multiple chunks support a claim, cite each "
    "relevant chunk."
)


@verbose_step("Creates the SambaNova client that will answer grounded user questions.")
def create_sambanova_client(api_key: str, base_url: str):
    if not api_key:
        raise RuntimeError(
            "SAMBANOVA_API_KEY is not set. Export it in your shell before running chat."
        )
    return SambaNova(
        api_key=api_key,
        base_url=base_url,
    )


@verbose_step(
    "Builds the final grounded user prompt from the query and retrieved context."
)
def build_grounded_user_message(
    query_text: str,
    source_guide: str,
    context_block: str,
) -> str:
    return (
        f"User question:\n{query_text}\n\n"
        "Answer the question using the retrieved context below. "
        "Cite supporting evidence inline with the chunk label and source.\n\n"
        f"Source guide:\n{source_guide}\n\n"
        f"Retrieved context:\n{context_block}"
    )


@verbose_step("Sends the grounded prompt to SambaNova and returns the model's answer.")
def request_grounded_answer(
    client,
    model_name: str,
    messages: list[dict[str, str]],
) -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.1,
        top_p=0.1,
    )
    return response.choices[0].message.content or ""
