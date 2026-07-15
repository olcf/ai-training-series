"""Document chunking helpers for the tutorial pipeline."""

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from rag.logging_utils import verbose_step
from rag.vector_store import get_chroma_collection, sanitize_for_chroma


@verbose_step("Normalizes whitespace so chunk boundaries are more consistent.")
def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@verbose_step(
    "Breaks long text into paragraph and sentence-sized units before chunking."
)
def split_into_units(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    units: list[str] = []

    for paragraph in paragraphs:
        paragraph = re.sub(r"[ \t]+", " ", paragraph).strip()
        if len(paragraph) <= 800:
            units.append(paragraph)
            continue

        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", paragraph)
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                units.append(sentence)

    return units


@verbose_step("Packs text units into overlapping chunks sized for retrieval.")
def split_into_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if not text:
        return []

    units = split_into_units(text)
    if not units:
        return []

    bounded_units: list[str] = []
    for unit in units:
        if len(unit) <= chunk_size:
            bounded_units.append(unit)
            continue
        for i in range(0, len(unit), chunk_size):
            part = unit[i : i + chunk_size].strip()
            if part:
                bounded_units.append(part)

    chunks: list[str] = []
    start = 0
    while start < len(bounded_units):
        end = start
        current_len = 0

        while end < len(bounded_units):
            unit = bounded_units[end]
            sep = 2 if current_len > 0 else 0
            candidate_len = current_len + sep + len(unit)
            if current_len > 0 and candidate_len > chunk_size:
                break
            current_len = candidate_len
            end += 1

        if end == start:
            end = start + 1

        chunk_text = "\n\n".join(bounded_units[start:end]).strip()
        if chunk_text:
            chunks.append(chunk_text)

        if end >= len(bounded_units):
            break

        if chunk_overlap <= 0:
            start = end
            continue

        prev_start = start
        overlap_chars = 0
        new_start = end
        while new_start > prev_start:
            unit = bounded_units[new_start - 1]
            sep = 2 if new_start < end else 0
            next_overlap = overlap_chars + sep + len(unit)
            if next_overlap > chunk_overlap and new_start < end:
                break
            overlap_chars = next_overlap
            new_start -= 1

        start = new_start if new_start > prev_start else end

    return chunks


@verbose_step("Reads plain text files from disk for chunk generation.")
def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


@verbose_step("Extracts text from each page of a PDF source document.")
def read_pdf_file(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "PDF input requires 'pypdf'. Install it with: pip install pypdf"
        ) from exc

    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n\n".join(parts)


@verbose_step("Chooses the right file reader based on the document extension.")
def read_source_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".markdown"}:
        return read_text_file(path)
    if suffix == ".pdf":
        return read_pdf_file(path)
    raise ValueError(f"Unsupported file type: {path}")


@verbose_step("Discovers supported source documents in the tutorial corpus.")
def iter_source_files(root: Path) -> list[Path]:
    exts = ("*.txt", "*.md", "*.markdown", "*.pdf")
    files: list[Path] = []
    for pattern in exts:
        files.extend(root.rglob(pattern))
    return sorted(p for p in files if p.is_file())


@verbose_step("Loads manifest metadata so chunks can inherit paper details.")
def load_manifest(manifest_path: Path) -> dict[str, dict]:
    if not manifest_path.exists():
        return {}

    index: dict[str, dict] = {}
    with manifest_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in manifest at line {line_num}: {exc}"
                ) from exc

            if not isinstance(rec, dict):
                continue

            keys = []
            for field in ("source", "relative_source", "filename"):
                val = rec.get(field)
                if isinstance(val, str) and val.strip():
                    keys.append(val.strip())

            for key in keys:
                index[key] = rec
                index[Path(key).name] = rec

    return index


@verbose_step("Matches a source file to its manifest metadata entry when available.")
def get_metadata_for_path(
    path: Path,
    source_dir: Path,
    manifest_index: dict[str, dict],
) -> dict:
    rel_source = str(path.relative_to(source_dir))
    candidates = (rel_source, str(path), path.name)
    for key in candidates:
        if key in manifest_index:
            return manifest_index[key]
    return {}


def build_chunk_id(relative_source: str, chunk_index: int, chunk_text: str) -> str:
    digest = hashlib.sha1(chunk_text.encode("utf-8")).hexdigest()[:12]
    return f"{relative_source}:{chunk_index}:{digest}"


@verbose_step("Runs the end-to-end chunking pass and writes chunk artifacts.")
def process_sources(
    source_dir: Path,
    output_file: Path,
    manifest_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    store_mode: str,
    chroma_path: Path,
    chroma_collection_name: str,
    chroma_batch_size: int,
) -> dict[str, int]:
    files = iter_source_files(source_dir)
    if not files:
        raise FileNotFoundError(f"No supported files found under: {source_dir}")

    manifest_index = load_manifest(manifest_path)
    write_jsonl = store_mode in ("jsonl", "both")
    write_chroma = store_mode in ("chroma", "both")

    chroma_collection = None
    chroma_ids: list[str] = []
    chroma_docs: list[str] = []
    chroma_metas: list[dict[str, str | int | float | bool]] = []
    if write_chroma:
        chroma_collection = get_chroma_collection(
            chroma_path=chroma_path,
            collection_name=chroma_collection_name,
        )

    file_count = 0
    chunk_count = 0
    out = output_file.open("w", encoding="utf-8") if write_jsonl else None
    try:
        for path in files:
            raw_text = read_source_text(path)
            text = normalize_text(raw_text)
            if not text:
                continue

            file_count += 1
            chunks = split_into_chunks(text, chunk_size, chunk_overlap)
            rel_source = str(path.relative_to(source_dir))
            metadata = get_metadata_for_path(path, source_dir, manifest_index)

            for idx, chunk_text in enumerate(chunks):
                chunk_id = build_chunk_id(rel_source, idx, chunk_text)
                record = {
                    "id": chunk_id,
                    "source": str(path),
                    "relative_source": rel_source,
                    "chunk_index": idx,
                    "chunk_text": chunk_text,
                }
                if metadata:
                    record["metadata"] = metadata
                if write_jsonl and out is not None:
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")

                if write_chroma and chroma_collection is not None:
                    flat_meta: dict[str, Any] = {
                        "id": chunk_id,
                        "source": str(path),
                        "relative_source": rel_source,
                        "chunk_index": idx,
                    }
                    if metadata:
                        for mk, mv in metadata.items():
                            flat_meta[f"meta_{mk}"] = mv

                    chroma_ids.append(chunk_id)
                    chroma_docs.append(chunk_text)
                    chroma_metas.append(sanitize_for_chroma(flat_meta))

                    if len(chroma_ids) >= max(1, chroma_batch_size):
                        chroma_collection.upsert(
                            ids=chroma_ids,
                            documents=chroma_docs,
                            metadatas=chroma_metas,
                        )
                        chroma_ids.clear()
                        chroma_docs.clear()
                        chroma_metas.clear()

                chunk_count += 1
    finally:
        if out is not None:
            out.close()

    if write_chroma and chroma_collection is not None and chroma_ids:
        chroma_collection.upsert(
            ids=chroma_ids,
            documents=chroma_docs,
            metadatas=chroma_metas,
        )

    return {"files": file_count, "chunks": chunk_count}
