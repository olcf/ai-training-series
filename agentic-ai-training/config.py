"""Centralized configuration exports for the tutorial RAG pipeline."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

# Source data and artifact paths.
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "source_docs"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "chunks.jsonl"
DEFAULT_MANIFEST_FILE = DEFAULT_SOURCE_DIR / "manifest.jsonl"
DEFAULT_CHROMA_PATH = PROJECT_ROOT / "chroma_db"

# Chunking defaults.
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_STORE_MODE = "jsonl"  # options: jsonl, chroma, both

# ChromaDB defaults.
DEFAULT_CHROMA_COLLECTION = "document_chunks"
DEFAULT_CHROMA_BATCH_SIZE = 100

# Embedding defaults.
DEFAULT_EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v2-moe"

# SambaNova settings.
SAMBANOVA_BASE_URL = "https://api.sambanova.ai/v1"
SAMBANOVA_MODEL_NAME = "gpt-oss-120b"
SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY", "")
