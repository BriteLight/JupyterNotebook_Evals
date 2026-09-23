#!/usr/bin/env python3


"""Small, local-only RAG pipeline for a directory of mathematical PDFs."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import chromadb
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_PDF_DIR = Path("/Users/brad/Documents/Notes/math_docs")
DEFAULT_DB_DIR = Path(__file__).resolve().parent / "chroma_math"
COLLECTION_NAME = "math_documents_v1"


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def page_chunks(pdf_path: Path, chunk_size: int = 1200, overlap: int = 180):
    """Yield chunks while preserving the PDF page needed for citations."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    digest = file_digest(pdf_path)
    with pymupdf.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf):
            text = page.get_text("text", sort=True).strip()
            if not text:
                continue
            for chunk_index, text_chunk in enumerate(splitter.split_text(text)):
                chunk_id = f"{digest[:16]}:{page_index + 1}:{chunk_index}"
                yield chunk_id, text_chunk, {
                    "source": str(pdf_path),
                    "title": pdf_path.name,
                    "page": page_index + 1,
                    "chunk": chunk_index,
                    "file_sha256": digest,
                }


def collection(db_dir: Path):
    client = chromadb.PersistentClient(path=db_dir)
    # Chroma's default local embedding function uses all-MiniLM-L6-v2 via ONNX.
    # Its small model is downloaded once on first use, then cached locally.
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def index_pdfs(pdf_dir: Path, db_dir: Path, rebuild: bool = False) -> None:
    client = chromadb.PersistentClient(path=db_dir)
    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    store = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    pdfs = sorted(pdf_dir.rglob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found beneath {pdf_dir}")

    for pdf_path in pdfs:
        digest = file_digest(pdf_path)
        old = store.get(where={"source": str(pdf_path)}, include=["metadatas"])
        old_digest = old["metadatas"][0].get("file_sha256") if old["metadatas"] else None
        if old_digest == digest:
            print(f"unchanged: {pdf_path.name}")
            continue
        if old["ids"]:
            store.delete(ids=old["ids"])

        rows = list(page_chunks(pdf_path))
        for start in range(0, len(rows), 100):
            batch = rows[start : start + 100]
            store.upsert(
                ids=[row[0] for row in batch],
                documents=[row[1] for row in batch],
                metadatas=[row[2] for row in batch],
            )
        print(f"indexed:   {pdf_path.name} ({len(rows)} chunks)")

    # Remove records for PDFs that no longer exist in the source directory.
    current_sources = {str(path) for path in pdfs}
    known = store.get(include=["metadatas"])
    stale_ids = [
        item_id
        for item_id, metadata in zip(known["ids"], known["metadatas"])
        if metadata["source"] not in current_sources
    ]
    if stale_ids:
        store.delete(ids=stale_ids)
        print(f"removed:   {len(stale_ids)} stale chunks")
    print(f"collection contains {store.count()} chunks")


def retrieve(question: str, db_dir: Path, top_k: int = 6) -> list[dict]:
    store = collection(db_dir)
    if store.count() == 0:
        raise SystemExit("The collection is empty. Run the index command first.")
    result = store.query(
        query_texts=[question],
        n_results=min(top_k, store.count()),
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"text": text, "metadata": metadata, "distance": distance}
        for text, metadata, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]


def make_prompt(question: str, hits: list[dict]) -> str:
    context = "\n\n".join(
        f"[S{i}: {hit['metadata']['title']}, page {hit['metadata']['page']}]\n{hit['text']}"
        for i, hit in enumerate(hits, 1)
    )
    return f"""You are a careful mathematics tutor. Answer only from the supplied sources.
If the sources are insufficient, say so. Preserve mathematical notation, distinguish
assumptions from conclusions, and cite claims inline as [S1], [S2], etc.

SOURCES
{context}

QUESTION
{question}
"""


def generate_locally(prompt: str, model: str) -> str:
    """Generate with a local Hugging Face model directory (or download once by ID)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    llm = AutoModelForCausalLM.from_pretrained(model, dtype="auto")
    try:
        llm.to("mps")
    except Exception:
        llm.to("cpu")
    messages = [{"role": "user", "content": prompt}]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(rendered, return_tensors="pt").to(llm.device)
    output = llm.generate(
        **inputs,
        max_new_tokens=700,
        do_sample=False,
        repetition_penalty=1.05,
    )
    new_tokens = output[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    subparsers = parser.add_subparsers(dest="command", required=True)
    index_parser = subparsers.add_parser("index")
    index_parser.add_argument("--rebuild", action="store_true")
    ask_parser = subparsers.add_parser("ask")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--top-k", type=int, default=6)
    ask_parser.add_argument(
        "--model",
        help="Optional local Hugging Face model directory or model ID",
    )
    args = parser.parse_args()

    if args.command == "index":
        index_pdfs(args.pdf_dir, args.db_dir, args.rebuild)
        return

    hits = retrieve(args.question, args.db_dir, args.top_k)
    if args.model:
        print(generate_locally(make_prompt(args.question, hits), args.model))
    else:
        for i, hit in enumerate(hits, 1):
            metadata = hit["metadata"]
            print(
                f"\n[S{i}] {metadata['title']}, page {metadata['page']} "
                f"(distance={hit['distance']:.3f})\n{hit['text']}"
            )


if __name__ == "__main__":
    main()
