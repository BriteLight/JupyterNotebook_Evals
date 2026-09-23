# Minimal local mathematical-PDF RAG

This is a small retrieval-augmented generation (RAG) system, not a newly trained
LLM. It keeps PDF text and the Chroma index on this Mac. It preserves page numbers
so answers can be checked against the original mathematics.

## 1. Activate the existing environment

```bash
cd "/Users/brad/Documents/ChatGPT/JupyterNotebook Evals"
source "/Users/brad/Documents/DevOps_DSOps/.venv/bin/activate"
python -m pip install -r requirements.txt
```
f
The inspected environment already has every required package. The install command
mainly makes the setup reproducible. `langchain-ai`, `langchain-community`,
`langchain-chroma`, Milvus, and a Chroma server are not required.

## 2. Build or update the index

```bash
python math_rag.py index
```

The first run lets Chroma download its small ONNX embedding model. Later runs use
the local cache and skip unchanged PDFs. Add a PDF under
`/Users/brad/Documents/Notes/math_docs/` and run the same command again. To discard
and recreate the index after changing the embedding model or chunking rules:

```bash
python math_rag.py index --rebuild
```

Do not change embedding models without rebuilding: vectors from different models
must never be mixed in one collection.

## 3. Test retrieval before adding an LLM

```bash
python math_rag.py ask "What is the role of Lorentz transformations in defining simultaneity?"
```

This prints the six closest passages with PDF names, pages, and distances. Retrieval
quality is the foundation of RAG; a generator cannot repair missing source passages.
Try 10-20 questions whose answers and pages you know. If relevant passages are
often missed, try `--top-k 10`, then adjust chunk size/overlap or upgrade the
embedding model before changing the LLM.

## 4. Add fully local answer generation (optional)

`transformers` and `torch` are already installed in the inspected environment.
Use an instruction-tuned model that fits the machine. The M4 Max with 64 GB unified
memory can run a 7B-class model, though an MLX-quantized runtime would be faster and
smaller than this deliberately minimal Transformers path.

For example, after choosing and downloading a compatible instruction model:

```bash
python math_rag.py ask \
  "Explain the relativity of simultaneity and state its assumptions." \
  --model "/absolute/path/to/the/local-model"
```

A Hugging Face model ID can replace the path, but that performs a one-time internet
download. An absolute local directory keeps subsequent operation offline. The code
uses deterministic generation and instructs the model to cite retrieved sources.

## Practical quality improvements, in order

1. Create a small evaluation file of questions, expected PDF/page, and key facts.
2. Inspect retrieval output independently of prose generation.
3. Use a stronger retrieval embedding model suited to scientific text; rebuild the
   collection whenever it changes.
4. Add an optional reranker only if top-10 retrieval contains the answer but top-6
   ordering is weak.
5. Extract equations/tables with a more specialized parser only for pages where
   PyMuPDF text extraction demonstrably loses essential notation.
6. Fine-tune only if the desired *behavior or style* remains poor after retrieval
   and prompting are sound. Fine-tuning is not the right mechanism for keeping a
   growing PDF knowledge base current.

## Important limitations

- PDF extraction represents formulas as text and can scramble multi-column layout,
  equation alignment, diagrams, or scanned pages. Always expose citations and check
  consequential mathematical claims in the PDF.
- The default Chroma embedding model is convenient and fast, not math-specialized.
- A local model can still invent steps. The prompt asks it to say when context is
  insufficient, but this is a guardrail rather than a proof of correctness.
- Re-indexing changed documents is implemented, but renaming a file causes its old
  records to be cleaned only at the end of a successful indexing run.
