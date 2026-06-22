# Telecom RAG assistant — project spec

## What this is
A retrieval-augmented support assistant over a telecom knowledge base. It grounds a local LLM in
retrieved document chunks and answers with inline citations, served as a FastAPI service. Runs fully
locally (Ollama + a local embedding model), no API keys.

## Pipeline
1. Ingest: load markdown docs, chunk with overlap.
2. Embed: BGE-small sentence embeddings.
3. Store: Chroma persistent vector store.
4. Retrieve: top-k by cosine similarity.
5. Generate: Llama-3.2 answers using only the retrieved context, citing sources as [n], and refuses
   when the answer is not present.

## Evaluation
A held-out eval set measures retrieval hit-rate (does the right source get retrieved) and
LLM-as-judge faithfulness (is the answer supported by the context). Reported with a chart.

## Honest framing
- Answers are only as good as the KB and the retrieval; the model is instructed to refuse out-of-KB
  questions, but it can still err.
- The knowledge base is sample telecom content, not a real operator's data.

## Maps to the JD
RAG, vector databases, prompt engineering, a scalable FastAPI service, containerization, and an
evaluation/testing harness — the core of an enterprise GenAI assistant.

## Decision log
- 2026-06-22: Built local-first (Ollama + BGE + Chroma) so it runs with no keys/accounts. Eval =
  retrieval hit-rate + LLM-judge faithfulness.
