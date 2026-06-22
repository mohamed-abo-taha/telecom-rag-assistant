"""Telecom knowledge-base RAG.

Embeds the docs into a Chroma vector store, retrieves the most relevant chunks, and asks a local
LLM (via Ollama) to answer grounded in that context with inline [n] citations.
"""
import os
import glob
import chromadb
from sentence_transformers import SentenceTransformer
import ollama

EMB_MODEL = 'BAAI/bge-small-en-v1.5'
LLM_MODEL = os.environ.get('RAG_LLM', 'llama3.2:3b')


class RagPipeline:
    def __init__(self, docs_dir, persist='outputs/chroma', collection='telecom_kb'):
        self.docs_dir = docs_dir
        self.embedder = SentenceTransformer(EMB_MODEL)
        self.client = chromadb.PersistentClient(path=persist)
        self.col = self.client.get_or_create_collection(collection)

    @staticmethod
    def _chunk(text, size=180, overlap=40):
        words = text.split()
        out, i = [], 0
        while i < len(words):
            out.append(' '.join(words[i:i + size]))
            i += size - overlap
        return out

    def build_index(self):
        ids, docs, metas = [], [], []
        for path in sorted(glob.glob(os.path.join(self.docs_dir, '*.md'))):
            name = os.path.basename(path)
            text = open(path, encoding='utf-8').read()
            for j, ch in enumerate(self._chunk(text)):
                ids.append(f'{name}#{j}')
                docs.append(ch)
                metas.append({'source': name, 'chunk': j})
        embs = self.embedder.encode(docs, normalize_embeddings=True).tolist()
        try:
            self.client.delete_collection(self.col.name)
        except Exception:
            pass
        self.col = self.client.get_or_create_collection(self.col.name)
        self.col.add(ids=ids, documents=docs, embeddings=embs, metadatas=metas)
        return len(ids)

    def retrieve(self, query, k=4):
        q = self.embedder.encode([query], normalize_embeddings=True).tolist()
        res = self.col.query(query_embeddings=q, n_results=k)
        return [{'text': d, 'source': m['source'], 'score': round(1 - dist, 3)}
                for d, m, dist in zip(res['documents'][0], res['metadatas'][0], res['distances'][0])]

    def answer(self, query, k=4):
        ctx = self.retrieve(query, k)
        context = '\n\n'.join(f"[{i + 1}] ({c['source']}) {c['text']}" for i, c in enumerate(ctx))
        prompt = (
            "You are a telecom support assistant. Answer the question using ONLY the context below. "
            "Cite the sources you use as [n]. If the answer is not in the context, say you do not "
            "have that information.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:")
        resp = ollama.chat(model=LLM_MODEL, messages=[{'role': 'user', 'content': prompt}],
                           options={'temperature': 0.1})
        return {
            'answer': resp['message']['content'].strip(),
            'sources': [{'source': c['source'], 'score': c['score']} for c in ctx],
        }
