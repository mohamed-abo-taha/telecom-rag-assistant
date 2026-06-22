"""FastAPI service wrapping the RAG pipeline. POST /query -> grounded answer + sources."""
import os
import sys
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(__file__))
from rag import RagPipeline   # noqa: E402

DOCS = os.path.join(os.path.dirname(__file__), '..', 'data', 'docs')
app = FastAPI(title='Telecom RAG assistant')
pipe = RagPipeline(docs_dir=DOCS)


class Query(BaseModel):
    question: str
    k: int = 4


@app.on_event('startup')
def _startup():
    if pipe.col.count() == 0:
        pipe.build_index()


@app.get('/health')
def health():
    return {'status': 'ok', 'chunks': pipe.col.count()}


@app.post('/query')
def query(q: Query):
    return pipe.answer(q.question, q.k)
