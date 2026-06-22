"""Build the index and answer a couple of questions (prints answer + sources)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from rag import RagPipeline   # noqa: E402

pipe = RagPipeline(os.path.join(ROOT, 'data', 'docs'), persist=os.path.join(ROOT, 'outputs', 'chroma'))
print('indexed chunks:', pipe.build_index())
for q in ['My data is not working abroad, what should I do?',
          'How much is the Unlimited Pro plan and how many international minutes does it include?']:
    r = pipe.answer(q)
    print('\nQ:', q)
    print('A:', r['answer'])
    print('sources:', [s['source'] for s in r['sources']])
