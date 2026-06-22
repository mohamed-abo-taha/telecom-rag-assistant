"""Evaluate the RAG pipeline: retrieval hit-rate + LLM-as-judge faithfulness."""
import os
import sys
import ollama

sys.path.insert(0, os.path.dirname(__file__))
from rag import RagPipeline, LLM_MODEL   # noqa: E402

EVALSET = [
    {'q': 'How much is the Unlimited Pro plan?', 'src': 'plans.md'},
    {'q': 'What happens if I pay my bill late?', 'src': 'billing.md'},
    {'q': 'My data is not working abroad, what should I do?', 'src': 'troubleshooting.md'},
    {'q': 'How do I enable autopay?', 'src': 'billing.md'},
    {'q': 'Why did my data slow down after heavy use?', 'src': 'troubleshooting.md'},
    {'q': 'How many international minutes does the Smart plan include?', 'src': 'plans.md'},
    {'q': 'How long do I have to dispute a charge?', 'src': 'billing.md'},
    {'q': 'Does the Starter plan include roaming data?', 'src': 'plans.md'},
]


def judge(question, answer, context):
    prompt = (f'Question: {question}\nContext: {context}\nAnswer: {answer}\n\n'
              'Is the answer fully supported by the context, with no fabrication? Reply only YES or NO.')
    r = ollama.chat(model=LLM_MODEL, messages=[{'role': 'user', 'content': prompt}],
                    options={'temperature': 0})
    return 'YES' in r['message']['content'].strip().upper()[:5]


def main():
    docs = os.path.join(os.path.dirname(__file__), '..', 'data', 'docs')
    pipe = RagPipeline(docs_dir=docs)
    if pipe.col.count() == 0:
        pipe.build_index()

    hit = faith = 0
    for ex in EVALSET:
        ctx = pipe.retrieve(ex['q'])
        if any(c['source'] == ex['src'] for c in ctx):
            hit += 1
        a = pipe.answer(ex['q'])
        if judge(ex['q'], a['answer'], ' '.join(c['text'] for c in ctx)):
            faith += 1

    n = len(EVALSET)
    metrics = {'retrieval_hit_rate': round(hit / n, 2), 'faithfulness': round(faith / n, 2), 'n': n}
    print(metrics)


if __name__ == '__main__':
    main()
