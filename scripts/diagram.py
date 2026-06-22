"""Render the RAG architecture diagram to assets/architecture.png."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fig, ax = plt.subplots(figsize=(12, 3.2))
ax.set_xlim(0, 12)
ax.set_ylim(0, 3.2)
ax.axis('off')

boxes = [
    (0.2, 'Telecom docs\n(plans, billing,\ntroubleshooting)', '#e3f2fd'),
    (2.6, 'Chunk +\nembed\n(BGE-small)', '#e8f5e9'),
    (5.0, 'Chroma\nvector store', '#fff3e0'),
    (7.4, 'Retrieve\ntop-k', '#e8f5e9'),
    (9.8, 'LLM (Llama-3.2)\ngrounded answer\n+ [n] citations', '#f3e5f5'),
]
w, h, y = 2.0, 1.6, 0.8
for x, label, color in boxes:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05', fc=color, ec='#555'))
    ax.text(x + w / 2, y + h / 2, label, ha='center', va='center', fontsize=9)
for i in range(len(boxes) - 1):
    x0 = boxes[i][0] + w
    x1 = boxes[i + 1][0]
    ax.add_patch(FancyArrowPatch((x0, y + h / 2), (x1, y + h / 2), arrowstyle='->',
                                 mutation_scale=16, color='#333'))
ax.text(6.0, 2.9, 'Telecom RAG assistant (FastAPI, fully local via Ollama)',
        ha='center', fontsize=12, weight='bold')
ax.text(6.0, 0.45, 'POST /query  ->  { answer, sources[] }', ha='center', fontsize=9, color='#444')

os.makedirs(os.path.join(root, 'assets'), exist_ok=True)
plt.tight_layout()
plt.savefig(os.path.join(root, 'assets', 'architecture.png'), dpi=130, bbox_inches='tight')
print('wrote assets/architecture.png')
