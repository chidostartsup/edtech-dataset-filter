# 📚 FineWeb-Edu Subject Filter

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-orange)](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)

**Slice FineWeb-Edu's 1.3T tokens by academic subject using vector embeddings.**

FineWeb-Edu is filtered for *educational quality* but not by *subject*. This pipeline fixes that — no classifier training required, just embedding similarity.

## Why?

Training a math-focused model? You don't need 1.3 trillion tokens of everything. You need the math.

This pipeline lets you extract subject-specific subsets:
- `fineweb-edu-mathematics`
- `fineweb-edu-physics`
- `fineweb-edu-computer-science`
- ...and 9 more subjects

## How It Works

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Subject Anchor │────▶│  Embed with     │────▶│  Cosine         │
│  Descriptions   │     │  BGE/GTE/etc    │     │  Similarity     │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
┌─────────────────┐     ┌─────────────────┐              │
│  Labeled        │◀────│  Assign Subject │◀─────────────┘
│  Dataset        │     │  + Confidence   │
└─────────────────┘     └─────────────────┘
```

1. Define anchor texts for each subject (e.g., *"calculus, derivatives, integrals"*)
2. Embed anchors → compute subject centroids
3. Embed each document → compute similarity to all centroids
4. Assign subject with highest similarity (+ confidence score)

No fine-tuning. No labeled training data. Just geometry.

## Subjects

| Subject | Anchors | Example Content |
|---------|---------|-----------------|
| Mathematics | algebra, calculus, proofs, statistics | Equations, theorems, problem sets |
| Physics | mechanics, thermodynamics, quantum | Forces, energy, particle physics |
| Chemistry | reactions, organic, biochemistry | Elements, bonding, stoichiometry |
| Biology | genetics, cells, ecology | DNA, evolution, anatomy |
| Computer Science | algorithms, ML, systems | Code, data structures, networks |
| History | civilizations, wars, movements | Historical events, political history |
| Literature | novels, poetry, criticism | Literary analysis, creative writing |
| Economics | micro, macro, finance | Markets, GDP, trade |
| Law | constitutional, criminal, civil | Cases, legislation, jurisprudence |
| Medicine | diagnosis, treatment, pathology | Clinical, pharmacology, public health |
| Philosophy | ethics, metaphysics, logic | Arguments, reasoning, epistemology |
| Psychology | cognitive, behavioral, clinical | Mental health, neuroscience, therapy |

## Quick Start

```bash
pip install datasets sentence-transformers faiss-cpu pandas numpy tqdm
```

```python
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np

# Load model + data
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train[:1000]")

# Define subject anchors
anchors = {
    "mathematics": "Mathematics, algebra, calculus, equations, proofs, theorems",
    "physics": "Physics, mechanics, thermodynamics, quantum, forces, energy",
    "biology": "Biology, cells, genetics, DNA, evolution, ecology",
    # ... add more
}

# Embed anchors
subject_embeddings = {k: model.encode(v, normalize_embeddings=True) for k, v in anchors.items()}
subjects = list(subject_embeddings.keys())
subject_matrix = np.stack([subject_embeddings[s] for s in subjects])

# Classify documents
doc_embeddings = model.encode(dataset["text"], normalize_embeddings=True)
similarities = doc_embeddings @ subject_matrix.T
labels = [subjects[i] for i in np.argmax(similarities, axis=1)]

print(f"Classified {len(labels)} documents")
```

## Full Pipeline

See [`fineweb_edu_subject_filter.ipynb`](./fineweb_edu_subject_filter.ipynb) for:
- Complete anchor definitions (5 per subject)
- Batch processing with progress bars
- Confidence thresholding
- Multi-label classification
- Streaming mode for full-scale processing
- HuggingFace Hub upload

## Scaling to Full Dataset

The notebook includes a streaming mode that processes the full 1.3T token dataset without loading it into memory:

```python
stream = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)

for batch in stream.iter(batch_size=1000):
    embeddings = model.encode(batch["text"], normalize_embeddings=True)
    # ... classify and save
```

Estimated processing time: ~24-48 hours on a single GPU (embedding inference).

## Results

Coming soon — subject distribution analysis across FineWeb-Edu.

## Extending

**Add a subject:**
```python
SUBJECT_ANCHORS["astronomy"] = [
    "Astronomy, stars, galaxies, planets, and cosmology",
    "Telescopes, celestial objects, and space exploration",
    # ...
]
```

**Use different embeddings:**
```python
# Swap for any sentence-transformers model
model = SentenceTransformer("Snowflake/snowflake-arctic-embed-m")
model = SentenceTransformer("thenlper/gte-large")
```

**Adjust confidence threshold:**
```python
# Only keep high-confidence classifications
filtered = dataset.filter(lambda x: x["subject_score"] >= 0.5)
```

## Citation

If you use this pipeline or the resulting datasets:

```bibtex
@software{fineweb_edu_subject_filter,
  author = {Chido Dzinotyiwei},
  title = {FineWeb-Edu Subject Filter},
  year = {2025},
  url = {https://github.com/chidostartsup/edtech-dataset-filter}
}
```

Built on top of [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) by HuggingFace.

## License

MIT
