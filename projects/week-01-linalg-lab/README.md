# linalg-lab — Week 1 reference project

**Goals:** Feel *at home* with vectors, matrices, dot products, projections, and SVD, in NumPy. Build the SVD image compressor the book chapter closes with.

## Prereqs
- Python 3.13
- uv (`brew install uv`)
- 200 MB free disk

## Run
```bash
uv sync
uv pip install -e ".[dev]"
uv run pytest
uv run jupyter lab notebooks/
```

## What's inside
- `src/linalg_lab/vectors.py` — dot, cosine similarity, projection.
- `src/linalg_lab/svd_compress.py` — SVD rank-k reconstruction, image compression.
- `tests/` — pytest + hypothesis property tests.
- `notebooks/01-svd-images.ipynb` — visual walkthrough.
- `assignments/{warmup,build,challenge}.md` — see the book chapter.
