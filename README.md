# Tacoma Load Research Homework

A reproducible repository for the Evaluation & Load Research Analyst homework. It includes the narrative write‑up,
notebooks and scripts for data prep and plotting, and a small helper module for feature engineering.

> **Code & Reproducibility**: If this is public on GitHub, add your repo URL here in this first paragraph.

## Repo structure

```
tacoma-load-research-homework/
├─ src/tacoma/                # Python package with reusable helpers
│  ├─ __init__.py
│  ├─ features.py             # load_and_prepare_data() with winter & weekend flags
│  └─ homework2.py            # your original script (copied in)
├─ notebooks/
│  └─ homework2.ipynb         # your original notebook (copied in)
├─ scripts/
│  └─ make_figures.py         # CLI to render figures into ./figures/
├─ data/                      # place source data here (ignored by git)
├─ figures/                   # output plots saved here
├─ reports/
│  ├─ Tacoma_Homework_Narrative.docx
│  └─ Tacoma_Homework_Improved.docx
├─ references/
│  ├─ HomeWorkAssignment-Evaluation&LoadResearchAnalyst.pdf
│  └─ Homework_solution.docx
├─ tests/
│  └─ test_features.py
├─ requirements.txt
├─ environment.yml
├─ .gitignore
├─ LICENSE
└─ README.md
```

## Quickstart

### Option A: venv + pip
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Option B: conda
```bash
conda env create -f environment.yml
conda activate tacoma-load-research-homework
```

## Data

Place the assignment dataset (e.g., `hp_extract.xlsx` or equivalent) into `./data/`. If your file uses a different name or columns,
you can supply them as CLI arguments to the figure script (see below).

## Reproducing Figures

```bash
python -m scripts.make_figures \
  --input ./data/hp_extract.xlsx \
  --datetime-col "date time" \
  --usage-col "kWh" \
  --temp-col "temp"
```
This will save the three figures described in the write‑up into `./figures/`.

## GitHub Publishing

```bash
git init
git add .
git commit -m "Initial commit — 2025-09-14"
git branch -M main
git remote add origin https://github.com/<your-username>/tacoma-load-research-homework.git
git push -u origin main
```

## License

MIT — see `LICENSE`.
