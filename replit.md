# Spike-Legal-NLP Research Framework

A complete experimental research framework for studying spike encoding vs. transformer-based legal text classification. Produces all results, figures, and reports needed for the research paper: *"A Semantic- and Energy-Aware Study of Spike Encoding for Legal Text Classification Under Domain Shift"*.

## Run & Operate

```bash
cd research

# Show framework info and configuration
python main.py info

# Quick test run (reduced samples, fast)
python main.py run --quick

# Full experiment on CaseHOLD with LegalBERT
python main.py run --dataset case_hold --encoder legal_bert

# Dataset management
python main.py dataset list
python main.py dataset info case_hold
python main.py dataset download case_hold

# Spike encoding demo
python main.py encode demo
python main.py encode compare --dataset case_hold

# Regenerate report from saved results
python main.py report
python main.py report --format latex
```

## Python Package Setup

Packages are installed into `.pythonlibs` via uv. To reinstall:
```bash
uv pip install transformers huggingface_hub datasets sentence-transformers snntorch matplotlib seaborn rich click pyyaml tqdm jinja2 Pillow evaluate pandas openpyxl pyarrow plotly nltk python-dotenv
```

## Stack

- Python 3.13, PyTorch 2.x, snntorch (LIF neurons)
- HuggingFace: transformers, datasets, sentence-transformers, huggingface_hub
- Spike encodings: Poisson Rate, Latency, Temporal, Population, Binary Threshold
- Visualization: matplotlib, seaborn (publication-quality, 300 DPI PDF)
- CLI: Click + Rich

## Where things live

```
research/
├── main.py                    # CLI entry point (python main.py ...)
├── config.yaml                # All experiment settings
├── requirements.txt           # Python dependencies
├── README.md                  # Full usage documentation
└── src/
    ├── datasets/              # DatasetManager, preprocessing, statistics
    ├── encoding/              # 5 spike encoding methods
    ├── models/                # TransformerBaseline, SNNClassifier
    ├── evaluation/            # ClassificationMetrics, SemanticPreservation,
    │                          # EnergyAnalyzer, DomainShiftEvaluator
    ├── visualization/         # ResearchPlotter (publication figures)
    ├── reporting/             # ReportGenerator (HTML/Markdown/LaTeX)
    └── experiments/           # ExperimentPipeline (full orchestration)
storage/
├── datasets/cache/            # Pickled HuggingFace dataset cache
├── embeddings/                # Cached transformer embeddings (per split)
├── results/figures/           # Generated PDF figures
└── results/reports/           # HTML/LaTeX/Markdown reports
```

## Architecture decisions

- **Embedding cache**: transformer embeddings are cached to disk keyed by model+dataset+split — avoids repeated forward passes across experiment runs
- **Spike encoding decoupled from models**: encoders receive raw numpy embeddings and output spike trains; SNN training is fully independent of encoding method
- **Energy model**: follows Horowitz (2014) — 4.6 pJ/MAC (GPU), 0.9 pJ/SOP (neuromorphic), 100 pJ/DRAM. Tests H3 (memory penalty reduces savings)
- **SNN**: snntorch Leaky Integrate-and-Fire with rate-coded output (sum spikes over time); falls back to PyTorch MLP if snntorch unavailable
- **Dataset cache**: first download auto-cached as pickle; subsequent runs fully offline

## Supported datasets (LexGLUE)

`case_hold`, `ecthr_a`, `ecthr_b`, `eurlex`, `ledgar`, `scotus`, `unfair_tos`
Custom: CSV, JSON, JSONL, Excel, Parquet (auto-detects text/label columns)

## Gotchas

- First run downloads transformer models (LegalBERT ~400MB, BERT ~400MB) — subsequent runs use HuggingFace cache
- Population coding expands feature dimension by `n_neurons` (default 10×) — spike train shape differs from other encodings
- `--quick` flag caps training samples at 100/50/100 for fast iteration
- Always `cd research` before running `python main.py`

## Pointers

- See `research/README.md` for full CLI documentation
- See `research/config.yaml` to adjust sample limits, model selection, encoding parameters
- See the `pnpm-workspace` skill for Node.js workspace structure
