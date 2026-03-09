# Autoresearch

## Purpose

Autoresearch is an autonomous AI research framework for running overnight LLM-driven hyperparameter tuning and architecture experiments on language model pretraining. An AI agent (or human) iteratively modifies `train.py`, runs fixed 5-minute training experiments, measures improvement via a vocabulary-size-independent metric (`val_bpb`), and keeps or reverts changes based on results.

The intended workflow: prompt the agent with `program.md`, walk away, and return to ~100 completed experiments in the morning.

## Architecture

The codebase has three core files:

| File | Mutable | Role |
|------|---------|------|
| `prepare.py` | No | Data download, tokenizer training, dataloader, evaluation metric |
| `train.py` | **Yes** | GPT model definition, MuonAdamW optimizer, training loop |
| `program.md` | Yes | Agent instructions and experiment protocol |

Supporting files: `analysis.ipynb` (results visualization), `results.tsv` (experiment log, gitignored).

### Key concepts

- **Metric**: Bits Per Byte (`val_bpb`) — lower is better; vocabulary-size-independent so architectural changes are fairly compared.
- **Time budget**: exactly 5 minutes of wall-clock training per run (`TIME_BUDGET = 300`).
- **Fixed evaluation**: always uses the pinned validation shard `shard_06542.parquet`.
- **Optimizer**: MuonAdamW — Muon (polar orthogonalization) for 2D matrix weights, AdamW for embeddings and scalars.
- **Model**: GPT with RoPE, GQA, Flash Attention 3, Value Embeddings (ResFormer), and per-layer scaling.

## Requirements

- NVIDIA GPU (H100 recommended; Flash Attention 3 requires Hopper architecture)
- Python 3.10
- [`uv`](https://github.com/astral-sh/uv) package manager
- CUDA 12.8

## Setup

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install Python dependencies
uv sync

# 3. Download training data and train tokenizer (one-time, ~2 minutes)
uv run prepare.py
```

Data and tokenizer are cached in `~/.cache/autoresearch/`.

## Running

```bash
# Run a single 5-minute training experiment
uv run train.py

# Capture output for metric extraction (as the agent does)
uv run train.py > run.log 2>&1

# Extract key metrics from log
grep "^val_bpb:\|^peak_vram_mb:" run.log
```

## Autonomous Research Mode

Start an experiment branch, then prompt an LLM agent with the contents of `program.md`:

```bash
git checkout -b autoresearch/<tag>   # e.g. autoresearch/mar9
```

The agent loop:
1. Reads `train.py`
2. Proposes a change (hyperparameters, architecture, optimizer tweak)
3. Edits `train.py` and commits
4. Runs `uv run train.py > run.log 2>&1`
5. Parses `val_bpb` from the log
6. Appends result to `results.tsv`
7. Keeps the commit if improved, otherwise `git reset --hard HEAD~1`
8. Repeats indefinitely

### Agent constraints (from `program.md`)
- Only edit `train.py` (never `prepare.py`)
- Do not install new packages
- Prefer simple changes with clear improvement rationale
- Record every experiment in `results.tsv`

## Editable Hyperparameters (in `train.py`)

The primary knobs the agent tunes (lines ~431–450):

```python
ASPECT_RATIO = 64          # model_dim = depth * aspect_ratio
HEAD_DIM = 128             # target attention head dimension
WINDOW_PATTERN = "SSSL"   # S = sliding (half-ctx), L = full attention
TOTAL_BATCH_SIZE = 2**19  # ~524K tokens per step
EMBEDDING_LR = 0.6
UNEMBEDDING_LR = 0.004
MATRIX_LR = 0.04
SCALAR_LR = 0.5
WEIGHT_DECAY = 0.2
WARMUP_RATIO = 0.0
WARMDOWN_RATIO = 0.5
DEPTH = 8                  # number of transformer layers
DEVICE_BATCH_SIZE = 128    # tokens per GPU per micro-batch
```

## Analyzing Results

Open `analysis.ipynb` in Jupyter to visualize experiment progress from `results.tsv`:

```bash
uv run jupyter notebook analysis.ipynb
```

The notebook plots `val_bpb` over experiments and computes improvement deltas.
