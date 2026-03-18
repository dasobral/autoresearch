# Autoresearch Extensions

This document proposes how to generalize the autoresearch framework beyond its current scope (GPT pretraining on a single GPU) while preserving the core philosophy: **a fixed time budget, a single scalar metric, a mutable experiment file, and a git-based keep/revert loop**.

---

## Core Philosophy (Invariants)

Before listing extensions, it is worth naming what must stay constant across all adaptations:

| Invariant | Why it matters |
|-----------|---------------|
| Fixed time budget per run | Makes results comparable; prevents the agent from gaming by running longer |
| Single scalar metric, lower-is-better | Reduces the decision to a trivial comparison; no multi-objective tradeoffs |
| One mutable file | Constrains the agent's search space; prevents drift into infra changes |
| Git commit → run → keep/revert | Creates a clean, reproducible history; failure is cheap and recoverable |
| Metric must be task-agnostic (not dataset-size-dependent) | Ensures fairness when architecture changes alter throughput or vocabulary |

---

## Extension 1: Other Pretraining Objectives

### 1.1 Masked Language Modeling (BERT-style)

Replace the causal LM loss with a masked token prediction objective.

- **Mutable file**: `train.py` (swap autoregressive forward pass for masked forward pass)
- **Metric**: `val_bpb` over masked positions only, normalized by bytes of masked tokens
- **Key changes**:
  - Data collator must produce `(input_ids_with_masks, labels)` pairs
  - The model forward pass returns per-position loss; evaluation masks padding and non-masked positions
  - Positional embeddings: switch from RoPE to learned or ALiBi (RoPE has no causal assumption, but attention mask pattern changes)
- **Why it fits**: Same time budget, same metric normalization, same keep/revert logic

### 1.2 Prefix LM / Encoder-Decoder

Split the context into prefix (fully attended) and continuation (causally attended).

- **Mutable file**: `train.py` (modify `WINDOW_PATTERN` semantics; add `PREFIX_RATIO` hyperparameter)
- **Metric**: BPB over continuation tokens only
- **Key changes**:
  - Attention mask: prefix tokens attend bidirectionally; continuation tokens attend causally
  - `WINDOW_PATTERN` extended to support `P` (prefix) token type
- **Why it fits**: Same infrastructure; only the attention mask and loss mask change

### 1.3 Diffusion Language Models

Replace the next-token prediction objective with a discrete diffusion process (e.g., MDLM, SEDD).

- **Mutable file**: `train.py` (replace loss function and forward pass; add noise schedule)
- **Metric**: Bits per byte estimated via ELBO or continuous-time bound
- **Key changes**:
  - Noise schedule hyperparameters (`NOISE_SCHEDULE`, `T_MAX`) added to the hyperparameter block
  - Forward pass computes denoising loss at sampled timesteps
  - Evaluation: compute log-likelihood lower bound under the diffusion objective
- **Extension point**: The `get_lr_multiplier` schedule abstraction cleanly separates LR from noise schedules

---

## Extension 2: Other Model Architectures

### 2.1 State Space Models (Mamba / RWKV)

Replace the transformer attention layers with recurrent or state-space layers.

- **Mutable file**: `train.py` (replace `CausalSelfAttention` with SSM layer)
- **Metric**: `val_bpb` (identical; the evaluation harness in `prepare.py` is architecture-agnostic)
- **Key changes**:
  - Remove `WINDOW_PATTERN`, `HEAD_DIM`, `ASPECT_RATIO`; add `SSM_D_STATE`, `SSM_EXPAND_FACTOR`
  - Optimizer grouping: Muon still applies to 2D weight matrices; AdamW for SSM recurrent parameters
  - Flash Attention dependency removed; compilation strategy may differ
- **Hyperparameter knobs**: `D_STATE`, `D_CONV`, `EXPAND_FACTOR`, `DEPTH`
- **Why it fits**: The keep/revert loop is architecture-agnostic; BPB is architecture-agnostic

### 2.2 Mixture of Experts (MoE)

Replace dense FFN layers with sparse MoE layers.

- **Mutable file**: `train.py` (replace `MLP` class with `MoELayer`; add routing logic)
- **Metric**: `val_bpb` (normalized per byte, not per FLOP — this is intentional; efficiency is captured implicitly by the fixed time budget)
- **Key changes**:
  - New hyperparameters: `NUM_EXPERTS`, `TOP_K`, `EXPERT_CAPACITY_FACTOR`
  - Auxiliary load-balancing loss added to training loss (weighted by `AUX_LOSS_WEIGHT`)
  - VRAM increases substantially; `DEVICE_BATCH_SIZE` ladder needs a new MoE column
- **Insight**: The fixed time budget implicitly rewards sparse models — if MoE runs more steps in 5 minutes than a dense model, and val_bpb is lower, it wins fairly

### 2.3 Linear Attention / Subquadratic Variants

Replace softmax attention with linear attention (e.g., GLA, RetNet, HGRN2).

- **Mutable file**: `train.py` (replace `CausalSelfAttention`)
- **Metric**: `val_bpb`
- **Hyperparameter knobs**: `DECAY_RATE`, `CHUNK_SIZE` (for chunk-wise parallel scan), `GATE_ACTIVATION`
- **Why it fits**: Same FLOP budget argument as MoE; faster inference at long context is captured if `MAX_SEQ_LEN` is increased

---

## Extension 3: Fine-Tuning Regimes

### 3.1 Supervised Fine-Tuning (SFT)

Start from a pretrained checkpoint and tune on instruction-following data.

- **Mutable file**: `train.py` (change data source, loss mask to completion tokens only)
- **Immutable file**: `prepare.py` adapted to a new `prepare_sft.py` (download instruction data, format with chat template)
- **Metric**: BPB over assistant response tokens only (instruction tokens excluded from loss and from denominator)
- **Key changes**:
  - Add `CHECKPOINT_PATH` hyperparameter (agent can try different base models)
  - Add `LORA_RANK` / `LORA_ALPHA` if LoRA is explored (new hyperparameter knobs)
  - Time budget still 5 minutes; metric is still lower-is-better
- **Agent loop change**: Revert logic must also restore the base weights if LoRA adapters are merged

### 3.2 LoRA / Parameter-Efficient Fine-Tuning

Run ablations over LoRA rank, target modules, and learning rates.

- **Mutable file**: `train.py` (inject LoRA into attention projections; add `LORA_RANK`, `LORA_ALPHA`, `LORA_DROPOUT`)
- **Metric**: BPB on held-out SFT eval set
- **Key changes**:
  - Only LoRA parameters are trained; base model frozen
  - Optimizer grouping: Muon for LoRA A/B matrices (2D); AdamW for bias/norm if unfrozen
  - `DEVICE_BATCH_SIZE` can be larger because most parameters are frozen

### 3.3 Reward Model Training

Train a scalar reward head on preference pairs (Bradley-Terry objective).

- **Mutable file**: `train.py` (add reward head, pairwise loss)
- **Metric**: Validation accuracy on held-out preference pairs (higher is better → negate for keep/revert logic, or use `1 - accuracy` as the scalar to minimize)
- **Key changes**:
  - Dataloader yields `(chosen_ids, rejected_ids)` pairs
  - Forward pass runs twice; loss = `-log σ(r_chosen - r_rejected)`
  - New hyperparameters: `REWARD_HEAD_DIM`, `MARGIN`

---

## Extension 4: Inference-Time Procedures

These extensions apply the same loop to optimize procedures that happen *after* training, not during it.

### 4.1 Speculative Decoding

Tune the draft model selection and acceptance threshold for a fixed target model.

- **Mutable file**: `speculate.py` (speculative decoding configuration: draft model, lookahead depth, temperature)
- **Metric**: Tokens per second on a fixed benchmark prompt set (higher is better → negate)
- **Time budget**: 5-minute benchmark run on a fixed prompt set
- **Key changes**:
  - No training; the loop purely explores inference configuration
  - Hyperparameters: `DRAFT_MODEL`, `LOOKAHEAD_K`, `ACCEPTANCE_THRESHOLD`, `TEMPERATURE`
  - Git loop unchanged: commit config → benchmark → keep/revert

### 4.2 Quantization Ablation

Explore quantization schemes (INT8, INT4, GPTQ, AWQ) for a fixed pretrained model.

- **Mutable file**: `quantize.py` (quantization config: method, bits, group size, calibration samples)
- **Metric**: BPB on a fixed eval set after quantization (measures accuracy degradation)
- **Secondary metric**: Inference throughput (tokens/sec) — can be logged but not used for keep/revert
- **Time budget**: 5 minutes of calibration + evaluation
- **Key changes**:
  - Hyperparameters: `METHOD` (gptq/awq/bnb), `BITS`, `GROUP_SIZE`, `ACT_ORDER`
  - The keep/revert loop rewards quantization schemes that preserve quality

### 4.3 KV Cache Optimization

Tune KV cache eviction policies and budget for long-context inference.

- **Mutable file**: `kvcache.py` (eviction strategy, budget ratio, sink tokens)
- **Metric**: BPB on long-context eval (e.g., passkey retrieval accuracy, or BPB on 32K-token documents)
- **Time budget**: Fixed benchmark duration
- **Hyperparameters**: `CACHE_BUDGET_RATIO`, `SINK_TOKENS`, `EVICTION_POLICY` (local/h2o/snapkv)

### 4.4 Decoding Strategy Search

Optimize sampling hyperparameters for a fixed model on a fixed task.

- **Mutable file**: `decode.py` (temperature, top-p, top-k, repetition penalty, beam width)
- **Metric**: Task-specific: ROUGE-L for summarization, pass@1 for code, exact match for QA (all normalized to [0,1], then negated)
- **Time budget**: 5-minute evaluation over a fixed prompt set
- **Key insight**: The loop treats decoding hyperparameters as the search space; same git discipline applies

---

## Extension 5: Multi-Task and Transfer Learning

### 5.1 Domain-Adaptive Pretraining

Continue pretraining a general model on a domain-specific corpus.

- **Mutable file**: `train.py` (same as base; add `DOMAIN_MIX_RATIO` between general and domain data)
- **Metric**: BPB on domain-specific held-out shard
- **New hyperparameters**: `DOMAIN_MIX_RATIO`, `REPLAY_RATIO` (proportion of general data to prevent forgetting)
- **The loop explores**: How much domain data to mix, at what learning rate, for how long

### 5.2 Cross-Lingual Transfer

Train a multilingual model and ablate language balance.

- **Mutable file**: `train.py` (same; add per-language sampling weights)
- **Metric**: Average BPB across a fixed set of per-language validation shards
- **New hyperparameters**: `LANG_WEIGHTS` dict, `TEMPERATURE_SAMPLING` (for language upsampling)

---

## Extension 6: Alternative Search Strategies for the Agent Loop

The current loop uses a greedy hill-climbing strategy (keep if better, else revert). Several alternatives could be encoded in `program.md`:

### 6.1 Simulated Annealing

Accept worse results with probability `exp(-Δbpb / T)` where `T` decays over time.

- **Change to**: `program.md` keep/revert logic
- **New parameter in `program.md`**: `INITIAL_TEMPERATURE`, `COOLING_SCHEDULE`
- **Why**: Escapes local minima in the hyperparameter landscape; especially useful for architecture search

### 6.2 Population-Based Training (Multi-Branch)

Maintain N concurrent experiment branches; periodically copy hyperparameters from the best branch.

- **Change to**: Multi-branch git strategy; `program.md` extended with branch selection logic
- **Implementation**: Agent runs N branches in parallel (requires N GPUs or sequential simulation); TSV tracks branch ID
- **Why**: Explores diverse regions of the search space simultaneously

### 6.3 Bayesian Optimization Loop

Replace random/heuristic proposals with a Gaussian Process surrogate model.

- **New file**: `suggest.py` (reads `results.tsv`, fits GP, proposes next hyperparameter setting)
- **Change to**: `program.md` step 2 ("propose a change") now calls `uv run suggest.py`
- **Why**: More sample-efficient than random search, especially for continuous hyperparameters

### 6.4 LLM-Guided Mutation

The agent reads the last N results from `results.tsv` and the current `train.py`, then proposes the next change based on patterns in the history.

- **This is the current behavior** (the agent is an LLM)
- **Enhancement**: Provide a structured "reflection" step where the agent summarizes what it has learned before proposing the next experiment
- **New section in `program.md`**: After every 10 experiments, summarize trends in `results.tsv` and update a `notes.md` file

---

## Extension 7: Different Compute Regimes

### 7.1 Multi-GPU Data Parallelism

Scale the same experiment loop to multiple GPUs.

- **Mutable file**: `train.py` (add `DDP` wrapper; replace `DEVICE_BATCH_SIZE` with per-GPU batch)
- **Metric**: `val_bpb` unchanged
- **Key changes**:
  - `TOTAL_BATCH_SIZE` must be divisible by `NUM_GPUS * DEVICE_BATCH_SIZE * MAX_SEQ_LEN`
  - Time budget stays 5 minutes; more GPUs means more steps, means lower BPB for the same architecture
  - The loop naturally rewards architectures that scale well

### 7.2 CPU-Only / Edge Inference

Use a smaller time budget (e.g., 60 seconds) and smaller models for CPU-bound research.

- **Mutable file**: `train.py` (remove Flash Attention dependency; add `torch.compile` with CPU backend)
- **Metric**: `val_bpb` (same)
- **Key changes**:
  - `TIME_BUDGET = 60` (or 120)
  - `DEPTH` range reduced to [2, 6]
  - `DEVICE_BATCH_SIZE` ladder replaced with CPU memory estimates
  - Flash Attention replaced with `F.scaled_dot_product_attention` (native PyTorch)

### 7.3 Cloud Spot Instance Loop

Run experiments on preemptible cloud instances with checkpoint-resume.

- **Mutable file**: `train.py` (add checkpoint save/load at fixed intervals)
- **Change to**: `program.md` (add preemption handling: if run exits early, check if `checkpoint.pt` exists and resume)
- **Metric**: BPB at exactly 300 training seconds regardless of preemptions
- **New files**: `checkpoint.py` (save/load logic; kept outside `train.py` to preserve the single-mutable-file constraint)

---

## Generalizing the File Structure

For any new research domain, the following template applies:

```
prepare_{domain}.py   # Immutable: data download, metric definition, evaluation harness
train_{domain}.py     # Mutable: model, optimizer, hyperparameters (agent edits this)
program_{domain}.md   # Agent instructions: loop protocol, constraints, metric name
results_{domain}.tsv  # Experiment log (gitignored)
analysis_{domain}.ipynb  # Visualization (optional)
```

The core loop in `program.md` needs only three substitutions:

```
# Original (LLM pretraining)
run_command: uv run train.py > run.log 2>&1
metric_name: val_bpb
metric_direction: lower_is_better

# SFT example
run_command: uv run train_sft.py > run.log 2>&1
metric_name: val_bpb_completions
metric_direction: lower_is_better

# Speculative decoding example
run_command: uv run benchmark_speculative.py > run.log 2>&1
metric_name: tokens_per_second
metric_direction: higher_is_better  # → negate in keep/revert logic
```

---

## Invariants Checklist for New Extensions

Before implementing a new extension, verify:

- [ ] The metric is a single scalar per run
- [ ] The metric is comparable across runs with different configurations (not dataset-size-dependent, not step-count-dependent)
- [ ] The time budget can be fixed (or a fixed work budget equivalent, e.g., fixed number of gradient steps)
- [ ] The mutable file is self-contained (all hyperparameters are literals in the file, not CLI args or config files)
- [ ] The evaluation harness is in the immutable file and cannot be accidentally modified
- [ ] The git commit → run → keep/revert loop produces a clean, reviewable history
- [ ] OOM and NaN crashes are handled gracefully (log as 'crash', move on)

---

## Summary Table

| Extension | Mutable File | Metric | Time Budget | Notes |
|-----------|-------------|--------|-------------|-------|
| MLM (BERT) | `train.py` | masked val_bpb | 5 min | Swap attention mask |
| Prefix LM | `train.py` | val_bpb (completion) | 5 min | Add PREFIX_RATIO |
| Diffusion LM | `train.py` | ELBO bpb | 5 min | Add noise schedule |
| Mamba/SSM | `train.py` | val_bpb | 5 min | Remove attn; add SSM |
| MoE | `train.py` | val_bpb | 5 min | Aux loss, expert count |
| Linear Attn | `train.py` | val_bpb | 5 min | GLA/RetNet variants |
| SFT | `train.py` | completion bpb | 5 min | Mask instruction tokens |
| LoRA ablation | `train.py` | completion bpb | 5 min | Freeze base weights |
| Reward model | `train.py` | preference accuracy | 5 min | Negate for minimize |
| Speculative decoding | `speculate.py` | tokens/sec | 5 min eval | No training; negate |
| Quantization | `quantize.py` | post-quant bpb | 5 min cal+eval | |
| KV cache | `kvcache.py` | long-ctx bpb | 5 min eval | |
| Decoding search | `decode.py` | task metric | 5 min eval | Task-specific normalization |
| Domain adapt | `train.py` | domain bpb | 5 min | Add DOMAIN_MIX_RATIO |
| Multi-GPU | `train.py` | val_bpb | 5 min | DDP wrapper |
| CPU/edge | `train.py` | val_bpb | 1-2 min | Smaller budget |
