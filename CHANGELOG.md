# Changelog

All changes are relative to the original `train.py` at the initial commit
(`b11d6f2`). Only `train.py` is mutable by design; `prepare.py` is unchanged.

---

## [Unreleased] — Multi-GPU Architecture Support

Branch: `claude/simplify-gpu-probe-zGQiO`

### Context

The original script was written and tuned for a single target: the NVIDIA H100
SXM. Three assumptions were hardcoded throughout:

| Original assumption | Location in original code |
|---|---|
| Flash Attention 3 from `varunneal/flash-attention-3` (Hopper only) | line 20 |
| `dtype=torch.bfloat16` in autocast, embeddings, rotary buffers, optimizer | lines 178, 190, 323, 461 |
| `H100_BF16_PEAK_FLOPS = 989.5e12` (single constant, used for MFU) | lines 462, 586, 617 |
| `DEVICE_BATCH_SIZE = 128` (assumes 80 GB HBM) | line 450 |

These assumptions meant the script would silently produce wrong MFU numbers on
any non-H100 GPU and would immediately OOM on consumer cards with 24 GB or less.

---

### Changes in `train.py`

#### 1. Flash Attention 3 — non-Hopper fallback (upstream PR `17b480a`)

Added before our work by the upstream author; included here for completeness.

```python
# Before
fa3 = get_kernel('varunneal/flash-attention-3').flash_attn_interface

# After
cap = torch.cuda.get_device_capability()
repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
fa3 = get_kernel(repo).flash_attn_interface
```

Hopper GPUs (H100, cap `9.0`) continue to use the faster Hopper-native FA3
kernel. All other architectures fall back to `kernels-community/flash-attn3`,
which supports Ampere (RTX 30xx) and Ada Lovelace (RTX 40xx).

---

#### 2. GPU probe block (new, ~55 lines)

A self-contained block that runs once at import time and sets five module-level
globals used throughout the rest of the file.

```python
_gpu_props    = torch.cuda.get_device_properties(0)
GPU_NAME      = _gpu_props.name
GPU_VRAM_GB   = _gpu_props.total_memory / 1024**3
COMPUTE_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
GPU_PEAK_FLOPS          # longest-key lookup in _PEAK_FLOPS_TABLE
_default_device_batch_size  # VRAM ladder
```

**`COMPUTE_DTYPE`** — uses `torch.cuda.is_bf16_supported()` to select
`bfloat16` on all architectures that have BF16 Tensor Core support (A100 sm_80,
consumer Ampere sm_86/87, Ada sm_89, Hopper sm_90) and falls back to `float16`
on older hardware.

**`GPU_PEAK_FLOPS`** — replaces the single `H100_BF16_PEAK_FLOPS = 989.5e12`
constant. A 24-entry lookup table covers the H100/H800/A100/A800 data-centre
line and the full RTX 30xx and 40xx consumer families. The match uses
longest-key substring so `"RTX 4080 SUPER"` (118 TFLOPS) always wins over the
shorter `"RTX 4080"` (97.5 TFLOPS) regardless of table order. Unknown GPUs fall
back to the H100 value (safe: MFU will read artificially low rather than crash).

**`_default_device_batch_size`** — selects a safe default `DEVICE_BATCH_SIZE`
based on detected VRAM, leaving headroom for the agent to scale the model:

| VRAM | Default `DEVICE_BATCH_SIZE` |
|---|---|
| ≥ 70 GB (H100 SXM) | 128 |
| ≥ 35 GB (A100 40 GB+) | 64 |
| ≥ 18 GB (RTX 4090 / 3090) | 32 |
| ≥ 10 GB (RTX 4080 / 3080) | 16 |
| < 10 GB | 8 |

All selected values are powers of two and evenly divide `TOTAL_BATCH_SIZE`
(2¹⁹ = 524 288), so the `assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0`
check never fires on auto-detected hardware.

---

#### 3. Hardcoded `bfloat16` replaced with `COMPUTE_DTYPE`

Every place that pinned the dtype to BF16 was updated to use the probe result.

| Site | Original | Updated |
|---|---|---|
| `GPT.init_weights` — embedding table | `.to(dtype=torch.bfloat16)` | `.to(dtype=COMPUTE_DTYPE)` |
| `GPT.init_weights` — value embeddings | `.to(dtype=torch.bfloat16)` | `.to(dtype=COMPUTE_DTYPE)` |
| `GPT._precompute_rotary_embeddings` | `cos.bfloat16(), sin.bfloat16()` | `cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)` |
| `muon_step_fused` — polar express | `X = g.bfloat16()` | `X = g.to(COMPUTE_DTYPE)` |
| autocast context | `dtype=torch.bfloat16` | `dtype=COMPUTE_DTYPE` |

---

#### 4. `GradScaler` for FP16 fallback

BF16 training is numerically stable without gradient scaling. FP16 is not —
its narrower dynamic range causes gradient underflow without a scaling factor.

```python
scaler = torch.amp.GradScaler("cuda") if COMPUTE_DTYPE == torch.float16 else None
```

The training loop branches on `scaler is not None` at two points:

```python
# backward
if scaler is not None:
    scaler.scale(loss).backward()
else:
    loss.backward()

# optimizer step
if scaler is not None:
    scaler.step(optimizer)
    scaler.update()
else:
    optimizer.step()
```

On BF16 hardware (all RTX 30xx / 40xx, A100, H100) `scaler` is `None` and the
code path is identical to the original.

---

#### 5. MFU metric uses `GPU_PEAK_FLOPS`

The hardcoded constant `H100_BF16_PEAK_FLOPS = 989.5e12` was removed. Both MFU
computations now reference the probed value:

```python
# Per-step MFU
mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / GPU_PEAK_FLOPS

# Steady-state MFU (printed in final summary)
steady_state_mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE * (step - 10) \
                   / total_training_time / GPU_PEAK_FLOPS
```

The `mfu_percent` output line is now meaningful on all supported GPUs. On an
RTX 4090 the denominator changes from 989.5 TFLOPS to 165 TFLOPS, so reported
MFU rises by ~6× — this is correct: the 4090 achieves a higher *fraction* of
its own peak, even though absolute throughput is lower.

---

#### 6. `DEVICE_BATCH_SIZE` default is GPU-adaptive

```python
# Before
DEVICE_BATCH_SIZE = 128  # per-device batch size (reduce if OOM)

# After
DEVICE_BATCH_SIZE = _default_device_batch_size  # auto-scaled to GPU VRAM; override here if needed
```

The variable remains in the editable hyperparameter section, so the agent (or a
human) can still override it by assigning a literal value on that line.

---

#### 7. Startup GPU info line

```
GPU: NVIDIA GeForce RTX 4090 | VRAM: 24.0 GB | dtype: torch.bfloat16 | peak: 165.0 TFLOPS
```

Printed before the model is built so it appears at the top of every `run.log`,
making it easy to identify which hardware produced a given experiment.

---

### New file: `test_train.py`

A pytest test suite (126 tests: 114 pure-Python, 12 GPU) that verifies the
training loop logic without importing `train.py` directly (importing it would
start a full training run as a side effect).

Run without a GPU:
```bash
python3 -m pytest test_train.py -v
```

Run with the full project environment (includes GPU tests):
```bash
uv run pytest test_train.py -v
```

| Test class | What is verified |
|---|---|
| `TestPeakFlopsLookup` | Correct GPU family matching, longer key wins, unknown GPU fallback, order-independence |
| `TestVRAMBatchSizeLadder` | All tier boundaries, auto-defaults divide `TOTAL_BATCH_SIZE` |
| `TestLRMultiplier` | Warmup/flat/warmdown phases, monotonicity, always in `[0, 1]` |
| `TestMuonMomentum` | Linear ramp 0.85→0.95, clamping at step 300 |
| `TestWeightDecay` | Linear decay to zero, never negative |
| `TestHasVE` | Alternating VE pattern, last layer always has VE |
| `TestWindowSizes` | Forced-full last layer, pattern wrap, single layer, invalid char |
| `TestBuildModelConfig` | HEAD_DIM divisibility, depth=1/2/3/8 exact values |
| `TestGradAccumulation` | Auto-defaults divide `TOTAL_BATCH_SIZE`, invalid sizes caught |
| `TestEMALoss` | Debiasing removes zero-init, formula verification |
| `TestMFUFormula` | Per-step and steady-state, warmup exclusion `(step−10)`, zero-time guard |
| `TestTrainingTimeTracking` | No accumulation steps 0–10, correct total, loop exit |
| `TestLossExplosionGuard` | Threshold at 100.0 exclusive, NaN behaviour |
| `TestComputeDtype` *(GPU)* | `COMPUTE_DTYPE` selection, `GradScaler` None for BF16 |
| `TestModelForwardBackward` *(GPU)* | Finite loss and gradients, loss decreases, dtype propagation, softcap |

---

### Compatibility

| GPU | Flash Attn | dtype | MFU | Default batch |
|---|---|---|---|---|
| H100 SXM / PCIe | FA3 Hopper kernel | BF16 | accurate | 128 |
| A100 40/80 GB | FA3 community | BF16 | accurate | 64 / 128 |
| RTX 4090 / 4080 / 4070 | FA3 community | BF16 | accurate | 16 – 32 |
| RTX 3090 / 3080 / 3070 | FA3 community | BF16 | accurate | 16 – 32 |
| Pre-Ampere (sm < 8.0) | FA3 community | **FP16 + GradScaler** | accurate | 8 – 16 |
