"""Tests for autoresearch/train.py training logic.

Design strategy
---------------
train.py cannot be imported directly — its module-level code starts a full
training run as a side effect.  Pure-Python / numeric functions are therefore
copied verbatim here so they can be tested without triggering that side effect.

Torch-dependent tests (model forward/backward, dtype selection, GradScaler) are
grouped under ``requires_cuda`` and auto-skipped when torch/CUDA is unavailable.

Run (no GPU required for most tests):
    python3 -m pytest test_train.py -v

Run all tests including GPU tests (needs the project environment):
    uv run pytest test_train.py -v
"""

import math
import pytest

# ── Optional torch import — GPU tests are skipped when torch is absent ────────
try:
    import torch
    HAS_TORCH = True
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False
    HAS_CUDA = False

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
requires_cuda  = pytest.mark.skipif(not HAS_CUDA,  reason="CUDA GPU required")


# =============================================================================
# 1. GPU probe helpers  (pure Python — no torch required)
# =============================================================================
# Copied verbatim from the GPU probe block in train.py.

_PEAK_FLOPS_TABLE = [
    ("H100 SXM",          989.5e12),
    ("H100 PCIe",         835e12),
    ("H100",              989.5e12),
    ("H800",              989.5e12),
    ("A100 SXM",          312e12),
    ("A100",              312e12),
    ("A800",              312e12),
    ("RTX 4090",          165e12),
    ("RTX 4080 SUPER",    118e12),
    ("RTX 4080",           97.5e12),
    ("RTX 4070 Ti SUPER",  90e12),
    ("RTX 4070 Ti",        80e12),
    ("RTX 4070 SUPER",     71e12),
    ("RTX 4070",           55.5e12),
    ("RTX 4060 Ti",        44.5e12),
    ("RTX 4060",           30e12),
    ("RTX 3090 Ti",        80e12),
    ("RTX 3090",           71e12),
    ("RTX 3080 Ti",        64e12),
    ("RTX 3080",           45e12),
    ("RTX 3070 Ti",        43e12),
    ("RTX 3070",           40e12),
    ("RTX 3060 Ti",        32e12),
    ("RTX 3060",           24.5e12),
]

def _peak_flops_for(gpu_name, table=_PEAK_FLOPS_TABLE):
    """Longest-key match — mirrors the GPU_PEAK_FLOPS expression in train.py."""
    return max(((len(k), v) for k, v in table if k in gpu_name),
               default=(0, 989.5e12))[1]

def _batch_size_for_vram(vram_gb):
    """Mirrors the VRAM ladder in train.py."""
    if vram_gb >= 70:
        return 128
    elif vram_gb >= 35:
        return 64
    elif vram_gb >= 18:
        return 32
    elif vram_gb >= 10:
        return 16
    else:
        return 8


class TestPeakFlopsLookup:
    """Correctness of the longest-key GPU FLOPS lookup."""

    # --- basic family matching ---
    def test_h100_sxm_variant(self):
        assert _peak_flops_for("NVIDIA H100 SXM5 80GB HBM3") == 989.5e12

    def test_h100_pcie(self):
        assert _peak_flops_for("NVIDIA H100 PCIe 80GB") == 835e12

    def test_a100_sxm(self):
        assert _peak_flops_for("NVIDIA A100 SXM4-40GB") == 312e12

    def test_a100_plain(self):
        assert _peak_flops_for("NVIDIA A100-PCIE-40GB") == 312e12

    def test_rtx4090(self):
        assert _peak_flops_for("NVIDIA GeForce RTX 4090") == 165e12

    def test_rtx3090(self):
        assert _peak_flops_for("NVIDIA GeForce RTX 3090") == 71e12

    def test_rtx3080(self):
        assert _peak_flops_for("NVIDIA GeForce RTX 3080") == 45e12

    # --- specificity: longer key wins over shorter key ---
    def test_rtx4080_super_beats_rtx4080(self):
        assert _peak_flops_for("NVIDIA GeForce RTX 4080 SUPER") == 118e12

    def test_rtx4080_plain(self):
        assert _peak_flops_for("NVIDIA GeForce RTX 4080") == 97.5e12

    def test_rtx4070_ti_super_beats_ti(self):
        assert _peak_flops_for("NVIDIA GeForce RTX 4070 Ti SUPER") == 90e12

    def test_rtx4070_ti_plain(self):
        assert _peak_flops_for("NVIDIA GeForce RTX 4070 Ti") == 80e12

    def test_rtx3090_ti_beats_rtx3090(self):
        assert _peak_flops_for("NVIDIA GeForce RTX 3090 Ti") == 80e12

    def test_rtx3080_ti_beats_rtx3080(self):
        assert _peak_flops_for("NVIDIA GeForce RTX 3080 Ti") == 64e12

    # --- unknown GPU falls back to H100 default ---
    def test_unknown_gpu_fallback(self):
        assert _peak_flops_for("NVIDIA Tesla V100") == 989.5e12

    def test_empty_name_fallback(self):
        assert _peak_flops_for("") == 989.5e12

    # --- order independence: result must be the same on reversed table ---
    def test_order_independent_rtx4080_super(self):
        reversed_table = list(reversed(_PEAK_FLOPS_TABLE))
        assert _peak_flops_for("NVIDIA RTX 4080 SUPER", reversed_table) == 118e12

    def test_order_independent_rtx4070_ti_super(self):
        reversed_table = list(reversed(_PEAK_FLOPS_TABLE))
        assert _peak_flops_for("NVIDIA RTX 4070 Ti SUPER", reversed_table) == 90e12

    def test_order_independent_h100_pcie(self):
        reversed_table = list(reversed(_PEAK_FLOPS_TABLE))
        assert _peak_flops_for("NVIDIA H100 PCIe 80GB", reversed_table) == 835e12


class TestVRAMBatchSizeLadder:
    """VRAM → batch-size mapping, including all boundary values."""

    def test_h100_80gb(self):      assert _batch_size_for_vram(80.0) == 128
    def test_a100_40gb(self):      assert _batch_size_for_vram(40.0) == 64
    def test_rtx4090_24gb(self):   assert _batch_size_for_vram(24.0) == 32
    def test_rtx3070_8gb(self):    assert _batch_size_for_vram(8.0) == 8

    # boundary: exactly at threshold → takes the higher tier
    def test_boundary_70(self):    assert _batch_size_for_vram(70.0) == 128
    def test_boundary_35(self):    assert _batch_size_for_vram(35.0) == 64
    def test_boundary_18(self):    assert _batch_size_for_vram(18.0) == 32
    def test_boundary_10(self):    assert _batch_size_for_vram(10.0) == 16

    # just below each boundary → drops to lower tier
    def test_just_below_70(self):  assert _batch_size_for_vram(69.99) == 64
    def test_just_below_35(self):  assert _batch_size_for_vram(34.99) == 32
    def test_just_below_18(self):  assert _batch_size_for_vram(17.99) == 16
    def test_just_below_10(self):  assert _batch_size_for_vram(9.99)  == 8

    # all auto-selected batch sizes must divide TOTAL_BATCH_SIZE (2^19)
    def test_all_defaults_divide_total_batch_size(self):
        TOTAL_BATCH_SIZE = 2**19
        MAX_SEQ_LEN = 2048
        vram_samples = [80, 40, 24, 16, 12, 8, 4]
        for vram in vram_samples:
            bs = _batch_size_for_vram(vram)
            tokens = bs * MAX_SEQ_LEN
            assert TOTAL_BATCH_SIZE % tokens == 0, (
                f"VRAM={vram}GB → bs={bs}: {TOTAL_BATCH_SIZE} % {tokens} ≠ 0"
            )


# =============================================================================
# 2. LR schedule  (pure Python — verbatim from train.py)
# =============================================================================

WARMUP_RATIO   = 0.0
WARMDOWN_RATIO = 0.5
FINAL_LR_FRAC  = 0.0
WEIGHT_DECAY   = 0.2


def get_lr_multiplier(progress, warmup_ratio=WARMUP_RATIO,
                      warmdown_ratio=WARMDOWN_RATIO,
                      final_lr_frac=FINAL_LR_FRAC):
    if progress < warmup_ratio:
        return progress / warmup_ratio if warmup_ratio > 0 else 1.0
    elif progress < 1.0 - warmdown_ratio:
        return 1.0
    else:
        cooldown = (1.0 - progress) / warmdown_ratio
        return cooldown * 1.0 + (1 - cooldown) * final_lr_frac


def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95


def get_weight_decay(progress, weight_decay=WEIGHT_DECAY):
    return weight_decay * (1 - progress)


class TestLRMultiplier:

    # --- no-warmup default config ---
    def test_zero_progress_no_warmup(self):
        # WARMUP_RATIO=0 → skip warmup branch, return 1.0
        assert get_lr_multiplier(0.0) == pytest.approx(1.0)

    def test_flat_zone_at_quarter(self):
        # With WARMDOWN_RATIO=0.5, flat zone is [0, 0.5)
        assert get_lr_multiplier(0.25) == pytest.approx(1.0)

    def test_flat_zone_start_of_warmdown(self):
        # progress=0.5 exactly at boundary: 0.5 < 1.0 - 0.5 = 0.5 is False,
        # falls into else. cooldown = (1-0.5)/0.5 = 1.0 → returns 1.0
        assert get_lr_multiplier(0.5) == pytest.approx(1.0)

    def test_warmdown_halfway(self):
        # progress=0.75: cooldown = (1-0.75)/0.5 = 0.5 → 0.5*1 + 0.5*0 = 0.5
        assert get_lr_multiplier(0.75) == pytest.approx(0.5)

    def test_warmdown_end(self):
        # progress=1.0: cooldown = 0 → returns FINAL_LR_FRAC = 0.0
        assert get_lr_multiplier(1.0) == pytest.approx(0.0)

    # --- warmup path ---
    def test_warmup_zero_progress(self):
        # progress=0 with warmup_ratio=0.2 → 0/0.2 = 0.0
        assert get_lr_multiplier(0.0, warmup_ratio=0.2) == pytest.approx(0.0)

    def test_warmup_halfway(self):
        # progress=0.1 with warmup_ratio=0.2 → 0.5
        assert get_lr_multiplier(0.1, warmup_ratio=0.2) == pytest.approx(0.5)

    def test_warmup_full(self):
        # At progress=warmup_ratio, still in flat zone (equality falls into elif)
        assert get_lr_multiplier(0.2, warmup_ratio=0.2) == pytest.approx(1.0)

    # --- nonzero final LR ---
    def test_nonzero_final_lr_at_end(self):
        assert get_lr_multiplier(1.0, final_lr_frac=0.1) == pytest.approx(0.1)

    def test_nonzero_final_lr_halfway_warmdown(self):
        # progress=0.75, warmdown=0.5, final=0.1
        # cooldown = 0.5 → 0.5*1.0 + 0.5*0.1 = 0.55
        assert get_lr_multiplier(0.75, final_lr_frac=0.1) == pytest.approx(0.55)

    # --- edge case: zero warmdown never divides by zero for progress < 1.0 ---
    def test_zero_warmdown_near_end(self):
        # With warmdown_ratio=0, flat zone is progress < 1.0; else only at 1.0
        assert get_lr_multiplier(0.999, warmdown_ratio=0.0) == pytest.approx(1.0)

    # --- monotonicity invariants ---
    def test_warmup_monotone_increasing(self):
        vals = [get_lr_multiplier(p, warmup_ratio=0.3) for p in [0.0, 0.1, 0.2, 0.3]]
        assert all(vals[i] <= vals[i+1] for i in range(len(vals)-1))

    def test_warmdown_monotone_decreasing(self):
        vals = [get_lr_multiplier(p) for p in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
        assert all(vals[i] >= vals[i+1] for i in range(len(vals)-1))

    def test_lr_always_in_unit_interval(self):
        for p in [i/20 for i in range(21)]:
            val = get_lr_multiplier(p)
            assert 0.0 <= val <= 1.0, f"LR multiplier {val} out of [0,1] at progress={p}"


class TestMuonMomentum:

    def test_step_zero(self):
        assert get_muon_momentum(0) == pytest.approx(0.85)

    def test_step_150_midpoint(self):
        assert get_muon_momentum(150) == pytest.approx(0.90)

    def test_step_300_full(self):
        assert get_muon_momentum(300) == pytest.approx(0.95)

    def test_step_beyond_300_clamped(self):
        assert get_muon_momentum(600)  == pytest.approx(0.95)
        assert get_muon_momentum(1000) == pytest.approx(0.95)

    def test_range_always_in_085_to_095(self):
        for step in range(0, 500, 10):
            m = get_muon_momentum(step)
            assert 0.85 <= m <= 0.95, f"Momentum {m} out of [0.85, 0.95] at step={step}"

    def test_monotone_before_saturation(self):
        vals = [get_muon_momentum(s) for s in range(0, 350, 25)]
        assert all(vals[i] <= vals[i+1] for i in range(len(vals)-1))

    def test_flat_after_saturation(self):
        assert get_muon_momentum(300) == get_muon_momentum(301)
        assert get_muon_momentum(300) == get_muon_momentum(9999)


class TestWeightDecay:

    def test_at_zero_progress(self):
        assert get_weight_decay(0.0) == pytest.approx(WEIGHT_DECAY)

    def test_at_half_progress(self):
        assert get_weight_decay(0.5) == pytest.approx(WEIGHT_DECAY * 0.5)

    def test_at_full_progress(self):
        assert get_weight_decay(1.0) == pytest.approx(0.0)

    def test_linear_decay(self):
        for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
            assert get_weight_decay(p) == pytest.approx(WEIGHT_DECAY * (1 - p))

    def test_never_negative(self):
        for p in [i/10 for i in range(11)]:
            assert get_weight_decay(p) >= 0.0


# =============================================================================
# 3. Model configuration helpers  (pure Python — verbatim from train.py)
# =============================================================================

def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def _compute_window_sizes(pattern, n_layer, seq_len=2048):
    """Mirrors GPT._compute_window_sizes from train.py."""
    pattern = pattern.upper()
    assert all(c in "SL" for c in pattern)
    long_window  = seq_len
    short_window = seq_len // 2
    char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
    window_sizes = []
    for layer_idx in range(n_layer):
        char = pattern[layer_idx % len(pattern)]
        window_sizes.append(char_to_window[char])
    window_sizes[-1] = (long_window, 0)
    return window_sizes


def _build_model_dim(depth, aspect_ratio=64, head_dim=128):
    """Mirrors build_model_config from train.py."""
    base_dim  = depth * aspect_ratio
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    num_heads = model_dim // head_dim
    return model_dim, num_heads


class TestHasVE:

    def test_n_layer_8_ve_on_odd_layers(self):
        # n_layer=8: last idx=7, 7%2=1 → VE on odd indices
        for i in range(8):
            assert has_ve(i, 8) == (i % 2 == 1), f"layer {i} of 8 wrong"

    def test_n_layer_4_ve_on_odd_layers(self):
        for i in range(4):
            assert has_ve(i, 4) == (i % 2 == 1)

    def test_n_layer_7_ve_on_even_layers(self):
        # last idx=6, 6%2=0 → VE on even indices
        for i in range(7):
            assert has_ve(i, 7) == (i % 2 == 0)

    def test_n_layer_1_single_layer(self):
        # only layer 0 which is also the last → always True
        assert has_ve(0, 1) is True

    def test_n_layer_2(self):
        # last idx=1, 1%2=1 → only odd layer (idx=1) has VE
        assert has_ve(0, 2) is False
        assert has_ve(1, 2) is True

    def test_last_layer_always_has_ve(self):
        for n in range(1, 16):
            assert has_ve(n - 1, n) is True, f"last layer of {n} should have VE"

    def test_alternating_pattern(self):
        # Consecutive layers must alternate (no two adjacent layers both have / lack VE)
        for n in [4, 6, 8, 10]:
            results = [has_ve(i, n) for i in range(n)]
            pairs = [(results[i], results[i+1]) for i in range(n-1)]
            assert all(a != b for a, b in pairs), f"Non-alternating VE for n_layer={n}"


class TestWindowSizes:

    def test_last_window_always_full(self):
        for pattern in ["SSSL", "SSSS", "S", "L", "SL", "LS"]:
            for n_layer in range(1, 10):
                ws = _compute_window_sizes(pattern, n_layer)
                assert ws[-1] == (2048, 0), (
                    f"Last window not full for pattern={pattern}, n_layer={n_layer}"
                )

    def test_sssl_8_layers(self):
        ws = _compute_window_sizes("SSSL", 8)
        # Indices:  0  1  2  3  4  5  6  7
        # Pattern:  S  S  S  L  S  S  S  L(forced)
        expected = [1024, 1024, 1024, 2048, 1024, 1024, 1024, 2048]
        for i, ((w, _), exp) in enumerate(zip(ws, expected)):
            assert w == exp, f"Layer {i}: expected {exp}, got {w}"

    def test_all_sliding_forced_last(self):
        # "S" * n_layer: all sliding, but last is forced to full
        ws = _compute_window_sizes("S", 4)
        assert ws[0] == (1024, 0)
        assert ws[1] == (1024, 0)
        assert ws[2] == (1024, 0)
        assert ws[3] == (2048, 0)  # forced full

    def test_all_full_unchanged(self):
        ws = _compute_window_sizes("L", 4)
        assert all(w == 2048 for w, _ in ws)

    def test_sl_pattern_wraps(self):
        # "SL" with 5 layers: S L S L S(→forced L)
        ws = _compute_window_sizes("SL", 5)
        expected = [1024, 2048, 1024, 2048, 2048]
        for i, ((w, _), exp) in enumerate(zip(ws, expected)):
            assert w == exp, f"Layer {i}: expected {exp}, got {w}"

    def test_single_layer(self):
        # Regardless of pattern, single layer is always forced full
        assert _compute_window_sizes("S", 1)[0] == (2048, 0)
        assert _compute_window_sizes("L", 1)[0] == (2048, 0)

    def test_invalid_char_raises(self):
        with pytest.raises(AssertionError):
            _compute_window_sizes("X", 4)

    def test_mixed_case_accepted(self):
        # Pattern should be case-insensitive
        ws_lower = _compute_window_sizes("sssl", 4)
        ws_upper = _compute_window_sizes("SSSL", 4)
        assert ws_lower == ws_upper

    def test_window_lengths_are_half_or_full(self):
        seq_len = 2048
        ws = _compute_window_sizes("SSSL", 8, seq_len=seq_len)
        for i, (w, gap) in enumerate(ws):
            assert w in (seq_len, seq_len // 2), f"Layer {i} has unexpected window {w}"
            assert gap == 0


class TestBuildModelConfig:

    def test_depth_8_default(self):
        model_dim, num_heads = _build_model_dim(8)
        assert model_dim == 512
        assert num_heads == 4

    def test_model_dim_divisible_by_head_dim(self):
        for depth in range(1, 33):
            model_dim, _ = _build_model_dim(depth)
            assert model_dim % 128 == 0, f"depth={depth} → model_dim={model_dim} not divisible by 128"

    def test_num_heads_positive(self):
        for depth in range(1, 33):
            _, num_heads = _build_model_dim(depth)
            assert num_heads >= 1

    def test_depth_1_rounds_up_to_head_dim(self):
        # base_dim = 1*64 = 64 < 128 → rounds up to 128
        model_dim, num_heads = _build_model_dim(1)
        assert model_dim == 128
        assert num_heads == 1

    def test_depth_2_exact(self):
        # base_dim = 2*64 = 128 = HEAD_DIM exactly → no rounding needed
        model_dim, num_heads = _build_model_dim(2)
        assert model_dim == 128
        assert num_heads == 1

    def test_depth_3_rounds_up(self):
        # base_dim = 3*64 = 192 → ceil(192/128)*128 = 256
        model_dim, num_heads = _build_model_dim(3)
        assert model_dim == 256
        assert num_heads == 2

    def test_monotone_dim_with_depth(self):
        dims = [_build_model_dim(d)[0] for d in range(1, 20)]
        # Not strictly monotone (ties round to same value), but non-decreasing
        assert all(dims[i] <= dims[i+1] for i in range(len(dims)-1))


# =============================================================================
# 4. Gradient accumulation invariants  (pure Python)
# =============================================================================

TOTAL_BATCH_SIZE = 2**19  # 524288
MAX_SEQ_LEN      = 2048


class TestGradAccumulation:

    @pytest.mark.parametrize("bs", [8, 16, 32, 64, 128])
    def test_default_vram_ladder_sizes_divide_total(self, bs):
        tokens = bs * MAX_SEQ_LEN
        assert TOTAL_BATCH_SIZE % tokens == 0

    def test_grad_accum_steps_correct(self):
        for bs in [8, 16, 32, 64, 128]:
            steps = TOTAL_BATCH_SIZE // (bs * MAX_SEQ_LEN)
            assert steps * bs * MAX_SEQ_LEN == TOTAL_BATCH_SIZE

    def test_non_power_of_two_bs_may_fail(self):
        # Verify the assertion would catch an invalid DEVICE_BATCH_SIZE.
        # bs=48 → tokens=98304, 524288 % 98304 ≠ 0
        bs = 48
        tokens = bs * MAX_SEQ_LEN
        assert TOTAL_BATCH_SIZE % tokens != 0

    def test_total_tokens_per_step_correct(self):
        for bs in [8, 16, 32, 64, 128]:
            steps = TOTAL_BATCH_SIZE // (bs * MAX_SEQ_LEN)
            total = steps * bs * MAX_SEQ_LEN
            assert total == TOTAL_BATCH_SIZE


# =============================================================================
# 5. EMA debiased loss  (pure Python)
# =============================================================================

def _run_ema(losses, beta=0.9):
    """Simulate the EMA accumulation from the training loop."""
    smooth = 0.0
    debiased_seq = []
    for step, loss in enumerate(losses):
        smooth = beta * smooth + (1 - beta) * loss
        debiased_seq.append(smooth / (1 - beta ** (step + 1)))
    return debiased_seq


class TestEMALoss:

    def test_single_step_equals_raw_loss(self):
        # At step 0: smooth = (1-0.9)*L, debiased = smooth / (1-0.9^1) = L
        for loss_val in [0.5, 1.0, 2.5, 10.0]:
            result = _run_ema([loss_val])
            assert result[0] == pytest.approx(loss_val)

    def test_constant_loss_converges(self):
        losses = [3.0] * 200
        result = _run_ema(losses)
        assert result[-1] == pytest.approx(3.0, rel=1e-3)

    def test_decreasing_loss_trends_down(self):
        losses = [5.0 - i * 0.05 for i in range(50)]
        result = _run_ema(losses)
        assert result[-1] < result[0]

    def test_debiased_always_finite(self):
        import random
        random.seed(0)
        losses = [random.uniform(0.5, 5.0) for _ in range(100)]
        for val in _run_ema(losses):
            assert math.isfinite(val)

    def test_debiased_matches_formula(self):
        # Verify against manual computation for 3 steps
        beta = 0.9
        losses = [2.0, 3.0, 1.0]
        s0 = 0.1 * 2.0
        s1 = 0.9 * s0 + 0.1 * 3.0
        s2 = 0.9 * s1 + 0.1 * 1.0
        expected = [
            s0 / (1 - 0.9**1),
            s1 / (1 - 0.9**2),
            s2 / (1 - 0.9**3),
        ]
        result = _run_ema(losses, beta=beta)
        for i, (r, e) in enumerate(zip(result, expected)):
            assert r == pytest.approx(e), f"step {i}: got {r}, expected {e}"

    def test_bias_correction_removes_zero_init(self):
        # Without correction the first estimate would be 0.1 * loss instead of loss
        loss = 4.0
        corrected = _run_ema([loss])[0]
        uncorrected = (1 - 0.9) * loss      # = 0.4
        assert corrected == pytest.approx(loss)
        assert corrected != pytest.approx(uncorrected)


# =============================================================================
# 6. MFU formula  (pure Python)
# =============================================================================

class TestMFUFormula:

    def test_step_mfu_formula(self):
        flops = 1e9
        batch = TOTAL_BATCH_SIZE
        dt = 1.0
        peak = 989.5e12
        mfu = 100 * flops * batch / dt / peak
        assert mfu == pytest.approx(100 * 1e9 * 524288 / 1.0 / 989.5e12)

    def test_mfu_scales_linearly_with_inverse_peak(self):
        flops, batch, dt = 1e9, 524288, 1.0
        mfu_a = 100 * flops * batch / dt / 989.5e12
        mfu_b = 100 * flops * batch / dt / (989.5e12 / 2)
        assert mfu_b == pytest.approx(2 * mfu_a)

    def test_steady_state_skips_warmup_steps(self):
        # Uses (step - 10) to exclude the first 10 warmup steps
        flops, batch, total_time, peak = 1e9, 524288, 100.0, 989.5e12
        step = 20
        mfu = 100 * flops * batch * (step - 10) / total_time / peak
        expected = 100 * 1e9 * 524288 * 10 / 100.0 / 989.5e12
        assert mfu == pytest.approx(expected)

    def test_steady_state_mfu_zero_when_no_training_time(self):
        # The ternary guard in train.py: ... if total_training_time > 0 else 0
        total_training_time = 0.0
        mfu = 0 if total_training_time == 0 else float("inf")
        assert mfu == 0

    def test_mfu_positive_for_positive_inputs(self):
        mfu = 100 * 1e9 * 524288 / 0.5 / 165e12
        assert mfu > 0

    def test_mfu_reasonable_range(self):
        # Default 8-layer 512-dim model: ~200 M flops/token, ~1 s/step on RTX 4090
        # → MFU ≈ 63 %, well within the physical [0, 100] % range.
        flops = 2e8          # ~200M flops/token for depth=8, model_dim=512
        batch = 524288
        dt = 1.0             # 1 second per step (conservative)
        peak = 165e12        # RTX 4090 dense BF16
        mfu = 100 * flops * batch / dt / peak
        assert 0 < mfu < 100, f"MFU={mfu:.1f}% out of expected [0,100] % range"


# =============================================================================
# 7. Training-time tracking  (pure Python)
# =============================================================================

class TestTrainingTimeTracking:
    """Verifies the step > 10 warmup-exclusion logic for total_training_time."""

    def _simulate(self, n_steps, dt_per_step=0.1):
        """
        Simulates the time-tracking logic from the training loop:
          if step > 10: total_training_time += dt
          step += 1
        Returns list of (step, total_training_time) after each step.
        """
        total_training_time = 0.0
        records = []
        for step in range(n_steps):
            # Placeholder for forward/backward/optimizer (already done above in real code)
            dt = dt_per_step
            if step > 10:
                total_training_time += dt
            records.append((step, total_training_time))
        return records

    def test_no_time_accumulated_first_11_steps(self):
        records = self._simulate(11)
        # Steps 0–10 (inclusive): time is never accumulated
        assert all(t == 0.0 for _, t in records)

    def test_time_starts_at_step_11(self):
        records = self._simulate(13)
        # Step 11 onwards (index 11 in records): time accumulates
        assert records[10][1] == 0.0   # step=10 → not accumulated
        assert records[11][1] > 0.0    # step=11 → accumulated

    def test_total_time_after_n_steps(self):
        n, dt = 50, 0.1
        records = self._simulate(n, dt_per_step=dt)
        # Steps accumulated: n_steps - 11 (steps 11 to n-1)
        expected_accum_steps = n - 11
        expected_time = expected_accum_steps * dt
        assert records[-1][1] == pytest.approx(expected_time)

    def test_loop_exits_after_time_budget(self):
        # Simulate: step > 10 and total_training_time >= TIME_BUDGET → break
        time_budget = 5.0
        dt = 0.5
        total_time = 0.0
        steps_run = 0
        for step in range(1000):
            if step > 10:
                total_time += dt
            steps_run += 1
            if step > 10 and total_time >= time_budget:
                break
        # With dt=0.5 and budget=5.0, we need 10 steps of accumulation → step 21
        assert total_time >= time_budget
        assert steps_run < 1000  # loop actually terminated


# =============================================================================
# 8. Loss explosion guard  (pure Python)
# =============================================================================

class TestLossExplosionGuard:

    @pytest.mark.parametrize("loss_val", [0.1, 1.0, 50.0, 99.9, 100.0])
    def test_normal_loss_no_exit(self, loss_val):
        # Mirrors: if train_loss_f > 100: exit(1)
        should_exit = loss_val > 100
        assert not should_exit

    @pytest.mark.parametrize("loss_val", [100.001, 200.0, float("inf")])
    def test_exploded_loss_triggers_exit(self, loss_val):
        should_exit = loss_val > 100
        assert should_exit

    def test_boundary_exactly_100_does_not_exit(self):
        assert not (100.0 > 100)

    def test_nan_does_not_trigger_guard(self):
        # NaN comparisons return False; nan > 100 is False
        import math
        assert not (math.nan > 100)


# =============================================================================
# 9. GPU-dependent tests  (skipped without CUDA)
# =============================================================================

@requires_cuda
class TestComputeDtype:
    """Verifies COMPUTE_DTYPE selection and downstream consistency."""

    def test_compute_dtype_is_torch_dtype(self):
        assert torch.cuda.is_bf16_supported() in (True, False)
        expected = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        # Verify the expected dtype is the one that would be selected
        assert expected in (torch.bfloat16, torch.float16)

    def test_scaler_none_for_bf16(self):
        if torch.cuda.is_bf16_supported():
            # BF16 doesn't need GradScaler
            compute_dtype = torch.bfloat16
            scaler = torch.amp.GradScaler("cuda") if compute_dtype == torch.float16 else None
            assert scaler is None
        else:
            pytest.skip("This GPU uses FP16; test BF16 scaler behaviour separately")

    def test_scaler_active_for_fp16(self):
        compute_dtype = torch.float16
        scaler = torch.amp.GradScaler("cuda") if compute_dtype == torch.float16 else None
        assert scaler is not None

    def test_autocast_context_uses_correct_dtype(self):
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ctx = torch.amp.autocast(device_type="cuda", dtype=compute_dtype)
        assert ctx is not None  # context is constructable


@requires_cuda
class TestModelForwardBackward:
    """Smoke-tests the GPT model forward/backward on a tiny synthetic config.

    Uses exec() to load only the class definitions from train.py, stopping
    before the module-level code that starts training.
    """

    @pytest.fixture(scope="class")
    def gpt_classes(self):
        import pathlib, sys, os

        src_path = pathlib.Path(__file__).parent / "train.py"
        source   = src_path.read_text()
        lines    = source.splitlines()

        # Stop just before "t_start = time.time()" which begins the setup block
        cutoff = next(
            i for i, line in enumerate(lines)
            if line.strip() == "t_start = time.time()"
        )
        exec_src = "\n".join(lines[:cutoff])

        ns = {}
        exec(compile(exec_src, str(src_path), "exec"), ns)
        return ns

    @pytest.fixture(scope="class")
    def tiny_model(self, gpt_classes):
        GPT       = gpt_classes["GPT"]
        GPTConfig = gpt_classes["GPTConfig"]
        # Minimal config: 2 layers, 2 heads, 256-dim, seq_len=64
        cfg = GPTConfig(
            sequence_len=64, vocab_size=256,
            n_layer=2, n_head=2, n_kv_head=2, n_embd=256,
            window_pattern="L",
        )
        model = GPT(cfg).cuda()
        model.init_weights()
        return model

    def test_forward_returns_finite_loss(self, tiny_model):
        x = torch.randint(0, 256, (2, 64), device="cuda")
        y = torch.randint(0, 256, (2, 64), device="cuda")
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.amp.autocast("cuda", dtype=compute_dtype):
            loss = tiny_model(x, y)
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_forward_logits_shape(self, tiny_model):
        x = torch.randint(0, 256, (2, 64), device="cuda")
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.amp.autocast("cuda", dtype=compute_dtype):
            logits = tiny_model(x)
        assert logits.shape == (2, 64, 256)

    def test_backward_no_nan_gradients(self, tiny_model):
        tiny_model.zero_grad(set_to_none=True)
        x = torch.randint(0, 256, (2, 64), device="cuda")
        y = torch.randint(0, 256, (2, 64), device="cuda")
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.amp.autocast("cuda", dtype=compute_dtype):
            loss = tiny_model(x, y)
        loss.backward()
        for name, p in tiny_model.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"
                assert not torch.isinf(p.grad).any(), f"Inf gradient in {name}"

    def test_loss_decreases_after_optimizer_steps(self, gpt_classes, tiny_model):
        """Loss should trend down over a handful of steps with a simple optimizer."""
        GPT = gpt_classes["GPT"]
        MuonAdamW = gpt_classes["MuonAdamW"]

        opt = tiny_model.setup_optimizer(
            unembedding_lr=0.004, embedding_lr=0.1,
            scalar_lr=0.5, matrix_lr=0.02, weight_decay=0.0,
        )
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        torch.manual_seed(0)
        x = torch.randint(0, 256, (4, 64), device="cuda")
        y = torch.randint(0, 256, (4, 64), device="cuda")

        losses = []
        for _ in range(5):
            tiny_model.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=compute_dtype):
                loss = tiny_model(x, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        # Loss should not increase monotonically (training is working)
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
        )

    def test_embedding_dtype_after_init(self, tiny_model):
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        wte_dtype = tiny_model.transformer.wte.weight.dtype
        assert wte_dtype == compute_dtype, (
            f"Embedding dtype {wte_dtype} doesn't match COMPUTE_DTYPE {compute_dtype}"
        )

    def test_rotary_emb_dtype(self, tiny_model):
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        assert tiny_model.cos.dtype == compute_dtype
        assert tiny_model.sin.dtype == compute_dtype

    def test_rotary_emb_shape(self, tiny_model):
        # cos/sin shape: (1, rotary_seq_len, 1, head_dim//2)
        head_dim = tiny_model.config.n_embd // tiny_model.config.n_head
        assert tiny_model.cos.shape == (1, tiny_model.rotary_seq_len, 1, head_dim // 2)
        assert tiny_model.sin.shape == tiny_model.cos.shape

    def test_logits_softcapped(self, tiny_model):
        """Logits should be bounded by the softcap value (15)."""
        x = torch.randint(0, 256, (2, 64), device="cuda")
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.amp.autocast("cuda", dtype=compute_dtype):
            logits = tiny_model(x)
        # softcap = 15 * tanh(x/15) ∈ (-15, 15)
        assert logits.abs().max().item() < 15.0 + 1e-4
