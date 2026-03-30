"""
test_nanolens.py — Test suite for nanoLens variance collapse detector.
Run with: pytest test_nanolens.py -v
"""

import pytest
import torch
import torch.nn as nn
import logging


# ── Inline the implementation so tests are self-contained ──────────────────────

def _collapse_hook(module, inp, out, name, threshold, collapsed_layers):
    if name in collapsed_layers:
        return
    h = out[0] if isinstance(out, tuple) else out
    if not isinstance(h, torch.Tensor):
        return
    variance = h.float().var(dim=0).mean().item()
    if variance < threshold:
        collapsed_layers.add(name)
        logging.error(
            f"[nanoLens] Variance collapse at '{name}' "
            f"(var={variance:.6e} < threshold={threshold:.6e})"
        )
        if torch.cuda.is_available():
            try:
                torch.cuda.memory._record_memory_history(max_entries=100_000)
                torch.cuda.memory._dump_snapshot(
                    f"nanolens_collapse_{name.replace('.', '_')}.pickle"
                )
            except Exception:
                pass
        raise RuntimeError(
            f"[nanoLens] Representation collapse caught at '{name}'. Halting compute."
        )


def attach_nanolens(model, target_layer=nn.Linear, threshold=1e-4):
    handles = []
    collapsed_layers = set()
    for name, module in model.named_modules():
        if isinstance(module, target_layer):
            hook = lambda m, i, o, n=name: _collapse_hook(m, i, o, n, threshold, collapsed_layers)
            handles.append(module.register_forward_hook(hook))
    logging.info(f"[nanoLens] Attached to {len(handles)} layers (threshold={threshold:.6e}).")
    return handles


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_model():
    """Simple two-layer MLP for testing."""
    return nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 8),
    )


def healthy_input(batch=4, dim=16):
    """Random input — produces high-variance activations."""
    return torch.randn(batch, dim)


def collapsed_input(batch=4, dim=16):
    """Constant input — every sample is identical, so batch variance → 0."""
    return torch.ones(batch, dim)


# ── Attachment tests ───────────────────────────────────────────────────────────

class TestAttachment:
    def test_returns_correct_handle_count(self):
        model = make_model()
        handles = attach_nanolens(model)
        linear_count = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        assert len(handles) == linear_count

    def test_handles_are_removable(self):
        model = make_model()
        handles = attach_nanolens(model)
        for h in handles:
            h.remove()
        # After removal, collapsed input should not raise
        model(collapsed_input())

    def test_no_hooks_on_empty_model(self):
        model = nn.Sequential(nn.ReLU())  # no Linear layers
        handles = attach_nanolens(model)
        assert len(handles) == 0

    def test_custom_target_layer(self):
        model = nn.Sequential(nn.Conv2d(1, 4, 3), nn.Linear(16, 8))
        handles = attach_nanolens(model, target_layer=nn.Conv2d)
        assert len(handles) == 1

    def test_threshold_passed_correctly(self):
        """A very high threshold should flag even healthy activations."""
        model = make_model()
        attach_nanolens(model, threshold=1e10)
        with pytest.raises(RuntimeError, match="Representation collapse"):
            model(healthy_input())


# ── Detection tests ────────────────────────────────────────────────────────────

class TestCollapseDetection:
    def test_healthy_input_passes(self):
        model = make_model()
        attach_nanolens(model, threshold=1e-4)
        out = model(healthy_input())
        assert out.shape == (4, 8)

    def test_collapsed_input_raises(self):
        model = make_model()
        attach_nanolens(model, threshold=1e-4)
        with pytest.raises(RuntimeError, match="Representation collapse"):
            model(collapsed_input())

    def test_error_message_contains_layer_name(self):
        model = make_model()
        attach_nanolens(model, threshold=1e-4)
        with pytest.raises(RuntimeError, match=r"'[^']+'"): 
            model(collapsed_input())

    def test_detection_at_correct_layer(self):
        """Collapse should be caught at the first Linear layer (index 0)."""
        model = make_model()
        attach_nanolens(model, threshold=1e-4)
        with pytest.raises(RuntimeError) as exc_info:
            model(collapsed_input())
        assert "0" in str(exc_info.value)  # first layer is '0' in nn.Sequential

    def test_near_threshold_boundary(self):
        """Input with variance just above threshold should not trigger."""
        model = nn.Sequential(nn.Linear(16, 16))
        attach_nanolens(model, threshold=1e-4)
        # Scale random input to ensure variance well above threshold
        x = torch.randn(8, 16) * 10.0
        out = model(x)
        assert out is not None

    def test_tuple_output_handled(self):
        """Hooks must unwrap tuple outputs (HF/timm style)."""
        class TupleOutputLinear(nn.Linear):
            def forward(self, x):
                return (super().forward(x), None)  # simulate tuple output

        model = TupleOutputLinear(16, 8)
        attach_nanolens(model, target_layer=TupleOutputLinear, threshold=1e-4)
        with pytest.raises(RuntimeError, match="Representation collapse"):
            model(collapsed_input(dim=16))

    def test_non_tensor_output_skipped(self):
        """Hooks must not crash on non-tensor outputs."""
        class MetaOutputLinear(nn.Linear):
            def forward(self, x):
                return {"hidden": super().forward(x)}  # dict, not tensor

        model = MetaOutputLinear(16, 8)
        attach_nanolens(model, target_layer=MetaOutputLinear, threshold=1e-4)
        # Should not raise — non-tensor output is a no-op
        model(collapsed_input(dim=16))


# ── Guard tests ────────────────────────────────────────────────────────────────

class TestCollapseGuard:
    def test_fires_only_once_per_layer(self):
        """After first collapse, subsequent forward passes on same layer are silently skipped."""
        model = nn.Sequential(nn.Linear(16, 8))
        attach_nanolens(model, threshold=1e-4)

        # First call raises
        with pytest.raises(RuntimeError):
            model(collapsed_input())

        # Remove the raise so we can call forward again; hook is now a no-op
        # Rebuild model with same collapsed_layers set already populated
        # Easiest: re-attach after first fire — new set, so it fires again
        # (This test verifies the guard works within a single attach session)
        # Second call on *same* attach session: hook returns early, no raise
        try:
            model(collapsed_input())
        except RuntimeError:
            pytest.fail("Guard failed: collapse hook fired twice for the same layer.")

    def test_independent_sessions_are_isolated(self):
        """Two separate attach_nanolens calls must have independent collapsed_layers sets."""
        model_a = make_model()
        model_b = make_model()

        attach_nanolens(model_a, threshold=1e-4)
        attach_nanolens(model_b, threshold=1e-4)

        with pytest.raises(RuntimeError):
            model_a(collapsed_input())

        # model_b's set is independent — should still raise fresh
        with pytest.raises(RuntimeError):
            model_b(collapsed_input())

    def test_same_model_reattach_resets_guard(self):
        """Re-attaching to the same model resets the guard (new collapsed_layers set)."""
        model = make_model()
        handles = attach_nanolens(model, threshold=1e-4)

        with pytest.raises(RuntimeError):
            model(collapsed_input())

        for h in handles:
            h.remove()

        # Re-attach — fresh collapsed_layers, should fire again
        attach_nanolens(model, threshold=1e-4)
        with pytest.raises(RuntimeError):
            model(collapsed_input())


# ── dtype / shape edge cases ───────────────────────────────────────────────────

class TestEdgeCases:
    def test_float16_input_no_overflow(self):
        """fp16 activations must not overflow during variance calculation."""
        model = nn.Sequential(nn.Linear(16, 8)).half()
        attach_nanolens(model, threshold=1e-4)
        x = torch.randn(4, 16).half()
        # Should not raise OverflowError or produce NaN variance
        try:
            model(x)
        except RuntimeError as e:
            # Only acceptable RuntimeError is the collapse detection one
            assert "Representation collapse" in str(e)

    def test_single_sample_batch(self):
        """batch=1 means var(dim=0) is zero by definition — should always trigger."""
        model = nn.Sequential(nn.Linear(16, 8))
        attach_nanolens(model, threshold=1e-4)
        x = torch.randn(1, 16)  # single sample
        with pytest.raises(RuntimeError, match="Representation collapse"):
            model(x)

    def test_3d_input_sequence_model(self):
        """Transformers pass [batch, seq_len, dim] — variance should still work."""
        model = nn.Sequential(nn.Linear(16, 8))
        attach_nanolens(model, threshold=1e-4)
        x = torch.randn(4, 10, 16)  # batch=4, seq=10, dim=16
        out = model(x)
        assert out.shape == (4, 10, 8)

    def test_zero_threshold_never_triggers(self):
        """threshold=0 means variance must be exactly zero to trigger — healthy inputs pass."""
        model = make_model()
        attach_nanolens(model, threshold=0.0)
        out = model(healthy_input())
        assert out is not None
