"""
test_nanolens.py — Test suite for nanoLens V2 (DDP-safe variance collapse detector).
Run with: pytest tests/test_nanolens.py -v
"""
import pytest
import torch
import torch.nn as nn
from nanolens import attach_nanolens, check_nanolens, calibrate

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
    """Constant input — every sample is identical, so inter-sample variance -> 0."""
    return torch.ones(batch, dim)

# ── Calibration Tests ──────────────────────────────────────────────────────────
class TestCalibration:
    def test_calibrate_returns_valid_threshold(self):
        model = make_model()
        sample = healthy_input()
        
        threshold, report = calibrate(model, sample, percentile=10)
        
        assert isinstance(threshold, float)
        assert threshold > 0.0  
        assert len(report) == 2 
        for name, var in report.items():
            assert var > 0.0

    def test_calibrate_threshold_below_healthy_variance(self):
        """Calibrated threshold must be safely below observed healthy variance."""
        model = make_model()
        sample = healthy_input()
        
        threshold, report = calibrate(model, sample, percentile=10)
        min_observed = min(report.values())
        
        # threshold should be well below the minimum healthy variance
        assert threshold < min_observed * 0.5

    def test_calibrate_empty_model_raises(self):
        model = nn.Sequential(nn.ReLU()) # No linear layers
        sample = healthy_input()
        
        with pytest.raises(RuntimeError, match="No target layers found"):
            calibrate(model, sample)

# ── Attachment Tests ───────────────────────────────────────────────────────────
class TestAttachment:
    def test_returns_correct_signature(self):
        model = make_model()
        handles, should_stop = attach_nanolens(model)
        
        assert len(handles) == 2
        assert isinstance(should_stop, torch.Tensor)
        assert should_stop.dtype == torch.bool
        assert should_stop.item() is False

    def test_handles_are_removable(self):
        model = make_model()
        # Use a massive threshold so healthy input WILL trigger it
        handles, should_stop = attach_nanolens(model, threshold=1e10) 
        
        model(healthy_input())
        assert should_stop.item() is True   # Confirm hooks were live
        
        # Now remove and reset
        for h in handles: 
            h.remove()
        should_stop.fill_(False)
        
        model(healthy_input())
        assert should_stop.item() is False  # Hooks are dead, flag stays clear

# ── DDP-Safe Detection Tests ───────────────────────────────────────────────────
class TestCollapseDetection:
    def test_healthy_input_passes(self):
        model = make_model()
        handles, should_stop = attach_nanolens(model, threshold=1e-4)
        
        model(healthy_input())
        
        assert should_stop.item() is False
        check_nanolens(should_stop)

    def test_collapsed_input_flips_flag_and_raises(self):
        model = make_model()
        # collapsed_input() is all-ones — var(dim=0) = 0.0 exactly,
        # which is below any positive threshold.
        handles, should_stop = attach_nanolens(model, threshold=1e-4)
        
        model(collapsed_input())
        assert should_stop.item() is True
        
        with pytest.raises(RuntimeError, match="Representation collapse"):
            check_nanolens(should_stop)

    def test_tuple_output_handled(self):
        class TupleOutputLinear(nn.Linear):
            def forward(self, x):
                return (super().forward(x), None)
                
        model = TupleOutputLinear(16, 8)
        handles, should_stop = attach_nanolens(model, target_layer=TupleOutputLinear, threshold=1e-2)
        
        model(collapsed_input(dim=16))
        assert should_stop.item() is True

# ── Variance Dimension Tests ───────────────────────────────────────────────────
class TestVarianceDimensions:
    def test_dim_0_vs_dim_neg_1(self):
        model = nn.Sequential(nn.Linear(16, 8))
        
        base_token = torch.arange(16, dtype=torch.float32) 
        batch_input = base_token.unsqueeze(0).expand(4, -1) 
        
        # 1. Test dim=0 (Inter-sample spread) - MUST CATCH COLLAPSE
        handles_0, should_stop_0 = attach_nanolens(model, threshold=1e-4, variance_dim=0)
        model(batch_input)
        assert should_stop_0.item() is True 
        for h in handles_0: h.remove()
            
        # 2. Test dim=-1 (Intra-token spread) - WILL SILENTLY PASS
        handles_1, should_stop_1 = attach_nanolens(model, threshold=1e-4, variance_dim=-1)
        model(batch_input)
        assert should_stop_1.item() is False 

# ── Edge Cases (Carried from V1) ───────────────────────────────────────────────
class TestEdgeCases:
    def test_single_sample_batch(self):
        """batch=1: var(dim=0, correction=0) = 0.0 — should always trigger."""
        model = nn.Sequential(nn.Linear(16, 8))
        handles, should_stop = attach_nanolens(model, threshold=1e-4)
        
        model(torch.randn(1, 16))
        assert should_stop.item() is True

    def test_3d_input_sequence_model(self):
        """Transformers pass (batch, seq, dim) — hooks must handle this shape."""
        model = nn.Sequential(nn.Linear(16, 8))
        handles, should_stop = attach_nanolens(model, threshold=1e-4)
        
        out = model(torch.randn(4, 10, 16))
        assert out.shape == (4, 10, 8)
        assert should_stop.item() is False

    def test_float16_no_overflow(self):
        """fp16 activations: .float() cast must prevent overflow in variance calc."""
        model = nn.Sequential(nn.Linear(16, 8)).half()
        handles, should_stop = attach_nanolens(model, threshold=1e-4)
        
        # Should not throw OverflowError or generate NaNs that crash the run
        model(torch.randn(4, 16).half())
        
    def test_guard_fires_only_once(self):
        """After collapse detected, subsequent forward passes are no-ops."""
        model = nn.Sequential(nn.Linear(16, 8))
        handles, should_stop = attach_nanolens(model, threshold=1e-4)
        
        model(collapsed_input())
        assert should_stop.item() is True
        
        should_stop.fill_(False) # Manually reset flag
        model(collapsed_input()) # Hook should be silenced by collapsed_layers guard
        assert should_stop.item() is False