"""Tests for gradient checkpointing and gradient accumulation compatibility.

Validates that:
- EF state is NOT double-updated during checkpoint recomputation
- EF state is NOT redundantly updated during gradient accumulation
- Outputs match between checkpointed and non-checkpointed forward passes
"""

from __future__ import annotations

import pytest
import torch
import torch.utils.checkpoint

from tcfp.nn import TCFPLinear

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

DEVICE = "cuda"


# ---------------------------------------------------------------------------
# Gradient Checkpointing (Feature 1)
# ---------------------------------------------------------------------------


class TestGradientCheckpointing:
    """EF state must not be mutated during checkpoint recomputation."""

    def test_checkpoint_no_ef_mutation_per_tensor(self) -> None:
        """Per-tensor TC: EF stable during checkpointed forward+backward.

        After the prime forward, EF is set. A subsequent checkpointed
        forward (without optimizer step) should NOT mutate EF — neither
        the recording pass (grad accum guard) nor the recomputation pass
        (no_grad context).
        """
        layer = TCFPLinear(
            128, 64, use_tensor_cores=True, error_feedback=True,
        ).to(DEVICE)
        layer._param_name = "test"

        # Prime EF buffer with an initial forward+backward
        x0 = torch.randn(8, 128, device=DEVICE, requires_grad=True)
        out0 = layer(x0)
        out0.sum().backward()

        # Snapshot EF after prime
        ef_before = layer._error_state._buffers["test"].clone()

        # Checkpointed forward+backward (no optimizer step → weight unchanged)
        x1 = torch.randn(8, 128, device=DEVICE, requires_grad=True)
        out1 = torch.utils.checkpoint.checkpoint(
            layer, x1, use_reentrant=False,
        )
        out1.sum().backward()

        # EF should be unchanged: weight._version didn't change since prime,
        # so both the recording and recomputation passes skip EF writes.
        ef_after = layer._error_state._buffers["test"]
        assert torch.equal(ef_before, ef_after), (
            "EF buffer changed during checkpoint without optimizer step"
        )

    def test_checkpoint_no_ef_mutation_block_scaled(self) -> None:
        """Block-scaled TC: EF stable during checkpointed forward+backward."""
        layer = TCFPLinear(
            128, 64,
            use_tensor_cores=True,
            error_feedback=True,
            scale_block_size=64,
        ).to(DEVICE)
        layer._param_name = "test"

        # Prime EF
        x0 = torch.randn(4, 128, device=DEVICE, requires_grad=True)
        out0 = layer(x0)
        out0.sum().backward()

        ef_before = layer._error_state._buffers["test"].clone()

        # Checkpointed forward+backward (no optimizer step → weight unchanged)
        x1 = torch.randn(4, 128, device=DEVICE, requires_grad=True)
        out1 = torch.utils.checkpoint.checkpoint(
            layer, x1, use_reentrant=False,
        )
        out1.sum().backward()

        ef_after = layer._error_state._buffers["test"]
        assert torch.equal(ef_before, ef_after), (
            "EF buffer changed during checkpoint without optimizer step"
        )

    def test_checkpoint_output_matches_per_tensor(self) -> None:
        """Checkpointed forward must produce same output as non-checkpointed."""
        layer = TCFPLinear(
            128, 64, use_tensor_cores=True, error_feedback=True,
        ).to(DEVICE)
        layer._param_name = "test"

        # Prime EF so both calls use the same effective weight.
        x = torch.randn(4, 128, device=DEVICE)
        layer(x)  # populates EF buffer

        # Now EF is stable (weight hasn't changed, so update_ef=False).
        ref = layer(x)
        out = torch.utils.checkpoint.checkpoint(
            layer, x, use_reentrant=False,
        )
        assert torch.allclose(ref, out, atol=1e-5)

    def test_checkpoint_output_matches_block_scaled(self) -> None:
        """Block-scaled: checkpointed forward matches non-checkpointed."""
        layer = TCFPLinear(
            128, 64,
            use_tensor_cores=True,
            error_feedback=True,
            scale_block_size=64,
        ).to(DEVICE)
        layer._param_name = "test"

        # Prime EF so both calls use the same effective weight.
        x = torch.randn(4, 128, device=DEVICE)
        layer(x)  # populates EF buffer

        ref = layer(x)
        out = torch.utils.checkpoint.checkpoint(
            layer, x, use_reentrant=False,
        )
        assert torch.allclose(ref, out, atol=1e-5)

    def test_checkpoint_backward_gradients(self) -> None:
        """Checkpointed backward produces finite, reasonable gradients."""
        layer = TCFPLinear(
            128, 64, use_tensor_cores=True, error_feedback=True,
        ).to(DEVICE)
        layer._param_name = "test"

        x = torch.randn(8, 128, device=DEVICE, requires_grad=True)
        out = torch.utils.checkpoint.checkpoint(
            layer, x, use_reentrant=False,
        )
        out.sum().backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all()


# ---------------------------------------------------------------------------
# Gradient Accumulation (Feature 2)
# ---------------------------------------------------------------------------


class TestGradientAccumulation:
    """EF state should only update when weight has changed."""

    def test_grad_accum_ef_stable_per_tensor(self) -> None:
        """Two forwards without optimizer.step() → EF updated only once."""
        layer = TCFPLinear(
            128, 64, use_tensor_cores=True, error_feedback=True,
        ).to(DEVICE)
        layer._param_name = "test"

        # First forward — primes EF and updates version tracking
        x1 = torch.randn(4, 128, device=DEVICE, requires_grad=True)
        layer(x1).sum().backward()

        ef_after_first = layer._error_state._buffers["test"].clone()

        # Second forward — same weight (no optimizer.step()) → EF should NOT change
        x2 = torch.randn(4, 128, device=DEVICE, requires_grad=True)
        layer(x2).sum().backward()

        ef_after_second = layer._error_state._buffers["test"]
        assert torch.equal(ef_after_first, ef_after_second), (
            "EF buffer changed during gradient accumulation (no optimizer.step())"
        )

    def test_grad_accum_ef_stable_block_scaled(self) -> None:
        """Block-scaled: EF stable across micro-batches."""
        layer = TCFPLinear(
            128, 64,
            use_tensor_cores=True,
            error_feedback=True,
            scale_block_size=64,
        ).to(DEVICE)
        layer._param_name = "test"

        x1 = torch.randn(4, 128, device=DEVICE, requires_grad=True)
        layer(x1).sum().backward()
        ef_after_first = layer._error_state._buffers["test"].clone()

        x2 = torch.randn(4, 128, device=DEVICE, requires_grad=True)
        layer(x2).sum().backward()
        ef_after_second = layer._error_state._buffers["test"]

        assert torch.equal(ef_after_first, ef_after_second)

    def test_grad_accum_resumes_after_step(self) -> None:
        """After optimizer.step(), EF update resumes."""
        layer = TCFPLinear(
            128, 64, use_tensor_cores=True, error_feedback=True,
        ).to(DEVICE)
        layer._param_name = "test"
        # Large LR ensures weight changes enough for different quant error
        opt = torch.optim.SGD(layer.parameters(), lr=1.0)

        # Forward + backward to prime EF
        x1 = torch.randn(4, 128, device=DEVICE, requires_grad=True)
        layer(x1).sum().backward()

        # Optimizer step changes weight in-place → version increments
        opt.step()
        opt.zero_grad()

        # Snapshot version BEFORE the post-step forward
        ver_after_step = layer.weight._version

        # Next forward should update EF (weight changed)
        x2 = torch.randn(4, 128, device=DEVICE, requires_grad=True)
        layer(x2).sum().backward()

        # The version tracking should have consumed the new version
        assert layer._last_weight_version == ver_after_step, (
            "Version tracking did not update after optimizer.step()"
        )

        # A second forward without step() should NOT update version tracking
        ver_before = layer._last_weight_version
        x3 = torch.randn(4, 128, device=DEVICE, requires_grad=True)
        layer(x3).sum().backward()
        assert layer._last_weight_version == ver_before, (
            "Version tracking changed without optimizer.step()"
        )

    def test_grad_accum_output_consistent(self) -> None:
        """Multiple micro-batches with same input produce same output."""
        layer = TCFPLinear(
            128, 64, use_tensor_cores=True, error_feedback=True,
        ).to(DEVICE)
        layer._param_name = "test"

        x = torch.randn(4, 128, device=DEVICE)

        # First forward primes EF; subsequent calls have stable EF.
        layer(x)
        out1 = layer(x)
        out2 = layer(x)  # Same weight, same EF, same input → same output

        assert torch.allclose(out1, out2, atol=1e-6), (
            "Outputs differ across micro-batches with unchanged weight"
        )
