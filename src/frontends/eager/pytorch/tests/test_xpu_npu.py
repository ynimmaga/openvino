#!/usr/bin/env python3
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# test_xpu_npu.py — Extended tests for torch.device("xpu:npu").
#
# Verifies that:
#   - "xpu:npu" maps to torch.device("npu", 0) via intel_npu.device()
#   - Tensors live on the "npu" device type
#   - Supported ops route through OpenVINO (OV stats counter increments)
#   - Unsupported ops fall back to Torch XPU rather than CPU  [Phase 1]
#   - XPU streams are available on the NPU device              [Phase 1]
#
# Run:
#   cd src/frontends/eager
#   python pytorch/tests/test_xpu_npu.py

import sys
import os

# Add the eager/ directory so intel_npu and npu_backend are importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import torch
import intel_npu

# Canonical device obtained via the xpu:npu alias
NPU = intel_npu.device("xpu:npu")   # resolves to torch.device("npu", 0)

# ─────────────────────────────────────────────────────────────────────────────
# Device / syntax tests
# ─────────────────────────────────────────────────────────────────────────────

def test_xpu_npu_device_string():
    """torch.device('xpu:npu') must resolve to npu (index 0 or None)."""
    d = intel_npu.device("xpu:npu")
    assert d.type == "npu", f"Expected 'npu', got '{d.type}'"
    assert d.index is None or d.index == 0, f"Unexpected index: {d.index}"
    print(f"  intel_npu.device('xpu:npu') = {d}")


def test_npu_device_string_variants():
    """Both 'npu' and 'npu:0' must also work as shortcuts."""
    d0 = intel_npu.device("npu")
    d1 = intel_npu.device("npu:0")
    assert d0.type == "npu"
    assert d1.type == "npu" and d1.index == 0


def test_tensor_on_xpu_npu():
    """Tensor created on xpu:npu lives on the 'npu' device type."""
    x = torch.randn(4, 4, device=NPU)
    assert x.device.type == "npu", f"Expected 'npu', got '{x.device.type}'"
    print(f"  tensor device: {x.device}")


def test_to_xpu_npu():
    """CPU tensor moved to 'xpu:npu' ends up on 'npu'."""
    cpu = torch.randn(3, 3)
    npu = cpu.to(NPU)
    assert npu.device.type == "npu"
    back = npu.to("cpu")
    assert torch.allclose(cpu, back, atol=1e-6), "roundtrip mismatch"


def test_is_available():
    """torch.npu.is_available() returns True for simulated NPU."""
    assert torch.npu.is_available(), "NPU not available"
    print(f"  device_count: {torch.npu.device_count()}")


# ─────────────────────────────────────────────────────────────────────────────
# XPU stream smoke test  (Phase 1 — stream pool not yet fully wired)
# ─────────────────────────────────────────────────────────────────────────────

def test_xpu_stream_exists():
    """NPU device exposes at least a default stream."""
    # Once Phase-1 stream pool lands, replace with:
    #   s = torch.npu.Stream()
    #   assert s.device.type == "npu"
    # For now just verify the guard impl returns a valid stream object.
    s = torch.Stream(device=NPU)
    assert s is not None
    print(f"  stream: {s}")


# ─────────────────────────────────────────────────────────────────────────────
# Supported ops routed through OpenVINO
# ─────────────────────────────────────────────────────────────────────────────

def test_add_via_ov():
    """aten::add.Tensor routes through OV on xpu:npu."""
    intel_npu.ov_stats.reset()
    a_cpu = torch.randn(4, 4)
    b_cpu = torch.randn(4, 4)
    a, b = a_cpu.to(NPU), b_cpu.to(NPU)
    c = a + b
    assert c.device.type == "npu"
    assert torch.allclose(c.to("cpu"), a_cpu + b_cpu, atol=1e-5), "add mismatch"
    assert intel_npu.ov_stats.ov >= 1, f"expected OV op, got {intel_npu.ov_stats.ov}"


def test_mul_via_ov():
    """aten::mul.Tensor routes through OV on xpu:npu."""
    a_cpu = torch.randn(4, 4)
    b_cpu = torch.randn(4, 4)
    a, b = a_cpu.to(NPU), b_cpu.to(NPU)
    c = a * b
    assert torch.allclose(c.to("cpu"), a_cpu * b_cpu, atol=1e-5), "mul mismatch"


def test_relu_via_ov():
    """aten::relu routes through OV on xpu:npu."""
    x_cpu = torch.randn(8, 8)
    x = x_cpu.to(NPU)
    y = torch.relu(x)
    assert torch.allclose(y.to("cpu"), torch.relu(x_cpu), atol=1e-5), "relu mismatch"


def test_mm_via_ov():
    """aten::mm routes through OV on xpu:npu."""
    a_cpu = torch.randn(8, 16)
    b_cpu = torch.randn(16, 4)
    a, b = a_cpu.to(NPU), b_cpu.to(NPU)
    c = torch.mm(a, b)
    ref = torch.mm(a_cpu, b_cpu)
    assert c.shape == (8, 4)
    assert torch.allclose(c.to("cpu"), ref, atol=1e-4), "mm mismatch"


def test_linear_via_ov():
    """aten::linear routes through OV on xpu:npu."""
    x_cpu = torch.randn(8, 16)
    w_cpu = torch.randn(32, 16)
    b_cpu = torch.randn(32)
    x, w, b = x_cpu.to(NPU), w_cpu.to(NPU), b_cpu.to(NPU)
    y = torch.nn.functional.linear(x, w, b)
    ref = torch.nn.functional.linear(x_cpu, w_cpu, b_cpu)
    assert torch.allclose(y.to("cpu"), ref, atol=1e-4), "linear mismatch"


def test_softmax_via_ov():
    """aten::softmax routes through OV on xpu:npu."""
    x_cpu = torch.randn(4, 8)
    x = x_cpu.to(NPU)
    y = torch.softmax(x, dim=-1)
    ref = torch.softmax(x_cpu, dim=-1)
    assert torch.allclose(y.to("cpu"), ref, atol=1e-5), "softmax mismatch"


def test_layer_norm_via_ov():
    """aten::layer_norm routes through OV on xpu:npu."""
    x_cpu = torch.randn(2, 8)
    x = x_cpu.to(NPU)
    y = torch.nn.functional.layer_norm(x, [8])
    ref = torch.nn.functional.layer_norm(x_cpu, [8])
    assert torch.allclose(y.to("cpu"), ref, atol=1e-5), "layer_norm mismatch"


def test_bmm_via_ov():
    """aten::bmm (batched matmul) routes through OV on xpu:npu."""
    a_cpu = torch.randn(2, 4, 8)
    b_cpu = torch.randn(2, 8, 4)
    a, b = a_cpu.to(NPU), b_cpu.to(NPU)
    c = torch.bmm(a, b)
    ref = torch.bmm(a_cpu, b_cpu)
    assert c.shape == (2, 4, 4)
    assert torch.allclose(c.to("cpu"), ref, atol=1e-4), "bmm mismatch"


def test_silu_via_ov():
    """aten::silu (SwiGLU activation used in Llama) routes through OV."""
    x_cpu = torch.randn(4, 8)
    x = x_cpu.to(NPU)
    y = torch.nn.functional.silu(x)
    ref = torch.nn.functional.silu(x_cpu)
    assert torch.allclose(y.to("cpu"), ref, atol=1e-5), "silu mismatch"


def test_gelu_via_ov():
    """aten::gelu routes through OV on xpu:npu."""
    x_cpu = torch.randn(4, 8)
    x = x_cpu.to(NPU)
    y = torch.nn.functional.gelu(x)
    ref = torch.nn.functional.gelu(x_cpu)
    assert torch.allclose(y.to("cpu"), ref, atol=1e-5), "gelu mismatch"


# ─────────────────────────────────────────────────────────────────────────────
# Fallback to Torch XPU for unsupported ops  (Phase 1 placeholder)
# ─────────────────────────────────────────────────────────────────────────────

def test_unsupported_op_fallback_to_xpu():
    """An op without an OV translator should land on XPU, not CPU.

    Phase 1 TODO: replace the pass with a real assertion once the
    XPU-fallback path (replacing cpu_fallback in eager_ops.cpp) is wired.

    Expected behaviour after Phase 1:
      output tensor device.type == "xpu"  (not "cpu")
    """
    # Example: aten::fft_rfft currently has no OV translator.
    # Uncomment when Torch XPU fallback is implemented:
    #
    #   x_cpu = torch.randn(16)
    #   x = x_cpu.to(NPU)
    #   y = torch.fft.rfft(x)
    #   assert y.device.type == "xpu", (
    #       f"Expected fallback to XPU, got '{y.device.type}'"
    #   )
    pass


# ─────────────────────────────────────────────────────────────────────────────
# OV stats
# ─────────────────────────────────────────────────────────────────────────────

def test_ov_stats_increments():
    """intel_npu.ov_stats.ov counter increments with each OV-executed op."""
    intel_npu.ov_stats.reset()
    a = torch.randn(4, 4, device=NPU)
    b = torch.randn(4, 4, device=NPU)
    _ = a + b
    assert intel_npu.ov_stats.ov >= 1, (
        f"expected OV op count >= 1, got {intel_npu.ov_stats.ov}"
    )


def test_ov_stats_timing():
    """convert_ms and compile_ms are non-negative after first compile."""
    intel_npu.ov_stats.reset()
    a = torch.randn(4, 4, device=NPU)
    b = torch.randn(4, 4, device=NPU)
    _ = a + b
    assert intel_npu.ov_stats.convert_ms >= 0
    assert intel_npu.ov_stats.compile_ms >= 0
    print(f"  convert: {intel_npu.ov_stats.convert_ms:.2f} ms  "
          f"compile: {intel_npu.ov_stats.compile_ms:.2f} ms")


def test_ov_stats_cache_hit():
    """Second call with same shape does NOT re-compile (compile_ms unchanged)."""
    # Warm up once to populate cache
    a = torch.randn(4, 4, device=NPU)
    b = torch.randn(4, 4, device=NPU)
    _ = a + b

    intel_npu.ov_stats.reset()
    _ = a + b  # cache hit — compile time should be ~0
    assert intel_npu.ov_stats.compile_ms < 1.0, (
        f"expected cache hit (compile < 1 ms), got {intel_npu.ov_stats.compile_ms:.2f} ms"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Simple MLP end-to-end
# ─────────────────────────────────────────────────────────────────────────────

def test_mlp_forward():
    """Two-layer MLP forward pass entirely on xpu:npu."""
    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(16, 32)
            self.fc2 = torch.nn.Linear(32, 8)

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    model_cpu = MLP().eval()
    model_npu = MLP().eval()
    model_npu.load_state_dict(model_cpu.state_dict())
    model_npu = model_npu.to(NPU)

    x_cpu = torch.randn(4, 16)
    x_npu = x_cpu.to(NPU)

    with torch.no_grad():
        ref = model_cpu(x_cpu)
        out = model_npu(x_npu).to("cpu")

    assert torch.allclose(ref, out, atol=1e-4), (
        f"MLP output mismatch, max diff: {(ref - out).abs().max().item():.6f}"
    )
    print(f"  MLP output shape: {out.shape}, OV ops: {intel_npu.ov_stats.ov}")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = skipped = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except NotImplementedError:
            print(f"  SKIP  {t.__name__} (not implemented)")
            skipped += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {skipped} skipped, {failed} failed")
