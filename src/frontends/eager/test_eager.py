#!/usr/bin/env python3
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# test_eager.py — Test OpenVINO NPU backend with PrivateUse1 device.
#
# No special context manager needed — just import intel_npu once
# (or rely on the auto-loading .pth after pip install) and use
# device="npu" like any other accelerator.

import sys
import os
# Add the eager/ directory to path so both intel_npu and npu_backend are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

# Register NPU backend (after pip install, this happens automatically)
import intel_npu

NPU = torch.device("npu", 0)


def test_device_registration():
    """Verify 'npu' device is registered via PrivateUse1."""
    d = torch.device("npu")
    assert d.type == "npu", f"Expected 'npu', got '{d.type}'"
    d0 = torch.device("npu:0")
    assert d0.type == "npu" and d0.index == 0
    print(f"  npu device: {d}, npu:0 device: {d0}")
    print(f"  is_available: {torch.npu.is_available()}")
    print(f"  device_count: {torch.npu.device_count()}")


def test_xpu_npu_syntax():
    """Verify 'xpu:npu' maps to the NPU device."""
    dev = intel_npu.device("xpu:npu")
    assert dev.type == "npu", f"Expected 'npu', got '{dev.type}'"
    print(f"  intel_npu.device('xpu:npu') = {dev}")


def test_tensor_creation():
    """Create tensors on NPU device."""
    x = torch.empty(3, 4, device=NPU)
    assert x.device.type == "npu", f"Expected npu, got {x.device}"
    assert x.shape == (3, 4)

    y = torch.zeros(2, 2, device=NPU)
    assert y.device.type == "npu"
    y_cpu = y.to("cpu")
    assert torch.all(y_cpu == 0)

    z = torch.ones(5, device=NPU, dtype=torch.float64)
    assert z.device.type == "npu"
    assert z.dtype == torch.float64


def test_cpu_npu_transfer():
    """Transfer tensors between CPU and NPU."""
    cpu_t = torch.randn(4, 4)
    npu_t = cpu_t.to(NPU)
    assert npu_t.device.type == "npu"

    back = npu_t.to("cpu")
    assert back.device == torch.device("cpu")
    assert torch.allclose(cpu_t, back, atol=1e-6), "roundtrip mismatch"


def test_add_via_ov():
    """aten::add.Tensor — should go through OV automatically."""
    a_cpu = torch.randn(3, 4)
    b_cpu = torch.randn(3, 4)
    a = a_cpu.to(NPU)
    b = b_cpu.to(NPU)

    c = a + b  # no context manager needed

    assert c.device.type == "npu"
    ref = a_cpu + b_cpu
    assert torch.allclose(c.to("cpu"), ref, atol=1e-5), "add mismatch"


def test_mul_via_ov():
    """aten::mul.Tensor via OpenVINO."""
    a_cpu = torch.randn(3, 4)
    b_cpu = torch.randn(3, 4)
    a = a_cpu.to(NPU)
    b = b_cpu.to(NPU)

    c = a * b

    ref = a_cpu * b_cpu
    assert torch.allclose(c.to("cpu"), ref, atol=1e-5), "mul mismatch"


def test_relu_via_ov():
    """aten::relu via OpenVINO."""
    x_cpu = torch.randn(4, 4)
    x = x_cpu.to(NPU)

    y = torch.relu(x)

    ref = torch.relu(x_cpu)
    assert torch.allclose(y.to("cpu"), ref, atol=1e-5), "relu mismatch"


def test_mm_via_ov():
    """aten::mm via OpenVINO."""
    a_cpu = torch.randn(4, 8)
    b_cpu = torch.randn(8, 3)
    a = a_cpu.to(NPU)
    b = b_cpu.to(NPU)

    c = torch.mm(a, b)

    ref = torch.mm(a_cpu, b_cpu)
    assert c.shape == (4, 3)
    assert torch.allclose(c.to("cpu"), ref, atol=1e-4), "mm mismatch"


def test_sigmoid_via_ov():
    """aten::sigmoid via OpenVINO."""
    x_cpu = torch.randn(3, 3)
    x = x_cpu.to(NPU)

    y = torch.sigmoid(x)

    ref = torch.sigmoid(x_cpu)
    assert torch.allclose(y.to("cpu"), ref, atol=1e-5), "sigmoid mismatch"


def test_fallback_to_cpu():
    """Unsupported op falls back to CPU via the C++ generic fallback."""
    x_cpu = torch.randn(4, 4)
    x = x_cpu.to(NPU)

    # lgamma is not in the OV frontend op_table → C++ CPU fallback handles it
    y = torch.lgamma(x)

    ref = torch.lgamma(x_cpu)
    assert torch.allclose(y.to("cpu"), ref, atol=1e-5), "fallback lgamma mismatch"


def test_mixed_computation():
    """Mix of OV-accelerated and fallback ops — no context manager."""
    x_cpu = torch.randn(4, 4)
    w_cpu = torch.randn(4, 4)
    bias_cpu = torch.randn(4)

    x = x_cpu.to(NPU)
    w = w_cpu.to(NPU)
    bias = bias_cpu.to(NPU)

    out = torch.mm(x, w)     # OV
    out = out + bias          # OV
    out = torch.sigmoid(out)  # OV

    ref = torch.sigmoid(torch.mm(x_cpu, w_cpu) + bias_cpu)
    assert torch.allclose(out.to("cpu"), ref, atol=1e-4), "mixed mismatch"
    print(f"  ov_stats: {intel_npu.ov_stats}")


def test_dtypes():
    """Multiple dtypes."""
    for dtype in [torch.float32, torch.float64]:
        a_cpu = torch.randn(4, 4, dtype=dtype)
        b_cpu = torch.randn(4, 4, dtype=dtype)
        a = a_cpu.to(NPU)
        b = b_cpu.to(NPU)

        c = a + b

        ref = a_cpu + b_cpu
        assert torch.allclose(c.to("cpu"), ref, atol=1e-5), f"dtype={dtype} mismatch"
        print(f"    {dtype} ok")


if __name__ == "__main__":
    tests = [
        ("Device Registration", test_device_registration),
        ("xpu:npu Syntax", test_xpu_npu_syntax),
        ("Tensor Creation", test_tensor_creation),
        ("CPU <-> NPU Transfer", test_cpu_npu_transfer),
        ("Add via OV", test_add_via_ov),
        ("Mul via OV", test_mul_via_ov),
        ("ReLU via OV", test_relu_via_ov),
        ("MatMul via OV", test_mm_via_ov),
        ("Sigmoid via OV", test_sigmoid_via_ov),
        ("Fallback to CPU", test_fallback_to_cpu),
        ("Mixed Computation", test_mixed_computation),
        ("Multiple Dtypes", test_dtypes),
    ]

    print("=" * 60)
    print("OpenVINO NPU Backend — Accelerator Tests")
    print("=" * 60)

    passed = failed = 0
    for name, fn in tests:
        print(f"\n--- {name} ---")
        try:
            fn()
            print(f"PASS: {name}")
            passed += 1
        except Exception as e:
            print(f"FAIL: {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print(f"{'=' * 60}")
