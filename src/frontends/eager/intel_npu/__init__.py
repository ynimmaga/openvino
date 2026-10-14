# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# __init__.py — OpenVINO NPU backend for PyTorch.
#
# Registers "npu" as a PrivateUse1 device. Usage:
#   import intel_npu
#   x = torch.randn(3, 4, device="npu")
#   y = x + x  # dispatched via OpenVINO
#
# For "xpu:npu" syntax:
#   dev = intel_npu.device("xpu:npu")
#   x = torch.randn(3, 4, device=dev)

import torch

# ── 1. Register "npu" as PrivateUse1 backend ──────────────────────────────
torch.utils.rename_privateuse1_backend("npu")

# Load C++ extension (allocator, guard, empty, copy_)
import npu_backend  # noqa: F401

# Generate .npu(), .is_npu, .to("npu") etc. on Tensor
torch.utils.generate_methods_for_privateuse1_backend()

# Expose torch.npu module
class _NpuModule:
    @staticmethod
    def is_available():
        return npu_backend.is_available()

    @staticmethod
    def device_count():
        return npu_backend.device_count()

    @staticmethod
    def current_device():
        return npu_backend.current_device()

    @staticmethod
    def synchronize(device=None):
        pass  # CPU-backed, always synchronous

torch._register_device_module("npu", _NpuModule())


# ── 2. Device helper for "xpu:npu" syntax ─────────────────────────────────

def device(spec="npu"):
    """Parse device spec. Accepts 'npu', 'npu:0', or 'xpu:npu'."""
    if isinstance(spec, str) and spec.lower() in ("xpu:npu",):
        return torch.device("npu", 0)
    return torch.device(spec)


# ── 3. Register OV-backed ops at PrivateUse1 dispatch key ─────────────────
from . import dispatch  # noqa: F401, E402  — triggers torch.library.impl registrations

# Expose stats for introspection
ov_stats = dispatch.stats

__all__ = ["device", "ov_stats"]
