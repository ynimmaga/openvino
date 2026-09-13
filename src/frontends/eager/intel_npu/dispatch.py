# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# dispatch.py — Stats interface for the OV eager backend.
#
# All op registrations are now done in C++ (csrc/eager_ops.cpp) using the
# same OV C++ op classes that the PyTorch frontend translators use
# (ov::op::v1::Add, ov::op::v0::Relu, etc.).  This file exposes stats.

import npu_backend


class _Stats:
    """Proxy that reads live counters from the C++ side."""

    @property
    def ov(self):
        return npu_backend.ov_op_count()

    @property
    def convert_ms(self):
        return npu_backend.ov_convert_us() / 1000.0

    @property
    def compile_ms(self):
        return npu_backend.ov_compile_us() / 1000.0

    def reset(self):
        npu_backend.reset_ov_stats()

    def __repr__(self):
        return (f"OVStats(ov={self.ov}, "
                f"convert={self.convert_ms:.2f}ms, "
                f"compile={self.compile_ms:.2f}ms)")


stats = _Stats()

