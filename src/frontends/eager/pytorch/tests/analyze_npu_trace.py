#!/usr/bin/env python
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# analyze_npu_trace.py
# ─────────────────────────────────────────────────────────────────────────────
# Reads the on-disk op trace produced by test_llm_xpu_npu_ops.py
# (default /tmp/llm_npu_trace.log) and prints the same per-op summary even
# if the test process segfaulted.  Useful for hard crashes inside an OV op.
#
# Usage:
#   python pytorch/tests/analyze_npu_trace.py [trace_file]
# ─────────────────────────────────────────────────────────────────────────────
import sys
from collections import Counter

PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/llm_npu_trace.log"

ov_calls       = Counter()
fallback_calls = Counter()
last_line      = None

with open(PATH) as f:
    for line in f:
        last_line = line.rstrip("\n")
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2:
            continue
        route, aten_name = parts[0], parts[1]
        if route == "OV":
            ov_calls[aten_name] += 1
        elif route == "FALLBACK":
            fallback_calls[aten_name] += 1

print(f"trace file  : {PATH}")
print(f"total lines : {sum(ov_calls.values()) + sum(fallback_calls.values())}")
print(f"last op     : {last_line}")
print()

print(f"─── OV-routed aten ops ({sum(ov_calls.values())} total calls, "
      f"{len(ov_calls)} unique) ───")
for aten, n in sorted(ov_calls.items(), key=lambda kv: -kv[1]):
    print(f"  {aten:32s}  x{n}")
print()

print(f"─── Fallback aten ops ({sum(fallback_calls.values())} total calls, "
      f"{len(fallback_calls)} unique) ───")
for aten, n in sorted(fallback_calls.items(), key=lambda kv: -kv[1]):
    print(f"  {aten:32s}  x{n}")
