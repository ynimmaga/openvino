#!/usr/bin/env python3
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# bench_convert.py — Benchmark OV model conversion: eager mode vs graph mode.
#
# "Convert" means: ATen op(s) → ov::Model (OV IR).
#
# Eager mode  : EagerFrontEnd::convert() — translates ONE op, normalize() is
#               a no-op (no graph transformations), called per-op at runtime.
#
# Graph mode  : ov.convert_model() — translates the full traced model using the
#               standard PyTorch frontend with all ~30 normalize() passes.
#
# This benchmark isolates the CONVERSION step only (not compile, not infer).
# We measure:
#   - Eager convert latency per single op (from ov_stats.convert_ms)
#   - Graph convert latency for a model containing N identical ops
#   - Per-op amortized cost in graph mode
#
# Run:
#   cd /home/icx-6338/ynimmaga/openvino_xpu/src/frontends/eager
#   python pytorch/tests/bench_convert.py [--ops N] [--repeats R]

import sys, os, time, argparse, statistics
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import torch
import openvino as ov
import intel_npu

NPU = intel_npu.device("xpu:npu")


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────
def _args():
    p = argparse.ArgumentParser(description="Benchmark OV conversion: eager vs graph")
    p.add_argument("--repeats", type=int, default=20,
                   help="Number of repeat measurements (default: 20)")
    p.add_argument("--shape",   type=int, default=256,
                   help="Square matrix size for benchmarks (default: 256)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Graph-mode conversion helpers
# ─────────────────────────────────────────────────────────────────────────────

class _ChainModel(torch.nn.Module):
    """N sequential ops of the same type to amortize graph conversion cost."""
    def __init__(self, op, n: int):
        super().__init__()
        self.op = op
        self.n  = n

    def forward(self, a, b=None):
        x = a
        for _ in range(self.n):
            x = self.op(x, b) if b is not None else self.op(x)
        return x


def _graph_convert_ms(module, example_inputs, repeats: int) -> list[float]:
    """Measure ov.convert_model() wall time, repeated `repeats` times."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = ov.convert_model(module, example_input=example_inputs)
        times.append((time.perf_counter() - t0) * 1e3)
    return times


def _eager_convert_ms(fn, make_inputs, repeats: int) -> list[float]:
    """
    Measure eager convert_ms only (from ov_stats), forcing a cold convert
    each iteration by varying input shape (cache key includes shape).
    `make_inputs(i)` must return a tuple of tensors with shape varying per i.
    """
    times = []
    for i in range(repeats):
        args = make_inputs(i)
        intel_npu.ov_stats.reset()
        _ = fn(*args)
        # convert_ms is the time for EagerFrontEnd::convert() only
        times.append(intel_npu.ov_stats.convert_ms)
    return times


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark cases
# ─────────────────────────────────────────────────────────────────────────────

def bench_add(shape: int, repeats: int):
    """aten::add.Tensor — elementwise add."""
    print(f"\n[bench_add]  shape=({shape},{shape})  repeats={repeats}")

    # Graph mode
    class AddN(torch.nn.Module):
        def forward(self, a, b): return a + b
    ex = (torch.randn(shape, shape), torch.randn(shape, shape))
    g_times = _graph_convert_ms(AddN(), ex, repeats)

    # Eager mode (one op per convert call)
    a = torch.randn(shape, shape, device=NPU)
    b = torch.randn(shape, shape, device=NPU)
    def _mk(i):
        m = shape + i
        return (torch.randn(m, shape, device=NPU), torch.randn(m, shape, device=NPU))
    e_times = _eager_convert_ms(lambda x, y: x + y, _mk, repeats=repeats)

    _report("add", g_times, e_times)


def bench_mm(shape: int, repeats: int):
    """aten::mm — matrix multiply."""
    print(f"\n[bench_mm]  shape=({shape},{shape})  repeats={repeats}")

    class MmN(torch.nn.Module):
        def forward(self, a, b): return torch.mm(a, b)
    ex = (torch.randn(shape, shape), torch.randn(shape, shape))
    g_times = _graph_convert_ms(MmN(), ex, repeats)

    def _mk(i):
        m = shape + i
        return (torch.randn(m, shape, device=NPU), torch.randn(shape, shape, device=NPU))
    e_times = _eager_convert_ms(torch.mm, _mk, repeats=repeats)

    _report("mm", g_times, e_times)


def bench_relu(shape: int, repeats: int):
    """aten::relu — unary activation."""
    print(f"\n[bench_relu]  shape=({shape},{shape})  repeats={repeats}")

    class ReluN(torch.nn.Module):
        def forward(self, x): return torch.relu(x)
    ex = torch.randn(shape, shape)
    g_times = _graph_convert_ms(ReluN(), ex, repeats)

    def _mk(i):
        m = shape + i
        return (torch.randn(m, shape, device=NPU),)
    e_times = _eager_convert_ms(torch.relu, _mk, repeats=repeats)

    _report("relu", g_times, e_times)


def bench_softmax(shape: int, repeats: int):
    """aten::softmax — reduction + normalize."""
    print(f"\n[bench_softmax]  shape=({shape},{shape})  repeats={repeats}")

    class SoftmaxN(torch.nn.Module):
        def forward(self, x): return torch.softmax(x, dim=-1)
    ex = torch.randn(shape, shape)
    g_times = _graph_convert_ms(SoftmaxN(), ex, repeats)

    def _mk(i):
        m = shape + i
        return (torch.randn(m, shape, device=NPU),)
    e_times = _eager_convert_ms(lambda x: torch.softmax(x, dim=-1), _mk, repeats=repeats)

    _report("softmax", g_times, e_times)


def bench_linear(shape: int, repeats: int):
    """aten::linear — matmul + bias."""
    print(f"\n[bench_linear]  shape=({shape},{shape})  repeats={repeats}")

    lin = torch.nn.Linear(shape, shape)
    ex  = torch.randn(shape, shape)
    g_times = _graph_convert_ms(lin.eval(), ex, repeats)

    w = lin.weight.to(NPU)
    b = lin.bias.to(NPU)
    def _mk(i):
        m = shape + i
        return (torch.randn(m, shape, device=NPU), w, b)
    e_times = _eager_convert_ms(
        lambda x, w, b: torch.nn.functional.linear(x, w, b), _mk,
        repeats=repeats)

    _report("linear", g_times, e_times)


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def _report(name: str, graph_times: list, eager_times: list):
    def _s(ts):
        ts_sorted = sorted(ts)
        return (statistics.mean(ts_sorted),
                statistics.median(ts_sorted),
                ts_sorted[0],
                ts_sorted[-1])

    g_mean, g_p50, g_min, g_max = _s(graph_times)
    e_mean, e_p50, e_min, e_max = _s(eager_times)
    ratio = g_mean / e_mean if e_mean > 0 else float("inf")

    print(f"  {'':30s}  {'mean':>8}  {'p50':>8}  {'min':>8}  {'max':>8}")
    print(f"  {'graph convert_ms':30s}  {g_mean:8.2f}  {g_p50:8.2f}"
          f"  {g_min:8.2f}  {g_max:8.2f}")
    print(f"  {'eager convert_ms (per-op)':30s}  {e_mean:8.2f}  {e_p50:8.2f}"
          f"  {e_min:8.2f}  {e_max:8.2f}")
    print(f"  graph/eager ratio: {ratio:.1f}x  "
          f"(graph is {'faster' if ratio < 1 else 'slower'} per individual convert)")
    print()


def _summary_table(results: dict):
    print("\n" + "=" * 70)
    print(f"  {'Op':15s}  {'Graph mean (ms)':>18}  {'Eager mean (ms)':>18}  {'Ratio':>8}")
    print("=" * 70)
    for name, (g, e) in results.items():
        ratio = g / e if e > 0 else float("inf")
        print(f"  {name:15s}  {g:18.2f}  {e:18.2f}  {ratio:8.1f}x")
    print("=" * 70)
    print("  Ratio > 1 means graph convert is slower per-op (expected for single ops).")
    print("  Eager converts one op at a time; graph converts the whole model at once")
    print("  and amortizes normalize() cost across all ops in the model.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = _args()
    R = args.repeats
    S = args.shape

    print(f"Benchmark: OV conversion — eager vs graph mode")
    print(f"  shape={S}x{S}  repeats={R}")
    print(f"  device={NPU}\n")

    results = {}

    def _run(name, fn, s, r):
        import io, contextlib
        # Capture per-op mean from bench function
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fn(s, r)
        output = buf.getvalue()
        sys.stdout.write(output)
        # Parse mean times from output lines
        def _first_float(line):
            for tok in line.split():
                try:
                    return float(tok)
                except ValueError:
                    continue
            return 0.0
        lines = [l for l in output.splitlines() if "convert_ms" in l]
        g_mean = _first_float(lines[0]) if len(lines) > 0 else 0.0
        e_mean = _first_float(lines[1]) if len(lines) > 1 else 0.0
        results[name] = (g_mean, e_mean)

    _run("add",      bench_add,      S, R)
    _run("mm",       bench_mm,       S, R)
    _run("relu",     bench_relu,     S, R)
    _run("softmax",  bench_softmax,  S, R)
    _run("linear",   bench_linear,   S, R)

    _summary_table(results)
