#!/usr/bin/env python3
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# bench_e2e.py — End-to-end benchmark: eager mode vs graph mode.
#
# Measures the full pipeline (convert + compile + infer) for:
#
#   Eager mode  : per-op dispatch via PrivateUse1 → EagerFrontEnd::convert()
#                 → compile_model (cached after first call) → infer().
#                 No graph transformations.  Compile is amortized via cache.
#
#   Graph mode  : ov.convert_model() (full model, with normalize()) →
#                 compile_model → InferRequest → infer().
#                 Compile is per-model (done once before the benchmark loop).
#
# Phases timed separately:
#   Phase 0 — first-call cost  (cold convert + compile for eager;
#                                convert + compile for graph)
#   Phase 1 — warm infer loop  (infer only, both modes)
#
# The key insight:
#   * Eager has higher FIRST-CALL latency (per-op convert+compile at runtime).
#   * After the cache is warm, eager infer latency ≈ graph infer latency
#     because both call the same OV InferRequest::infer() path.
#   * For LLM decode loops (same shapes repeated), the cache amortizes cost.
#
# Run:
#   cd /home/icx-6338/ynimmaga/openvino_xpu/src/frontends/eager
#   python pytorch/tests/bench_e2e.py [--shape N] [--repeats R] [--warmup W]

import sys, os, time, argparse, statistics
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import torch
import openvino as ov
import intel_npu

NPU     = intel_npu.device("xpu:npu")
OV_CORE = ov.Core()


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────
def _args():
    p = argparse.ArgumentParser(description="E2E benchmark: eager vs graph mode")
    p.add_argument("--shape",   type=int, default=512)
    p.add_argument("--repeats", type=int, default=50,
                   help="Warm infer iterations (default: 50)")
    p.add_argument("--warmup",  type=int, default=5,
                   help="Warm-up iterations before timing (default: 5)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Graph-mode baseline: compile once, infer N times
# ─────────────────────────────────────────────────────────────────────────────

def _graph_mode_bench(module, ex_inputs_cpu, npu_inputs, repeats, warmup):
    """
    Full graph-mode benchmark.
    Returns dict with cold_ms, warm_mean_ms, warm_p50_ms, warm_min_ms.
    """
    # --- Cold: convert + compile ---
    t0 = time.perf_counter()
    ov_model  = ov.convert_model(module, example_input=ex_inputs_cpu)
    compiled  = OV_CORE.compile_model(ov_model, "CPU")   # "CPU" = NPU simulation
    infer_req = compiled.create_infer_request()
    cold_ms   = (time.perf_counter() - t0) * 1e3

    # --- Set inputs (OV tensors wrapping npu/cpu data) ---
    def _set_and_infer():
        for i, t in enumerate(npu_inputs):
            # Wrap via numpy (zero-copy for fp32). For benchmark purposes this
            # is fine; the eager path uses zero-copy via data_ptr in C++.
            arr = t.detach().to("cpu").contiguous().numpy()
            ov_t = ov.Tensor(arr)
            infer_req.set_input_tensor(i, ov_t)
        infer_req.infer()

    # Warm-up
    for _ in range(warmup):
        _set_and_infer()

    # Timed loop
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _set_and_infer()
        times.append((time.perf_counter() - t0) * 1e3)

    return {
        "cold_ms":     cold_ms,
        "warm_mean":   statistics.mean(times),
        "warm_p50":    statistics.median(times),
        "warm_min":    min(times),
        "warm_max":    max(times),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Eager-mode benchmark: per-op via NPU dispatch (compile cached after first)
# ─────────────────────────────────────────────────────────────────────────────

def _eager_mode_bench(fn, npu_inputs, repeats, warmup):
    """
    Eager-mode benchmark.
    `fn` is a callable that takes *npu_inputs and returns a tensor on NPU.
    """
    # --- Cold: first call (cache miss → convert + compile + infer) ---
    intel_npu.ov_stats.reset()
    t0 = time.perf_counter()
    _  = fn(*npu_inputs)
    torch.npu.synchronize()
    cold_total_ms  = (time.perf_counter() - t0) * 1e3
    cold_convert   = intel_npu.ov_stats.convert_ms
    cold_compile   = intel_npu.ov_stats.compile_ms

    # Warm-up (cache hits)
    for _ in range(warmup):
        _ = fn(*npu_inputs)
    torch.npu.synchronize()

    # Timed warm loop (cache hits — infer only)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _  = fn(*npu_inputs)
        torch.npu.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)

    return {
        "cold_ms":      cold_total_ms,
        "cold_convert": cold_convert,
        "cold_compile": cold_compile,
        "warm_mean":    statistics.mean(times),
        "warm_p50":     statistics.median(times),
        "warm_min":     min(times),
        "warm_max":     max(times),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark cases
# ─────────────────────────────────────────────────────────────────────────────

def bench_mm(shape, repeats, warmup):
    print(f"\n{'='*60}")
    print(f"[bench_mm]  shape=({shape},{shape}x{shape})")

    class MM(torch.nn.Module):
        def forward(self, a, b): return torch.mm(a, b)

    a_cpu = torch.randn(shape, shape)
    b_cpu = torch.randn(shape, shape)
    a_npu = a_cpu.to(NPU)
    b_npu = b_cpu.to(NPU)

    g = _graph_mode_bench(MM().eval(), (a_cpu, b_cpu), [a_cpu, b_cpu], repeats, warmup)
    e = _eager_mode_bench(torch.mm, [a_npu, b_npu], repeats, warmup)
    _print_comparison("aten::mm", g, e)


def bench_add(shape, repeats, warmup):
    print(f"\n{'='*60}")
    print(f"[bench_add]  shape=({shape},{shape})")

    class Add(torch.nn.Module):
        def forward(self, a, b): return a + b

    a_cpu = torch.randn(shape, shape)
    b_cpu = torch.randn(shape, shape)
    a_npu = a_cpu.to(NPU)
    b_npu = b_cpu.to(NPU)

    g = _graph_mode_bench(Add().eval(), (a_cpu, b_cpu), [a_cpu, b_cpu], repeats, warmup)
    e = _eager_mode_bench(lambda a, b: a + b, [a_npu, b_npu], repeats, warmup)
    _print_comparison("aten::add", g, e)


def bench_linear(shape, repeats, warmup):
    print(f"\n{'='*60}")
    print(f"[bench_linear]  shape=({shape},{shape})")

    lin   = torch.nn.Linear(shape, shape).eval()
    x_cpu = torch.randn(shape, shape)
    x_npu = x_cpu.to(NPU)
    w_npu = lin.weight.to(NPU)
    b_npu = lin.bias.to(NPU)

    g = _graph_mode_bench(lin, x_cpu, [x_cpu], repeats, warmup)
    e = _eager_mode_bench(
        lambda x, w, b: torch.nn.functional.linear(x, w, b),
        [x_npu, w_npu, b_npu], repeats, warmup)
    _print_comparison("aten::linear", g, e)


def bench_softmax(shape, repeats, warmup):
    print(f"\n{'='*60}")
    print(f"[bench_softmax]  shape=({shape},{shape})")

    class Softmax(torch.nn.Module):
        def forward(self, x): return torch.softmax(x, dim=-1)

    x_cpu = torch.randn(shape, shape)
    x_npu = x_cpu.to(NPU)

    g = _graph_mode_bench(Softmax().eval(), x_cpu, [x_cpu], repeats, warmup)
    e = _eager_mode_bench(lambda x: torch.softmax(x, dim=-1), [x_npu], repeats, warmup)
    _print_comparison("aten::softmax", g, e)


def bench_mlp(shape, repeats, warmup):
    """Multi-op MLP: linear → relu → linear.  Graph has 2 ops; eager compiles each separately."""
    print(f"\n{'='*60}")
    print(f"[bench_mlp]  input=({shape},{shape})  hidden={shape*2}  out={shape}")

    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(shape, shape * 2)
            self.fc2 = torch.nn.Linear(shape * 2, shape)
        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    mlp   = MLP().eval()
    x_cpu = torch.randn(shape, shape)
    x_npu = x_cpu.to(NPU)
    mlp_npu = MLP().eval()
    mlp_npu.load_state_dict(mlp.state_dict())
    mlp_npu = mlp_npu.to(NPU).eval()

    g = _graph_mode_bench(mlp, x_cpu, [x_cpu], repeats, warmup)
    with torch.no_grad():
        e = _eager_mode_bench(lambda x: mlp_npu(x), [x_npu], repeats, warmup)
    _print_comparison("MLP (linear+relu+linear)", g, e, note="eager compiles 3 ops separately")


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def _print_comparison(name, g, e, note=""):
    cold_ratio = g["cold_ms"] / e["cold_ms"] if e["cold_ms"] > 0 else float("inf")
    warm_ratio = g["warm_mean"] / e["warm_mean"] if e["warm_mean"] > 0 else float("inf")

    print(f"\n  Op: {name}  {('(' + note + ')') if note else ''}")
    print(f"  {'':40s}  {'graph':>10}  {'eager':>10}")
    print(f"  {'Cold (convert+compile+1×infer) ms':40s}  {g['cold_ms']:10.2f}"
          f"  {e['cold_ms']:10.2f}  ratio={cold_ratio:.2f}x")
    print(f"  {'  └ convert ms':40s}  {'--':>10}  {e['cold_convert']:10.2f}")
    print(f"  {'  └ compile ms':40s}  {'--':>10}  {e['cold_compile']:10.2f}")
    print(f"  {'Warm infer mean ms':40s}  {g['warm_mean']:10.3f}  {e['warm_mean']:10.3f}"
          f"  ratio={warm_ratio:.2f}x")
    print(f"  {'Warm infer p50 ms':40s}  {g['warm_p50']:10.3f}  {e['warm_p50']:10.3f}")
    print(f"  {'Warm infer min ms':40s}  {g['warm_min']:10.3f}  {e['warm_min']:10.3f}")


def _print_dtype(t: torch.dtype) -> ov.Type:
    return _pt_to_ov_dtype(t)


def _pt_to_ov_dtype(dt: torch.dtype) -> ov.Type:
    _map = {
        torch.float32: ov.Type.f32,
        torch.float64: ov.Type.f64,
        torch.float16: ov.Type.f16,
        torch.bfloat16: ov.Type.bf16,
        torch.int32:  ov.Type.i32,
        torch.int64:  ov.Type.i64,
        torch.bool:   ov.Type.boolean,
    }
    return _map.get(dt, ov.Type.f32)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = _args()
    S, R, W = args.shape, args.repeats, args.warmup

    print(f"E2E Benchmark: eager vs graph mode  (NPU simulated on CPU)")
    print(f"  shape={S}  repeats={R}  warmup={W}")
    print(f"  device={NPU}")

    bench_add(S, R, W)
    bench_mm(S, R, W)
    bench_linear(S, R, W)
    bench_softmax(S, R, W)
    bench_mlp(S, R, W)

    print(f"\n{'='*60}")
    print("Key takeaways:")
    print("  * Cold cost: eager pays convert+compile per op at first call.")
    print("               graph pays once for the whole model.")
    print("  * Warm cost: both modes call the same OV InferRequest::infer().")
    print("               Eager warm ≈ graph warm — cache amortizes cold cost.")
    print("  * LLM decode: shapes repeat → eager cache hits dominate → parity.")
