# OpenVINO Eager NPU Backend for PyTorch

A PyTorch `PrivateUse1` device (exposed as `npu` / `xpu:npu`) that transparently
routes every aten op through the OpenVINO PyTorch frontend and runtime. No
graph capture, no `torch.compile`, no model rewriting — users only change the
device string.

```python
import torch, intel_npu
dev = intel_npu.device("xpu:npu")          # or simply "npu"
x = torch.randn(4, 8, device=dev)
y = torch.softmax(x @ x.T, dim=-1)         # every op runs through OpenVINO
```

---

## Directory Structure

```
eager/
├── README.md                       ← this file
├── setup.py                        ← builds the C++ extension `npu_backend`
├── intel_npu/                      ← Python package
│   ├── __init__.py                 ← registers PrivateUse1 → "npu", loads ext
│   └── dispatch.py                 ← optional Python-side op overrides + stats
├── common/
│   └── src/
│       └── npu_backend.cpp         ← device infra: allocator, guard, empty,
│                                     copy_, view, as_strided, _to_copy
├── pytorch/
│   ├── include/                    ← OpenVINO PT-frontend public headers
│   ├── src/
│   │   ├── eager_decoder.h         ← TorchDecoder shim (graph/op/const)
│   │   │                             that fakes a one-op TorchScript graph
│   │   │                             so FE translators can be reused
│   │   └── eager_ops.cpp           ← boxed PrivateUse1 fallback:
│   │                                 parses stack → builds ov::Model →
│   │                                 compiles → caches → infers → copies out
│   └── tests/
│       ├── test_correctness.py     ← per-op CPU-vs-NPU numerical parity
│       ├── test_single_op_e2e.py   ← targeted single-op regression cases
│       ├── test_compare_conversion.py
│       ├── test_eager.py           ← basic eager mode smoke tests
│       ├── test_llm_xpu_npu.py     ← HF LLM end-to-end (the user-facing demo)
│       ├── test_llm_xpu_npu_ops.py ← LLM run + per-op coverage table
│       ├── test_llm_safetensor.py
│       ├── test_xpu_npu.py         ← lower-level device tests
│       ├── test_xpu_stream_npu.py
│       ├── bench_convert.py        ← microbench for FE convert+compile cost
│       ├── bench_e2e.py            ← end-to-end throughput bench
│       └── analyze_npu_trace.py    ← post-mortem analysis of op stats
├── onnx/                           ← (placeholder for future ONNX FE wiring)
└── tflite/                         ← (placeholder for future TFLite FE wiring)
```

---

## Execution Flow

```
            ┌──────────────────────────────────────────────────────────────┐
   torch    │ user code:  x.to("npu");  y = torch.softmax(x @ x.T, -1)    │
            └──────────────────────────────────────────────────────────────┘
                                     │
                       PyTorch dispatcher (PrivateUse1)
                                     │
        ┌────────────────────────────┴───────────────────────────┐
        │                                                        │
   layout/copy ops                                       every other aten op
   (view, as_strided,                                            │
    empty, copy_, _to_copy, …)                                   │
        │                                            ┌───────────▼───────────┐
        ▼                                            │ npu_generic_fallback  │
   npu_backend.cpp                                   │  (eager_ops.cpp)      │
   (in-place memcpy /                                └───────────┬───────────┘
    stride math, no OV)                                          │
                                                ┌────────────────┼─────────────────┐
                                                │                │                 │
                                  canonicalize_op            list-typed?   FE has translator?
                                  (_softmax→softmax,             │                 │
                                   drop noise args)              ▼                 ▼
                                                       try_build_custom_model   build_model_for_op
                                                       (direct ov::Concat /     (EagerGraphDecoder
                                                        ReduceMean / etc.)       → FE.convert →
                                                                │                 real translator)
                                                                └────────┬────────┘
                                                                         ▼
                                                              core.compile_model("CPU")
                                                                         │
                                                                         ▼
                                                            cache by (op,dtypes,shapes,
                                                                       attrs hex)
                                                                         │
                                                                         ▼
                                                       bind inputs (memcpy or stride-walk)
                                                                         │
                                                                         ▼
                                                                infer_request.infer()
                                                                         │
                                                                         ▼
                                                       copy outputs into NPU tensors
                                                       (stride policy: dense-permuted
                                                        from ref input, or contiguous
                                                        for clone/contiguous/_to_copy)
                                                                         │
                                                                         ▼
                                                       result back on the dispatcher stack
```

### Per-call cost
- **Cache miss (first call with that op + shape + dtype):** decoder construction + FE convert + `compile_model` (ms-scale; `compile_model` dominates).
- **Cache hit (every subsequent call):** input memcpy + `infer()` + output memcpy.
- Cache is process-global, in-memory only. Restart = recompile.

### When does the CPU fallback fire?
- FE has no translator for the op (e.g. `aten::isin`, `aten::isneginf`).
- Stack contains a kind we don't parse (string, dict, generator).
- FE/OV throws during convert or compile.
- No NPU tensor among the inputs (e.g. `aten::arange` with all scalars).

`per_op_exec_counts()` reports `aten_name → (ov_calls, cpu_calls)` so you can
see exactly where each op went.

---

## Build & Setup

### Prerequisites
- Python 3.10 in a virtualenv (`$VIRTUAL_ENV` must point to it).
- PyTorch 2.x (`torch.utils.rename_privateuse1_backend` and
  `generate_methods_for_privateuse1_backend` must exist).
- OpenVINO 2026.x installed in the same venv as `pip install openvino` so
  that `$VIRTUAL_ENV/lib/python3.10/site-packages/openvino/` exists with
  `include/` and `libs/libopenvino*.so.2610`.
- A C++17 compiler.

### Build

```bash
cd /home/icx-6338/ynimmaga/openvino_xpu/src/frontends/eager
python setup.py install
```

This builds the C++ extension `npu_backend` from
`common/src/npu_backend.cpp` + `pytorch/src/eager_ops.cpp`, links it against
`libopenvino.so.2610` and `libopenvino_pytorch_frontend.so.2610`, and installs
the `intel_npu` package together with an `intel_npu.pth` so it auto-loads on
interpreter start.

### Iterative dev build (avoid full reinstall)

After modifying any source file:

```bash
cd /home/icx-6338/ynimmaga/openvino_xpu/src/frontends/eager
touch common/src/npu_backend.cpp pytorch/src/eager_ops.cpp pytorch/src/eager_decoder.h
python setup.py install
```

If you'd rather work in-place without `pip install`:

```bash
printf '%s\n' \
  '/home/icx-6338/ynimmaga/openvino_xpu/src/frontends/eager' \
  'import intel_npu' \
  > $VIRTUAL_ENV/lib/python3.10/site-packages/intel_npu.pth
```

### Verify
```bash
python -c "import torch, intel_npu; print(torch.randn(2,3, device='npu') + 1)"
```

---

## Testing

All tests live in `pytorch/tests/`. Run from the `eager/` directory.

### Per-op numerical parity
```bash
python pytorch/tests/test_correctness.py                # all ops
python pytorch/tests/test_correctness.py --verbose      # per-op diffs
python pytorch/tests/test_correctness.py --atol 1e-3
```

### Single-op regression cases
```bash
python pytorch/tests/test_single_op_e2e.py
```

### Eager mode smoke
```bash
python pytorch/tests/test_eager.py
```

### End-to-end LLM
```bash
HF_MODEL=meta-llama/Llama-3.2-1B  python pytorch/tests/test_llm_xpu_npu.py
HF_MODEL=Qwen/Qwen2.5-0.5B        python pytorch/tests/test_llm_xpu_npu.py
HF_MODEL=Qwen/Qwen3-0.6B          python pytorch/tests/test_llm_xpu_npu.py
HF_MODEL=gpt2                     python pytorch/tests/test_llm_xpu_npu.py
```

### Op-coverage table (which ops ran on OV vs CPU)
```bash
python pytorch/tests/test_llm_xpu_npu_ops.py
```

### Microbenchmarks
```bash
python pytorch/tests/bench_convert.py     # FE convert + compile cost
python pytorch/tests/bench_e2e.py         # end-to-end token/s
```

### Stats from any script
```python
import npu_backend
print(npu_backend.ov_op_count())          # total OV-routed calls
print(npu_backend.ov_compile_us())        # cumulative compile time (µs)
print(npu_backend.per_op_exec_counts())   # {aten_name: (ov_calls, cpu_calls)}
print(npu_backend.per_op_ov_ops())        # {aten_name: [ov_op_types,...]}
npu_backend.reset_ov_stats()
```

---

## Adding Support for a New Op

1. **Check FE coverage first.**
   ```python
   import npu_backend
   "aten::your_op" in npu_backend.supported_op_names()
   ```
   If True, the op already works through the generic boxed fallback — just run
   it and confirm `per_op_exec_counts()` shows it under `ov_calls`.

2. **Op falls back because of a noise arg / rename.** Add an entry to
   `canonicalize_op()` in `eager_ops.cpp` (rewrite name, list arg indices to
   drop). Pattern: `aten::_softmax → aten::softmax`, drop `{2}`.

3. **Op needs a list-typed input the FE expects via `prim::ListConstruct`.**
   Extend `try_build_custom_model()` in `eager_ops.cpp` to construct the OV
   graph directly. Pattern: `aten::cat`, `aten::mean.dim`, `aten::stack`.

4. **Op is layout/storage-only** (view-like, in-place fill, etc.). Add a
   targeted kernel in `npu_backend.cpp` and register it inside the
   `TORCH_LIBRARY_IMPL(aten, PrivateUse1, …)` block.

5. **Rebuild and re-run `test_correctness.py`** for the relevant op.

---

## Implementation Notes

- **Storage:** the NPU allocator returns plain CPU memory tagged with
  `PrivateUse1`. All "device transfers" are `memcpy`. The OV runtime currently
  compiles to the **CPU plugin**; the device tag is just a dispatch routing
  trick.
- **Strides:** `npu_view`, `npu_as_strided`, `_to_copy`, `copy_`, and the
  output-materialization path in `execute_via_ov` all preserve PyTorch's
  stride conventions. Output strides for pointwise ops use
  `at::infer_dense_strides(ref_input.sizes(), ref_input.strides())` to match
  the dispatcher's restride expectations; otherwise downstream `as_strided`
  reads garbage.
- **Recursion guard:** never call `t.contiguous()` on an NPU tensor from
  inside the fallback — it dispatches `aten::clone` and re-enters. Use the
  internal `make_contiguous_npu()` helper or stride-walk directly.
- **Cache key:** op name + per-input `(dtype, shape)` + per-Constant raw bytes
  (hex) + tensor-list shape signature. This is what makes
  `mean(x, dim=1)` and `mean(x, dim=2)` two cache entries instead of one.
- **`.out` variants:** writable-aliased args (`Tensor(a!)`) are detected via
  `arg.alias_info().isWrite()`, captured separately, and the OV result is
  copied back into them after `infer()`.
