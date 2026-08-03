# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# test_llm_xpu_npu_ops.py
# ─────────────────────────────────────────────────────────────────────────────
# Adapted from the canonical HF snippet for meta-llama/Llama-3.2-1B-Instruct
# (https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct).
#
# What this script does:
#   1. Loads Llama-3.2-1B-Instruct via transformers.
#   2. Moves the model + inputs onto the NPU via torch.device("xpu:npu")
#      (resolved by intel_npu.device(...) → torch.device("npu", 0)).
#   3. Hooks the dispatch path with a TorchDispatchMode so that for every aten
#      op we can tell whether it ran through OpenVINO or fell back to torch
#      (CPU/XPU). For OV-routed ops it also dumps the OV op types created by
#      the PyTorch frontend translator.
#   4. Runs a short generate() and prints:
#        - per-aten-op hit / fallback counts
#        - the OV op subgraph created for each unique aten op
#        - a list of aten ops that were NOT supported by the frontend and
#          therefore fell back to torch xpu/cpu
#
# Run:
#   export HF_TOKEN=...                   # if needed for the gated model
#   python pytorch/tests/test_llm_xpu_npu_ops.py
#
# Override model:
#   HF_MODEL=meta-llama/Llama-3.2-1B-Instruct  python ...
#   HF_MODEL=hf-internal-testing/tiny-random-LlamaForCausalLM  python ...
#   MAX_NEW_TOKENS=8  python ...
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import time
from collections import Counter, defaultdict

import torch
import intel_npu          # registers the "npu" PrivateUse1 backend
import npu_backend        # C++ bindings: supported_op_names(), per_op_ov_ops()

from torch.utils._python_dispatch import TorchDispatchMode

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_ID       = os.environ.get("HF_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "16"))
_DTYPE_ENV     = os.environ.get("DTYPE", "bfloat16").lower()
DTYPE          = {
    "bfloat16": torch.bfloat16,
    "float16":  torch.float16,
    "float32":  torch.float32,
}.get(_DTYPE_ENV, torch.bfloat16)

# Resolve "xpu:npu" → torch.device("npu", 0) via the helper
NPU = intel_npu.device("xpu:npu")

print(f"[setup] HF model      : {MODEL_ID}")
print(f"[setup] device spec   : torch.device('xpu:npu')  →  {NPU}")
print(f"[setup] dtype         : {DTYPE}")
print(f"[setup] max_new_tokens: {MAX_NEW_TOKENS}")
print(f"[setup] OV-frontend supports {len(npu_backend.supported_op_names())} aten ops")
print()


# ── Dispatch tracer ──────────────────────────────────────────────────────────
#
# We sit *above* the PrivateUse1 boxed fallback. For every aten op we record:
#   - whether the op was hit while operating on NPU tensors
#   - whether the op was in the frontend's supported set (OV path) or not
#     (torch/cpu fallback)
#
# After the run we cross-reference with `npu_backend.per_op_ov_ops()` to print
# the actual OV subgraph created by each op's translator.
class OpTracer(TorchDispatchMode):
    def __init__(self, log_path=None):
        self.ov_calls       = Counter()   # aten op name -> #calls routed via OV
        self.fallback_calls = Counter()   # aten op name -> #calls that fell back
        self.first_seen     = {}          # aten op name -> first call shapes
        self._supported     = set(npu_backend.supported_op_names())
        # Persist the *current* op being dispatched to disk so that even if the
        # process segfaults inside an op kernel we know which one was last.
        self._log = open(log_path, "w") if log_path else None

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        def _is_npu(t):
            return isinstance(t, torch.Tensor) and t.device.type == "npu"

        flat_args = list(args) + list(kwargs.values())
        on_npu = any(_is_npu(a) for a in flat_args)

        full_name = str(func)              # "aten.add.Tensor"
        head = full_name.split(".")[0:2]
        aten_name = "::".join(head) if len(head) == 2 else full_name

        if on_npu:
            route = "OV" if aten_name in self._supported else "FALLBACK"
            if self._log:
                shapes = [tuple(a.shape) for a in flat_args if _is_npu(a)]
                self._log.write(f"{route}\t{aten_name}\t{shapes}\n")
                self._log.flush()
            if aten_name in self._supported:
                self.ov_calls[aten_name] += 1
            else:
                self.fallback_calls[aten_name] += 1
            self.first_seen.setdefault(
                aten_name,
                tuple(tuple(a.shape) if _is_npu(a) else type(a).__name__
                      for a in flat_args)
            )

        return func(*args, **kwargs)

    def close(self):
        if self._log:
            self._log.close()


# ── Load model ───────────────────────────────────────────────────────────────
print("[load ] importing transformers...")
from transformers import AutoTokenizer, AutoModelForCausalLM

print(f"[load ] loading tokenizer: {MODEL_ID}")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id

print(f"[load ] loading model: {MODEL_ID}")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE)
model.eval()

print(f"[load ] moving model to {NPU}")
model = model.to(NPU)

# ── Prepare input ────────────────────────────────────────────────────────────
messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user",   "content": "Who are you?"},
]
# Most chat templates produce a single string; tokenize directly.
try:
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
except Exception:
    prompt = "Q: Who are you?\nA:"

inputs = tok(prompt, return_tensors="pt").to(NPU)
print(f"[input] prompt tokens : {inputs['input_ids'].shape}")
print()


# ── Generate under the tracer ────────────────────────────────────────────────
intel_npu.ov_stats.reset()
TRACE_LOG = os.environ.get("TRACE_LOG", "/tmp/llm_npu_trace.log")
print(f"[trace] streaming op trace to {TRACE_LOG}")
tracer = OpTracer(log_path=TRACE_LOG)

print("[run  ] generating...")
t0 = time.perf_counter()
try:
    with tracer, torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
    elapsed = time.perf_counter() - t0
    text = tok.decode(out[0], skip_special_tokens=True)
    print(f"[run  ] done in {elapsed:.2f}s")
    print()
    print("─── generated text ───")
    print(text)
    print("──────────────────────")
except Exception as e:
    elapsed = time.perf_counter() - t0
    print(f"[run  ] FAILED after {elapsed:.2f}s: {type(e).__name__}: {e}")
    print("[run  ] continuing to print op trace collected before the failure")

print()


# ── Report ───────────────────────────────────────────────────────────────────
tracer.close()
print("=" * 78)
print("  Aten ops dispatched on NPU")
print("=" * 78)
print(f"  total OV-routed calls     : {sum(tracer.ov_calls.values())}")
print(f"  total fallback calls      : {sum(tracer.fallback_calls.values())}")
print(f"  unique OV-routed aten ops : {len(tracer.ov_calls)}")
print(f"  unique fallback aten ops  : {len(tracer.fallback_calls)}")
print(f"  OV stats                  : {intel_npu.ov_stats}")
print()

print("─── OV-routed aten ops (calls)  →  OV subgraph from PyTorch frontend ───")
per_op_ov = npu_backend.per_op_ov_ops()   # only filled for ops that actually compiled
for aten_name, n in sorted(tracer.ov_calls.items(), key=lambda kv: -kv[1]):
    ov_ops = per_op_ov.get(aten_name)
    if ov_ops is None:
        # Translator was supported but no compile happened (e.g. all-cache-hits
        # for shapes that another decode step already populated). Show "—".
        ov_str = "(no fresh compile in this run — already cached or skipped)"
    else:
        # Drop wrappers that aren't interesting to the user.
        useful = [o for o in ov_ops if o not in ("Parameter", "Result")]
        ov_str = " → ".join(useful) if useful else "(identity)"
    print(f"  {aten_name:32s}  x{n:<5d}  {ov_str}")
print()

if tracer.fallback_calls:
    print("─── Aten ops that FELL BACK to torch xpu/cpu (unsupported by frontend) ───")
    for aten_name, n in sorted(tracer.fallback_calls.items(), key=lambda kv: -kv[1]):
        is_supported = aten_name in set(npu_backend.supported_op_names())
        reason = "not in frontend op_table" if not is_supported else "rejected by dispatch (e.g. complex args)"
        print(f"  {aten_name:32s}  x{n:<5d}  ({reason})")
    print()
else:
    print("─── No fallbacks: every NPU op was routed through OpenVINO. ───")
    print()

print("Done.")
