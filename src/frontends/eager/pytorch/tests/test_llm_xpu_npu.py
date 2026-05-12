#!/usr/bin/env python3
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# test_llm_xpu_npu.py — HF LLM snippet using torch.device("xpu:npu")
#
# Demonstrates the user-facing API: a PyTorch application replaces
#   torch.device("cpu")  or  torch.device("cuda")
# with
#   torch.device("xpu:npu")
# and gets NPU acceleration through OpenVINO — transparently.
#
# Run:
#   cd /home/icx-6338/ynimmaga/openvino_xpu/src/frontends/eager
#   HF_MODEL=meta-llama/Llama-3.2-1B python pytorch/tests/test_llm_xpu_npu.py

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import torch
import intel_npu   # registers "npu" as PrivateUse1 / xpu:npu alias

# ── Device selection ──────────────────────────────────────────────────────
# This is the ONLY change a user makes to their existing PyTorch code.
# Swap "cpu" / "cuda" / "xpu" with intel_npu.device("xpu:npu").
DEVICE = intel_npu.device("xpu:npu")   # → torch.device("npu", 0)

MODEL_ID  = os.environ.get("HF_MODEL", "meta-llama/Llama-3.2-1B")
PROMPT    = "OpenVINO NPU eager mode enables"
MAX_NEW   = 30
DTYPE     = torch.bfloat16


# ─────────────────────────────────────────────────────────────────────────────
# Snippet 1: Minimal usage — works exactly like CUDA/XPU snippets
# ─────────────────────────────────────────────────────────────────────────────
def snippet_minimal():
    """
    The canonical user snippet.  Drop-in replacement for:
        device = torch.device("cuda")   # before
        device = intel_npu.device("xpu:npu")  # after
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n[snippet_minimal] Loading {MODEL_ID} on {DEVICE} ...")
    tok   = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, device_map="cpu"
    ).to(DEVICE).eval()

    inputs = tok(PROMPT, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    elapsed = time.perf_counter() - t0

    new_ids = out[0, inputs["input_ids"].shape[1]:]
    text    = tok.decode(new_ids, skip_special_tokens=True)
    ntok    = new_ids.shape[0]

    print(f"  Device  : {DEVICE}")
    print(f"  Prompt  : {PROMPT!r}")
    print(f"  Output  : {text!r}")
    print(f"  Tokens  : {ntok}  in {elapsed:.2f}s  ({ntok/elapsed:.1f} tok/s)")
    print(f"  OV ops  : {intel_npu.ov_stats.ov}")
    assert len(text.strip()) > 0, "empty generation"


# ─────────────────────────────────────────────────────────────────────────────
# Snippet 2: Embedding + attention forward (transformer building blocks)
# ─────────────────────────────────────────────────────────────────────────────
def snippet_transformer_ops():
    """
    Show the key transformer ops running on xpu:npu without a full model.
    Exercises: embedding, linear, softmax, bmm, layer_norm.
    """
    print(f"\n[snippet_transformer_ops] Running on {DEVICE}")
    intel_npu.ov_stats.reset()

    B, H, T, D = 1, 4, 16, 64   # batch, heads, seq_len, head_dim
    V = 512                       # vocab size

    # Embedding lookup
    emb_weight = torch.randn(V, H * D, device=DEVICE, dtype=DTYPE)
    token_ids  = torch.randint(0, V, (B, T), device=DEVICE)
    hidden     = torch.nn.functional.embedding(token_ids, emb_weight)  # (B, T, H*D)

    # Self-attention (scaled dot-product via bmm + softmax)
    q = hidden.view(B * H, T, D)
    k = hidden.view(B * H, T, D).transpose(1, 2)   # (B*H, D, T)
    v = hidden.view(B * H, T, D)
    scores = torch.bmm(q, k) * (D ** -0.5)
    attn   = torch.softmax(scores, dim=-1)
    ctx    = torch.bmm(attn, v)                     # (B*H, T, D)

    # Feed-forward projection
    proj   = torch.nn.functional.linear(
        ctx.view(B, T, H * D),
        torch.randn(H * D, H * D, device=DEVICE, dtype=DTYPE),
    )
    out    = torch.nn.functional.layer_norm(proj, [H * D])

    assert out.device.type == "npu", f"expected npu, got {out.device}"
    assert out.shape == (B, T, H * D)
    print(f"  Output shape : {tuple(out.shape)}")
    print(f"  OV ops fired : {intel_npu.ov_stats.ov}")


# ─────────────────────────────────────────────────────────────────────────────
# Snippet 3: Device-agnostic code pattern
# ─────────────────────────────────────────────────────────────────────────────
def snippet_device_agnostic():
    """
    Show how existing device-agnostic torch code works unchanged when the
    device is set to xpu:npu.
    """
    print(f"\n[snippet_device_agnostic] device = {DEVICE}")

    class SimpleModel(torch.nn.Module):
        def __init__(self, d_in, d_hidden, d_out):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(d_in, d_hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(d_hidden, d_out),
            )
        def forward(self, x):
            return self.net(x)

    model = SimpleModel(64, 128, 10).to(DEVICE).eval()
    x     = torch.randn(8, 64, device=DEVICE)

    intel_npu.ov_stats.reset()
    with torch.no_grad():
        out = model(x)

    assert out.device.type == "npu"
    assert out.shape == (8, 10)
    print(f"  Output shape : {tuple(out.shape)}")
    print(f"  OV ops fired : {intel_npu.ov_stats.ov}")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import traceback
    snippets = [snippet_transformer_ops, snippet_device_agnostic]

    # Full LLM snippet only if transformers is installed
    try:
        import transformers  # noqa: F401
        snippets.insert(0, snippet_minimal)
    except ImportError:
        print("[SKIP] snippet_minimal — transformers not installed")

    passed = failed = 0
    for fn in snippets:
        try:
            fn()
            print(f"  -> PASS")
            passed += 1
        except Exception as e:
            print(f"  -> FAIL: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
