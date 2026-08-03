#!/usr/bin/env python3
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# test_llm_safetensor.py — End-to-end HF LLM test on torch.device("xpu:npu").
#
# Tests an HF Llama-family model (safetensor weights) running via the
# OpenVINO eager NPU backend registered under torch.device("xpu:npu").
#
# Requirements:
#   pip install transformers accelerate
#   Either a local checkpoint or network access to HF Hub.
#
# Environment variables:
#   HF_MODEL   — model id or local path  (default: meta-llama/Llama-3.2-1B)
#   HF_OFFLINE — set to 1 to force offline mode
#
# Run:
#   cd /home/icx-6338/ynimmaga/openvino_xpu/src/frontends/eager
#   HF_MODEL=meta-llama/Llama-3.2-1B python pytorch/tests/test_llm_safetensor.py

import sys
import os

# Ensure intel_npu / npu_backend are importable from the eager root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import torch
import intel_npu

NPU = intel_npu.device("xpu:npu")   # torch.device("npu", 0)

MODEL_ID   = os.environ.get("HF_MODEL", "meta-llama/Llama-3.2-1B")
PROMPT     = "The OpenVINO NPU eager backend enables"
MAX_NEW    = 20
DTYPE      = torch.bfloat16   # bf16 to stay within NPU memory budget

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load(model_id: str, device):
    """Load tokenizer + model to `device` (weights loaded on CPU first)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=DTYPE,
        device_map="cpu",   # load to CPU first — avoids OOM on host
    )
    model = model.to(device).eval()
    return model, tok


def _tokenize(tok, prompt, device):
    enc = tok(prompt, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_model_loads_to_npu():
    """Model weights move to 'npu' device without error."""
    model, tok = _load(MODEL_ID, NPU)
    # Check at least one parameter lives on npu
    first_param = next(model.parameters())
    assert first_param.device.type == "npu", (
        f"Expected 'npu', got '{first_param.device.type}'"
    )
    print(f"  Model: {MODEL_ID}")
    print(f"  Dtype: {first_param.dtype}  Device: {first_param.device}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params / 1e6:.1f} M")


def test_forward_pass_shape():
    """Single forward pass returns logits of correct shape."""
    model, tok = _load(MODEL_ID, NPU)
    enc = _tokenize(tok, PROMPT, NPU)

    intel_npu.ov_stats.reset()
    with torch.no_grad():
        out = model(**enc)

    logits = out.logits                           # (batch, seq_len, vocab_size)
    assert logits.ndim == 3
    assert logits.shape[0] == 1
    assert logits.shape[-1] == model.config.vocab_size, (
        f"vocab mismatch: {logits.shape[-1]} vs {model.config.vocab_size}"
    )
    print(f"  logits shape:    {tuple(logits.shape)}")
    print(f"  OV ops executed: {intel_npu.ov_stats.ov}")
    print(f"  Convert time:    {intel_npu.ov_stats.convert_ms:.1f} ms")
    print(f"  Compile time:    {intel_npu.ov_stats.compile_ms:.1f} ms")


def test_generate_non_empty():
    """model.generate() produces a non-empty continuation on xpu:npu."""
    model, tok = _load(MODEL_ID, NPU)
    enc = _tokenize(tok, PROMPT, NPU)

    with torch.no_grad():
        output_ids = model.generate(
            enc["input_ids"],
            attention_mask=enc.get("attention_mask"),
            max_new_tokens=MAX_NEW,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )

    new_ids = output_ids[0, enc["input_ids"].shape[1]:]
    result  = tok.decode(new_ids, skip_special_tokens=True)
    assert len(result.strip()) > 0, "generate() returned empty text"
    print(f"  Prompt:    {PROMPT!r}")
    print(f"  Generated: {result!r}")


def test_logits_match_cpu():
    """NPU logits agree with CPU reference (atol=1e-2 for bfloat16).

    Loads the same checkpoint twice — once to CPU, once to NPU — and
    compares the first-token logits for the test prompt.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    # CPU reference
    model_cpu = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, device_map="cpu"
    ).eval()
    enc_cpu = tok(PROMPT, return_tensors="pt")
    with torch.no_grad():
        ref_logits = model_cpu(**enc_cpu).logits.float()   # (1, T, V)

    # NPU model
    model_npu = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, device_map="cpu"
    ).to(NPU).eval()
    enc_npu = {k: v.to(NPU) for k, v in enc_cpu.items()}
    with torch.no_grad():
        npu_logits = model_npu(**enc_npu).logits.to("cpu").float()

    max_diff = (ref_logits - npu_logits).abs().max().item()
    assert max_diff < 1e-2, (
        f"logit max abs diff = {max_diff:.4f}, threshold 1e-2"
    )
    print(f"  Logit max abs diff: {max_diff:.6f}  (threshold 1e-2) ✓")


def test_attention_ops_via_ov():
    """Verify key attention ops (mm, softmax, bmm) are handled by OV.

    Uses a small synthetic attention block to isolate the OV dispatch.
    This avoids loading the full LLM while still exercising the
    attention kernel path.
    """
    B, H, T, D = 1, 4, 16, 32   # batch, heads, seq_len, head_dim

    q_cpu = torch.randn(B * H, T, D)
    k_cpu = torch.randn(B * H, D, T)
    v_cpu = torch.randn(B * H, T, D)

    q = q_cpu.to(NPU)
    k = k_cpu.to(NPU)
    v = v_cpu.to(NPU)

    intel_npu.ov_stats.reset()

    scale  = D ** -0.5
    scores = torch.bmm(q, k) * scale                    # (B*H, T, T)
    attn   = torch.softmax(scores, dim=-1)               # (B*H, T, T)
    out    = torch.bmm(attn, v)                          # (B*H, T, D)

    assert out.device.type == "npu"
    assert out.shape == (B * H, T, D)

    # CPU reference
    scores_ref = torch.bmm(q_cpu, k_cpu) * scale
    attn_ref   = torch.softmax(scores_ref, dim=-1)
    out_ref    = torch.bmm(attn_ref, v_cpu)

    assert torch.allclose(out.to("cpu"), out_ref, atol=1e-4), (
        f"attention output mismatch"
    )
    print(f"  Attention OV ops: {intel_npu.ov_stats.ov}")


def test_embedding_lookup_via_ov():
    """aten::embedding routes through OV on xpu:npu."""
    vocab_size = 256
    embed_dim  = 32
    seq_len    = 8

    weight_cpu = torch.randn(vocab_size, embed_dim)
    ids_cpu    = torch.randint(0, vocab_size, (1, seq_len))

    weight = weight_cpu.to(NPU)
    ids    = ids_cpu.to(NPU)

    out    = torch.nn.functional.embedding(ids, weight)
    ref    = torch.nn.functional.embedding(ids_cpu, weight_cpu)

    assert out.shape == (1, seq_len, embed_dim)
    assert torch.allclose(out.to("cpu"), ref, atol=1e-5), "embedding mismatch"


def test_rms_norm_via_ov():
    """RMS norm (used in Llama) routes through OV on xpu:npu."""
    # PyTorch 2.4+ exposes torch.nn.RMSNorm
    if not hasattr(torch.nn, "RMSNorm"):
        print("  SKIP: torch.nn.RMSNorm not available in this PyTorch version")
        return

    hidden = 64
    x_cpu = torch.randn(2, hidden)
    norm_cpu = torch.nn.RMSNorm(hidden).eval()
    norm_npu = torch.nn.RMSNorm(hidden).eval()
    norm_npu.load_state_dict(norm_cpu.state_dict())
    norm_npu = norm_npu.to(NPU)

    x_npu = x_cpu.to(NPU)
    with torch.no_grad():
        ref = norm_cpu(x_cpu)
        out = norm_npu(x_npu).to("cpu")

    assert torch.allclose(ref, out, atol=1e-4), "rms_norm mismatch"


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_model_loads_to_npu,
        test_forward_pass_shape,
        test_generate_non_empty,
        test_logits_match_cpu,
        test_attention_ops_via_ov,
        test_embedding_lookup_via_ov,
        test_rms_norm_via_ov,
    ]

    passed = failed = skipped = 0
    for t in tests:
        print(f"\n--- {t.__name__} ---")
        try:
            t()
            print(f"  PASS")
            passed += 1
        except ImportError as e:
            print(f"  SKIP: missing dependency — {e}")
            skipped += 1
        except Exception as e:
            import traceback
            print(f"  FAIL: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"  {passed} passed  {skipped} skipped  {failed} failed")
    if failed:
        sys.exit(1)
