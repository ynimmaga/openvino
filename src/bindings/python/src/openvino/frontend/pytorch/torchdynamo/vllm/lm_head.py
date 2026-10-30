# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# mypy: ignore-errors

"""Run lm_head through OV instead of torch.

vLLM calls compute_logits() outside the OV-compiled region, so lm_head's GEMM
normally runs on torch/oneDNN. Moving it to OV frees it from OMP_NUM_THREADS
tuning (oneDNN needs a per-model, per-machine thread count; OV doesn't), but
on CPU a second live InferRequest interleaved with the main graph's roughly
doubles the main graph's per-step time -- a plugin-level cost, not tunable
here. So on CPU it is off unless OV_LM_HEAD=1.

On GPU it is on unless OV_LM_HEAD=0: there the alternative is a CPU GEMM
between two GPU steps, and an NNCF per-row int8 lm_head (or tied embedding)
is compiled compressed, reading half the bytes of the dense f16 weight.
"""

import logging
import os

logger = logging.getLogger(__name__)

# Distinct row-counts to keep staging buffers for; past it we realloc rather
# than grow without bound.
_MAX_CACHED_SHAPES = 32


def _ov_type_for(torch_dtype):
    """Map a torch float dtype to the OV type, or None if we should not try."""
    import openvino as ov
    import torch

    return {
        torch.bfloat16: ov.Type.bf16,
        torch.float16: ov.Type.f16,
        torch.float32: ov.Type.f32,
    }.get(torch_dtype)


def _copy_into(ov_tensor, tensor):
    """Fill an OV tensor from a torch tensor, preserving bf16 bits.

    numpy has no bfloat16, so bf16 goes through a uint16 re-tag on both sides
    (see execute._torch_to_numpy) rather than wrapping an ndarray directly,
    which would declare the port f16 and get rejected.
    """
    import numpy as np
    import torch

    dst = np.asarray(ov_tensor.data)
    src = tensor.contiguous()
    if src.dtype == torch.bfloat16:
        dst.view(np.uint16)[:] = src.view(torch.uint16).numpy()
    else:
        dst[:] = src.numpy()


def _int8_weight(int8, et):
    """Dense-shaped [V, H] lm_head weight from NNCF per-row int8, as a graph.

    The u8 Constant -> Convert -> Subtract -> Multiply chain is the pattern the
    device plugins recognize as a compressed FC, so the weight stays int8 in
    device memory instead of being folded to `et`.
    """
    import openvino as ov
    import openvino.opset13 as op

    stored, scale, zero_point = int8
    weight = op.convert(op.constant(stored.contiguous().numpy()), et)
    if zero_point is not None:
        weight = op.subtract(
            weight, op.convert(op.constant(zero_point.contiguous().numpy()), et))
    scale = op.constant(scale.contiguous().numpy())
    if scale.get_element_type() != et:
        scale = op.convert(scale, et)
    return op.multiply(weight, scale)


def build_ov_lm_head(weight, nthreads=None, device="CPU", int8=None):
    """Compile ``[M, H] x [H, V]`` on OV and return a cpu_linear-shaped callable.

    `int8` is the checkpoint's `(stored u8 [V, H], scale [V, 1], zero_point
    [V, 1] or None)`; given, the weight is compiled compressed rather than from
    the dense `weight`, which then only supplies dtype and shape.

    Returns None if the dtype is not one we handle or anything fails, so the
    caller can fall back to the oneDNN dispatch.
    """
    import numpy as np
    import openvino as ov
    import openvino.opset13 as op
    import torch

    et = _ov_type_for(weight.dtype)
    if et is None:
        logger.debug("[OV plugin] lm_head dtype %s not handled by OV path", weight.dtype)
        return None
    on_cpu = device.startswith("CPU")
    # The GPU plugin has no bf16: compute in f16 and move bf16 in and out as
    # f32, which loses nothing on either side of the conversion.
    io_et = et if on_cpu or et != ov.Type.bf16 else ov.Type.f32
    compute_et = et if on_cpu else ov.Type.f16

    if nthreads is None:
        # Same pool the main graph gets; the two never run concurrently, so
        # there is nothing to share it with. 0 means "let OV decide".
        nthreads = int(os.environ.get("OV_INFERENCE_NUM_THREADS", "0") or 0)

    vocab_size, hidden_size = weight.shape
    if int8 is not None:
        w = _int8_weight(int8, io_et)
    else:
        wt = ov.Tensor(et, [vocab_size, hidden_size])
        _copy_into(wt, weight)
        w = op.constant(wt)
        if io_et != et:
            w = op.convert(w, io_et)

    # transpose_b: vLLM stores lm_head as [vocab, hidden]; avoids copying a
    # 131-467 MB weight to consume it the other way.
    param = op.parameter([-1, hidden_size], io_et, name="x")
    model = ov.Model([op.result(op.matmul(param, w, False, True))],
                     [param], "ov_lm_head")

    # Compute at the model's own dtype, not preset's f16->bf16 substitute --
    # bf16's narrower mantissa would flip greedy argmax on near-ties.
    cfg = {"INFERENCE_PRECISION_HINT": compute_et.get_type_name()}
    if nthreads and on_cpu:
        cfg["INFERENCE_NUM_THREADS"] = nthreads
    compiled = ov.Core().compile_model(model, device, cfg)

    req = compiled.create_infer_request()
    iport, oport = compiled.inputs[0], compiled.outputs[0]
    staging = {}

    def cpu_linear(inp, weight, bias):
        rows = inp.shape[0]
        buf = staging.get(rows)
        if buf is None:
            if len(staging) >= _MAX_CACHED_SHAPES:
                staging.clear()
            buf = staging[rows] = ov.Tensor(io_et, [rows, hidden_size])
        _copy_into(buf, inp if io_et == et else inp.float())
        req.set_tensor(iport, buf)
        req.infer()

        out = req.get_tensor(oport).data
        # .clone(): the next infer() overwrites this buffer in place, but
        # callers may hold logits across steps.
        out = torch.from_numpy(np.asarray(out)).clone()
        if io_et == ov.Type.bf16:
            out = out.view(torch.bfloat16)
        elif io_et != et:
            out = out.to(weight.dtype)
        return out + bias if bias is not None else out

    logger.info("[OV plugin] lm_head compiled on OV %s: [%d, %d] %s%s, %s threads",
                device, vocab_size, hidden_size, compute_et.get_type_name(),
                " int8-compressed" if int8 is not None else "",
                nthreads if nthreads and on_cpu else "auto")
    return cpu_linear
