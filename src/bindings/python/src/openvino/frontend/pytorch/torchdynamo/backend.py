# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
# mypy: ignore-errors

import logging
import os
from functools import partial
from hashlib import sha256

import torch
from torch._decomp import decomposition_table, get_decompositions, register_decomposition
from torch._decomp.decompositions import aten, pw_cast_for_opmath
from torch._dynamo.backends.common import fake_tensor_unsupported, aot_autograd
from torch._dynamo.backends.registry import register_backend
from torch._inductor.compile_fx import compile_fx
from torch.fx.experimental.proxy_tensor import make_fx

from openvino.frontend import FrontEndManager
from openvino.runtime import Core, Type, PartialShape
from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
from openvino.frontend.pytorch.torchdynamo.partition import Partitioner
from openvino.frontend.pytorch.torchdynamo.execute import execute, execute_cached
from openvino.frontend.pytorch.torchdynamo.compile import cached_model_name, openvino_compile_cached_model
from openvino.frontend.pytorch.torchdynamo.backend_utils import _get_cache_dir, _get_device, _get_model_caching

from openvino.runtime import Core, Type, PartialShape

log = logging.getLogger(__name__)

"""
    This is a preview feature in OpenVINO. This feature
    enables users to compile PyTorch models using torch.compile
    with OpenVINO as a target backend in PyTorch applications

    Sample usage:
    This sample code loads resnet50 torchvision model and compiles it using torch dynamo.
    We can then use this model for inference. We only need to add two lines of code to
    the Pytorch applications which are marked in the code below

    1) import openvino.torch
    model = torchvision.models.resnet50()
    2) model = torch.compile(model, backend="openvino")
"""


@register_backend
@fake_tensor_unsupported
def openvino(subgraph, example_inputs, options=None):
    return fx_openvino(subgraph, example_inputs, options)

@register_backend
@fake_tensor_unsupported
def openvino_ts(subgraph, example_inputs):
    return ts_openvino(subgraph, example_inputs)

def ts_openvino(subgraph, example_inputs):
    try:
        model = torch.jit.script(subgraph)
        model.eval()
        fr_model = torch.jit.freeze(model)

        core = Core()
        fe_manager = FrontEndManager()
        fe = fe_manager.load_by_framework('pytorch')
        dtype_mapping = {
            torch.float64: Type.f64,
            torch.float32: Type.f32,
            torch.float16: Type.f16,
            torch.int64: Type.i64,
            torch.int32: Type.i32,
            torch.uint8: Type.u8,
            torch.int8: Type.i8,
            torch.bool: Type.boolean,
        }
        decoder = TorchScriptPythonDecoder(fr_model)

        # TODO: Use convert_model instead when mo --convert_model api becomes a part of OV runtime
        im = fe.load(decoder)
        om = fe.convert(im)

        for idx, input_data in enumerate(example_inputs):
            om.inputs[idx].get_node().set_element_type(dtype_mapping[input_data.dtype])
            om.inputs[idx].get_node().set_partial_shape(PartialShape(list(input_data.shape)))
        om.validate_nodes_and_infer_types()

        device = "CPU"
        if (os.getenv("OPENVINO_TORCH_BACKEND_DEVICE") is not None):
            device = os.getenv("OPENVINO_TORCH_BACKEND_DEVICE")
            assert device in core.available_devices, "Specified device " + device + " is not in the list of OpenVINO Available Devices"

        compiled_model = core.compile_model(om, device)

        def _call(*args):
            if not hasattr(_call, "execute_on_ov"):
                _call.execute_on_ov = True
            execute_on_ov = getattr(_call, "execute_on_ov")
            if execute_on_ov:
                ov_inputs = [a.detach().cpu().numpy() for a in args]
                try:
                    res = compiled_model(ov_inputs)
                except Exception as e:
                    log.debug(f"Failed in OpenVINO execution: {e}")
                    _call.execute_on_ov = False
                    return subgraph.forward(*args)
                result = [torch.from_numpy(res[out]) for out in compiled_model.outputs]
                return result
            else:
                return subgraph.forward(*args)
        return _call
    except Exception as e:
        log.debug(f"Failed in compilation: {e}")
        return compile_fx(subgraph, example_inputs)

@register_decomposition(aten.convolution_backward)
@pw_cast_for_opmath
def convolution_backward(
    grad_output,
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    output_mask,
):

    if (stride == [2,2]):
        output_padding = [1,1]

    # Compute the gradient of the input tensor
    grad_input = torch.nn.functional.conv_transpose2d(
        grad_output, weight, stride=stride, padding=padding, dilation=dilation, groups=groups, output_padding=output_padding
    )

    # Compute the gradient of the weight tensor
    grad_weight = torch.nn.functional.conv_transpose2d(
        #input, weight.transpose(0,1), stride=stride, padding=padding, dilation=dilation, groups=groups
        input, weight.transpose(0,1), stride=stride, padding=padding, dilation=dilation, groups=groups, output_padding=output_padding
    )

    # Compute the gradient of the bias tensor
    if bias is not None:
        grad_bias = grad_output.sum([0, 2, 3], keepdim=True)
    else:
        grad_bias = None

    return grad_input, grad_weight, grad_bias


@register_decomposition(aten._scaled_dot_product_flash_attention.default)
def scaled_dot_product_flash_attention(
    query,
    key,
    value,
    dropout_p = 0.0,
    is_causal = False,
    *,
    return_debug_mask = False,
    scale = None,
):
    dtype = query.dtype
    batchSize, num_head, qSize, headSize = (
        query.shape[0],
        query.shape[1],
        query.shape[2],
        query.shape[3],
    )

    logsumexp = torch.empty([batchSize, qSize, num_head, headSize], dtype=torch.float)
    cum_seq_q, cum_seq_k = torch.empty([], dtype=torch.long), torch.empty(
        [], dtype=torch.long
    )
    max_q, max_k = 0, 0
    philox_seed, philox_offset = torch.empty([], dtype=torch.long), torch.empty(
        [], dtype=torch.long
    )
    debug_attn_mask = torch.empty(
        [],
        dtype=query.dtype,
        device=query.device,
        requires_grad=query.requires_grad,
    )
    output, _ = aten._scaled_dot_product_attention_math.default(
        query, key, value, None, dropout_p, is_causal, None, scale=scale
    )

    scores = torch.matmul(query, key.transpose(-2,-1))/ (key.size(-1) ** 0.5)
    logsumexp = torch.logsumexp(scores, dim=-1)

    output = output.transpose(1, 2).contiguous(memory_format=torch.contiguous_format)
    return (
        output.transpose(1, 2),
        logsumexp,
        cum_seq_q,
        cum_seq_k,
        max_q,
        max_k,
        philox_seed,
        philox_offset,
        debug_attn_mask,
    )


aot_ovgraphs = aot_autograd(fw_compiler=openvino, bw_compiler=openvino)
register_backend(name="aot_openvino", compiler_fn=aot_ovgraphs)

def fx_openvino(subgraph, example_inputs, options):
    try:
        executor_parameters = None
        inputs_reversed = False
        openvino_model_caching = _get_model_caching(options)
        if openvino_model_caching is not None and openvino_model_caching:
            # Create a hash to be used for caching
            model_hash_str = sha256(subgraph.code.encode('utf-8')).hexdigest()
            executor_parameters = {"model_hash_str": model_hash_str}
            # Check if the model was fully supported and already cached
            example_inputs.reverse()
            inputs_reversed = True
            maybe_fs_cached_name = cached_model_name(model_hash_str + "_fs", _get_device(options), example_inputs, _get_cache_dir(options))
            if os.path.isfile(maybe_fs_cached_name + ".xml") and os.path.isfile(maybe_fs_cached_name + ".bin"):
                # Model is fully supported and already cached. Run the cached OV model directly.
                compiled_model = openvino_compile_cached_model(maybe_fs_cached_name, options, *example_inputs)
                def _call(*args):
                    res = execute_cached(compiled_model, *args)
                    return res
                return _call
        if inputs_reversed:
            example_inputs.reverse()
        model = make_fx(subgraph,
                        decomposition_table=get_decompositions([torch.ops.aten.convolution_backward.default,
                                                                torch.ops.aten.gelu_backward.default,
                                                                torch.ops.aten.native_group_norm_backward.default,
                                                                torch.ops.aten.native_layer_norm_backward.default,
                                                                torch.ops.aten._softmax_backward_data.default,
                                                                torch.ops.aten._scaled_dot_product_flash_attention.default,
                                                                torch.ops.aten.slice_backward.default,
                                                                torch.ops.aten._softmax.default,
                                                                torch.ops.aten.native_group_norm.default,
                                                                torch.ops.aten.native_layer_norm.default
                                                               ]))(*example_inputs)
        with torch.no_grad():
            model.eval()
        partitioner = Partitioner()
        compiled_model = partitioner.make_partitions(model)

        if executor_parameters is not None and 'model_hash_str' in executor_parameters:
            # Check if the model is fully supported.
            fully_supported = partitioner.check_fully_supported(compiled_model)
            if fully_supported:
                executor_parameters["model_hash_str"] += "_fs"

        def _call(*args):
            res = execute(compiled_model, *args, executor="openvino",
                          executor_parameters=executor_parameters, options=options)
            return res
        return _call
    except Exception as e:
        log.debug(f"Failed in OpenVINO execution: {e}")
        return compile_fx(subgraph, example_inputs)

def reset():
    clear_caches()
