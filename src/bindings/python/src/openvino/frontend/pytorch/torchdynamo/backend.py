# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
# mypy: ignore-errors

import logging
import os
from functools import partial
from hashlib import sha256

import torch
from torch._dynamo.backends.common import fake_tensor_unsupported, aot_autograd
from torch._dynamo.backends.registry import register_backend
from torch._decomp import get_decompositions
from torch._inductor.compile_fx import compile_fx
from torch.fx.experimental.proxy_tensor import make_fx
from functorch.compile import min_cut_rematerialization_partition, nop
from torch.fx.experimental.proxy_tensor import DecompositionInterpreter
from torch._decomp import decomposition_table

from openvino.frontend import FrontEndManager
from openvino.runtime import Core, Type, PartialShape
from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
from openvino.frontend.pytorch.torchdynamo.partition import Partitioner
from openvino.frontend.pytorch.torchdynamo.execute import execute
from openvino.frontend.pytorch.torchdynamo.op_support import OperatorSupport, aten2aten_decomp

log = logging.getLogger(__name__)

"""
    This is a preview feature in OpenVINO. Torchscript backend
    enables users to compile PyTorch models using torch.compile
    with OpenVINO as a target backend in PyTorch applications

    Sample usage:
    This sample code loads resnet50 torchvision model and compiles it using torch dynamo.
    We can then use this model for inference. We only need to add two lines of code to
    the Pytorch applications which are marked in the code below

    1) import openvino.frontend.pytorch.torchdynamo.backend
    model = torchvision.models.resnet50()
    2) model = torch.compile(model, backend="openvino")
"""


@register_backend
@fake_tensor_unsupported
def openvino(subgraph, example_inputs):
    if (os.getenv("PYTORCH_TRACING_MODE") is not None):
        if (os.getenv("PYTORCH_TRACING_MODE") == "TORCHFX"):
            return fx_openvino(subgraph, example_inputs)
    return ts_openvino(subgraph, example_inputs)

@register_backend
@fake_tensor_unsupported
def openvino_bw(subgraph, example_inputs):
    if (os.getenv("PYTORCH_TRACING_MODE") is not None):
        if (os.getenv("PYTORCH_TRACING_MODE") == "TORCHFX"):
            return fx_openvino_bw(subgraph, example_inputs)
    return ts_openvino(subgraph, example_inputs)


aot_eager = aot_autograd(fw_compiler=nop)
#register_backend(name="aot_eager", compiler_fn=aot_eager)

aot_ovgraphs = aot_autograd(fw_compiler=openvino, bw_compiler=openvino_bw)
register_backend(name="aot_openvino", compiler_fn=aot_ovgraphs)


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
        if (os.getenv("OPENVINO_TS_BACKEND_DEVICE") is not None):
            device = os.getenv("OPENVINO_TS_BACKEND_DEVICE")
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

bw_subgraph_num = 0
def fx_openvino_bw(subgraph, example_inputs):
    try:
        global bw_subgraph_num
        bw_subgraph_num = bw_subgraph_num + 1
        if (bw_subgraph_num == 1): 
            return subgraph

        executor_parameters = None
        if os.getenv("OPENVINO_TORCH_MODEL_CACHING") is not None:
            model_hash_str = sha256(subgraph.code.encode('utf-8')).hexdigest()
            executor_parameters = {"model_hash_str": model_hash_str}
        model = make_fx(subgraph,
                        decomposition_table=get_decompositions([torch.ops.aten.convolution_backward.default,
                                                                torch.ops.aten.gelu_backward.default,
                                                                torch.ops.aten.native_group_norm_backward.default,
                                                                torch.ops.aten.native_layer_norm_backward.default,
                                                                torch.ops.aten._softmax_backward_data.default,
                                                                torch.ops.aten.slice_backward.default
                                                                ]))(*example_inputs)
        partitioner = Partitioner()
        compiled_model = partitioner.make_partitions(model)
 
        def _call(*args):
            res = execute(compiled_model, *args, executor="openvino",
                          executor_parameters=executor_parameters)
            return res
        return _call
    except Exception as e:
        log.debug(f"Failed in OpenVINO execution: {e}")
        return compile_fx(subgraph, example_inputs)

fw_subgraph_num = 0
def fx_openvino(subgraph, example_inputs):
    try:
        global fw_subgraph_num
        fw_subgraph_num = fw_subgraph_num + 1
        executor_parameters = None
        if os.getenv("OPENVINO_TORCH_MODEL_CACHING") is not None:
            model_hash_str = sha256(subgraph.code.encode('utf-8')).hexdigest()
            executor_parameters = {"model_hash_str": model_hash_str}
        model = make_fx(subgraph,
                        decomposition_table=get_decompositions([torch.ops.aten._softmax.default,
                                                                torch.ops.aten.native_group_norm.default,
                                                                torch.ops.aten.native_layer_norm.default]))(*example_inputs)

        with torch.no_grad():
            model.eval()
        partitioner = Partitioner()
        compiled_model = partitioner.make_partitions(model)

        def _call(*args):
            res = execute(compiled_model, *args, executor="openvino",
                          executor_parameters=executor_parameters)
            return res
        return _call
    except Exception as e:
        log.debug(f"Failed in OpenVINO execution: {e}")
        return compile_fx(subgraph, example_inputs)


def reset():
    clear_caches()
