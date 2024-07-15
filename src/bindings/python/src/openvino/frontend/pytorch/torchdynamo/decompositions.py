# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
# mypy: ignore-errors

import torch
from torch._decomp.decompositions import aten, pw_cast_for_opmath
from torch._decomp import register_decomposition, get_decompositions

@register_decomposition(aten.index_put)
@pw_cast_for_opmath
def index_put_decomposition(input_tensor, indices, values, accumulate=False):
    # Handle None in indices by replacing with appropriate ranges
    def handle_none_index(indices):
        expanded_indices = []
        for i in range(0, len(input_tensor.size())):
            if i < len(indices):
                if indices[i] is None:
                    expanded_indices.append(torch.arange(input_tensor.size(i)))
                else:
                    expanded_indices.append(indices[i].long())
            else:
                expanded_indices.append(torch.arange(input_tensor.size(i)))
        return expanded_indices

    expanded_indices = handle_none_index(indices)

    # Generate a meshgrid of indices to cover the multi-dimensional case
    grid_shape = [len(tensor) for tensor in expanded_indices]

    # Create the meshgrid
    meshgrid_indices = []
    for i, tensor in enumerate(expanded_indices):
        shape = [1] * len(expanded_indices)
        shape[i] = -1
        grid = tensor.view(*shape).expand(*grid_shape)
        meshgrid_indices.append(grid)

    # Flatten indices and values
    flat_indices = torch.stack([torch.flatten(grid) for grid in meshgrid_indices], dim=-1)
    flat_values = torch.flatten(values)

    prods = []
    prod_item = 1
    for i in range(0, len(input_tensor.size())):
        prod_item = prod_item * input_tensor.size(i)
        prods.append(prod_item)

    prod_tensor = torch.as_tensor(prods)
    strides = torch.div(prod_tensor[-1], prod_tensor).long()
    linear_indices = torch.matmul(flat_indices, strides)

    # Flatten the input tensor for the scatter operation
    input_flat = torch.flatten(input_tensor)
    flat_values = torch.ops.aten.expand(flat_values, linear_indices.size()) #flat_values.expand_as(linear_indices)

    if accumulate:
        x = torch.scatter_add(input_flat, 0, linear_indices, flat_values)
    else:
        x = torch.scatter(input_flat, 0, linear_indices, flat_values)

    # Reshape back to the original tensor shape
    output_tensor =  torch.ops.aten.view(x, input_tensor.size()) #input_flat.view(input_tensor.size())
    return output_tensor


@register_decomposition(aten.convolution_backward)
@pw_cast_for_opmath
def convolution_backward(
    grad_output,
    inp,
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
    if stride == [2, 2]:
        output_padding = [1, 1]

    # Compute the gradient of the input tensor
    grad_input = torch.nn.functional.conv_transpose2d(
        grad_output, weight, stride=stride, padding=padding, dilation=dilation, groups=groups, output_padding=output_padding
    )

    # Compute the gradient of the weight tensor
    grad_weight = torch.nn.functional.conv_transpose2d(
        inp, weight.transpose(0, 1), stride=stride, padding=padding, dilation=dilation, groups=groups, output_padding=output_padding
    )

    # Compute the gradient of the bias tensor
    if bias is not None:
        grad_bias = grad_output.sum([0, 2, 3], keepdim=True)
    else:
        grad_bias = None

    return grad_input, grad_weight, grad_bias

if len(get_decompositions([aten._scaled_dot_product_flash_attention.default])) == 0:
    @register_decomposition(aten._scaled_dot_product_flash_attention.default)
    def scaled_dot_product_flash_attention(
        query,
        key,
        value,
        dropout_p=0.0,
        is_causal=False,
        *,
        return_debug_mask=False,
        scale=None,
    ):
        batch_size, num_head, q_size, head_size = (
            query.shape[0],
            query.shape[1],
            query.shape[2],
            query.shape[3],
        )

        logsumexp = torch.empty([batch_size, q_size, num_head, head_size], dtype=torch.float)
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

        scores = torch.matmul(query, key.transpose(-2, -1)) / (key.size(-1) ** 0.5)
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


def get_aot_decomposition_list():
    return ([torch.ops.aten._scaled_dot_product_flash_attention.default,
             torch.ops.aten._softmax.default,
             torch.ops.aten._softmax_backward_data.default,
             torch.ops.aten.convolution_backward.default,
             torch.ops.aten.gelu_backward.default,
             torch.ops.aten.native_group_norm.default,
             torch.ops.aten.native_group_norm_backward.default,
             torch.ops.aten.native_layer_norm.default,
             torch.ops.aten.native_layer_norm_backward.default,
             torch.ops.aten.slice_backward.default])

def get_inf_decomposition_list():
    return ([torch.ops.aten.nll_loss_forward.default,
             torch.ops.aten.index_put.default])
