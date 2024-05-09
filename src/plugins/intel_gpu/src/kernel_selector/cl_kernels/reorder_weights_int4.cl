// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_weights.cl"

KERNEL(reorder_weights_int4)(const __global INPUT0_TYPE* input, __global OUTPUT_TYPE* output) {
#if defined(INPUT0_LAYOUT_IOYX) && defined(OUTPUT_LAYOUT_OIYX)
    const uint out_byte_offset = get_global_id(0);

    const uint offset0 = out_byte_offset * 2 + 0;
    const uint offset1 = out_byte_offset * 2 + 1;

    const uint i0 = offset0 % OUTPUT_IFM_NUM;
    const uint i1 = offset1 % OUTPUT_IFM_NUM;

    const uint o0 = offset0 / OUTPUT_IFM_NUM;
    const uint o1 = offset1 / OUTPUT_IFM_NUM;

    const uint input0_offset = GET_FILTER_INDEX(INPUT0, 0, o0, i0, 0, 0);
    const uint input1_offset = GET_FILTER_INDEX(INPUT0, 0, o1, i1, 0, 0);

    const uint input0_idx = input0_offset % 2;
    const uint input1_idx = input1_offset % 2;

    INPUT0_TYPE in0 = (input[input0_offset / 2] >> input0_idx*4) & 0x0F;
    INPUT0_TYPE in1 = (input[input1_offset / 2] >> input1_idx*4) & 0x0F;

    OUTPUT_TYPE out = in0 | (in1 << 4);
    output[out_byte_offset] = out;
#elif defined(OUTPUT_LAYOUT_OS_IYX_OSV32)
    // os_iyx osv32 layout for int4 packed weight
    // k0_f0f16 | k0_f1f17 | .... | k0_f15f31 || k1_f0f16 | k1_f1f17 | ... | k1_f15f31
    // k2_f0f16 | k2_f1f17 | .... | k2_f15f31 || k3_f0f16 | k3_f1f17 | ... | k3_f15f31
    // ...
    const unsigned o = (uint)get_global_id(0);
    const unsigned i = (uint)get_global_id(1);

    const unsigned o0 = (o / 16) * 32 + (o % 16);
    const unsigned o1 = (o / 16) * 32 + (o % 16) + 16;

    const uint input0_offset = GET_FILTER_INDEX(INPUT0, 0, o0, i, 0, 0);
    const uint input1_offset = GET_FILTER_INDEX(INPUT0, 0, o1, i, 0, 0);

    const uint input0_idx = input0_offset % 2;
    const uint input1_idx = input1_offset % 2;

    INPUT0_TYPE in0 = (input[input0_offset / 2] >> input0_idx*4) & 0x0F;
    INPUT0_TYPE in1 = (input[input1_offset / 2] >> input1_idx*4) & 0x0F;

    INPUT0_TYPE packed_out_channels = in0 | (in1 << 4);

    const uint output_idx = GET_FILTER_OS_IYX_OSV_INDEX(OUTPUT, o, i, 0, 0, 32 / 2); // Calculate offset as osv16 due to packing
    output[output_idx] = packed_out_channels;

#elif defined(OUTPUT_LAYOUT_OS_IS_YX_OSV32_ISV2)
    // osv32_isv2 layout for int4 packed weight
    // f0_k0k1 | f1_k0k1 | ....  | f15_k0k1|| f16_k0k1 | f17_k0k1 | ... | f31_k0k1
    // f0_k2k3 | f1_k2k3 | ....  | f15_k2k3|| f16_k2k3 | f17_k2k3 | ... | f31_k2k3
    // ...
    const unsigned o = (uint)get_global_id(0);
    const unsigned i = (uint)get_global_id(1) * 2;

    const uint input0_offset = GET_FILTER_INDEX(INPUT0, 0, o, i, 0, 0);

    INPUT0_TYPE in1 = input[input0_offset / 2] & 0xFF;

    INPUT0_TYPE packed_out_channels = in1;

    const uint output_idx = GET_FILTER_OS_IS_YX_OSV_ISV_INDEX_INT4_PACKED(OUTPUT, o, i/2, 0, 0, 32); // Calculate offset as osv16 due to packing
    output[output_idx] = packed_out_channels;
#else
#error "reorder_weights_int4: unsupported layouts combination"
#endif
}
