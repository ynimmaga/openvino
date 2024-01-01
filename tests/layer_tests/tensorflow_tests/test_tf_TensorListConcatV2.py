# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from sys import platform

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

class TestTensorListConcatV2(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'x' in inputs_info
        x_shape = inputs_info['x']
        inputs_data = {}
        inputs_data['x'] = np.random.randint(-10, 10, x_shape).astype(self.input_type)
        return inputs_data

    def create_tensor_list_resize(self, input_shape, input_type):
        self.input_type = input_type
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(input_type, input_shape, 'x')
            tensor_list = tf.raw_ops.TensorListFromTensor(tensor=x,
                                                          element_shape=tf.constant(input_shape[1:], dtype=tf.int32))
            tf.raw_ops.TensorListConcatV2(input_handle=tensor_list, element_shape=tf.constant(input_shape[1:], dtype=tf.int32),
                                       element_dtype=input_type,
                                          leading_dims=tf.constant([], dtype=tf.int64))
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(input_shape=[10, 20], input_type=np.float32),
        dict(input_shape=[2, 3, 4], input_type=np.int32),
        dict(input_shape=[3,2,4], input_type=np.float32),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    @pytest.mark.skipif(platform == 'darwin', reason="Ticket - 122182")
    def test_tensor_list_resize_basic(self, params, ie_device, precision, ir_version, temp_dir,
                                      use_new_frontend, use_old_api):
        self._test(*self.create_tensor_list_resize(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)
