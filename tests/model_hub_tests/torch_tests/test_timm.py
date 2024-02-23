# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import timm
import torch
from models_hub_common.constants import hf_hub_cache_dir
from models_hub_common.utils import cleanup_dir, get_models_list

from torch_utils import TestTorchConvertModel, process_pytest_marks
from openvino import convert_model
from torch.export import export
from packaging import version


def filter_timm(timm_list: list) -> list:
    unique_models = dict()
    filtered_list = []
    ignore_list = ["base", "xxtiny", "xxs", "pico", "xtiny", "xs", "nano", "tiny", "s", "mini", "small", "lite",
                   "medium", "m", "big", "large", "l", "xlarge", "xl", "huge", "xxlarge", "gigantic", "giant", "enormous"]
    ignore_set = set(ignore_list)
    for name in sorted(timm_list):
        # first: remove datasets
        name_parts = name.split(".")
        _name = "_".join(name.split(".")[:-1]) if len(name_parts) > 1 else name
        # second: remove sizes
        name_set = set([n for n in _name.split("_") if not n.isnumeric()])
        size_set = name_set.intersection(ignore_set)
        size_idx = 100
        if len(size_set) > 0:
            size_idx = ignore_list.index(list(sorted(size_set))[0])
        name_set = name_set.difference(ignore_set)
        name_join = "_".join(sorted(name_set))
        if name_join not in unique_models:
            unique_models[name_join] = (size_idx, name)
            filtered_list.append(name)
        elif unique_models[name_join][0] > size_idx:
            unique_models[name_join] = (size_idx, name)
    return sorted([v[1] for v in unique_models.values()])


def get_all_models() -> list:
    return process_pytest_marks(os.path.join(os.path.dirname(__file__), "timm_models"))


# To make tests reproducible we seed the random generator
torch.manual_seed(0)


class TestTimmConvertModel(TestTorchConvertModel):
    def load_model_impl(self, model_name, model_link):
        m = timm.create_model(model_name, pretrained=True)
        cfg = timm.get_pretrained_cfg(model_name)
        shape = [1] + list(cfg.input_size)
        self.example = (torch.randn(shape),)
        self.inputs = (torch.randn(shape),)
        return m

    def infer_fw_model(self, model_obj, inputs):
        fw_outputs = model_obj(*[torch.from_numpy(i) for i in inputs])
        if isinstance(fw_outputs, dict):
            for k in fw_outputs.keys():
                fw_outputs[k] = fw_outputs[k].numpy(force=True)
        elif isinstance(fw_outputs, (list, tuple)):
            fw_outputs = [o.numpy(force=True) for o in fw_outputs]
        else:
            fw_outputs = [fw_outputs.numpy(force=True)]
        return fw_outputs

    def teardown_method(self):
        # remove all downloaded files from cache
        cleanup_dir(hf_hub_cache_dir)
        super().teardown_method()

    @pytest.mark.parametrize("name", ["mobilevitv2_050.cvnets_in1k",
                                      "poolformerv2_s12.sail_in1k",
                                      "vit_base_patch8_224.augreg_in21k",
                                      "beit_base_patch16_224.in22k_ft_in22k",
                                      "sequencer2d_l.in1k",
                                      "gcresnext26ts.ch_in1k"])
    @pytest.mark.precommit
    def test_convert_model_precommit(self, name, ie_device):
        self.mode = "trace"
        self.run(name, None, ie_device)

    @pytest.mark.nightly
    @pytest.mark.parametrize("mode", ["trace"]) # disable "export" for now
    @pytest.mark.parametrize("name", get_all_models())
    def test_convert_model_all_models(self, mode, name, ie_device):
        self.mode = mode
        self.run(name, None, ie_device)

    @pytest.mark.nightly
    def test_models_list_complete(self, ie_device):
        m_list = timm.list_pretrained()
        all_models_ref = set(filter_timm(m_list))
        all_models = set([m for m, _, _, _ in get_models_list(
            os.path.join(os.path.dirname(__file__), "timm_models"))])
        assert all_models == all_models_ref, f"Lists of models are not equal."
