
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
# mypy: ignore-errors

from typing import Optional, Any
import os
from openvino.runtime import  Core

def _get_device(options) -> Optional[Any]:
    core = Core()
    device = "CPU"

    if "device_id" in options:
        device = options["device_id"]
    else:
        device = os.getenv("OPENVINO_TORCH_BACKEND_DEVICE")

    if device is not None:
        assert device in core.available_devices, "Specified device " + device + " is not in the list of OpenVINO Available Devices"

    return device

def _get_cache_dir(options) -> Optional[Any]:
    if options is not None and "cache_dir" in options:
        return options["cache_dir"]
    else:
        return os.getenv("OPENVINO_TORCH_CACHE_DIR")