
# How to Add OpenVINO Support to Unsupported Ops when using torch.compile with openvino backend.

##  Introduction

This guide covers the steps to add support for operations that are currently unsupported when using the OpenVINO  via [`torch.compile` feature](https://docs.openvino.ai/2023.2/pytorch_2_0_torch_compile.html). By following these steps, you can extend OpenVINO to support additional PyTorch operations via the TorchDynamo integration.

## Prerequisites

-   Knowledge of Python and C++ programming
-   Familiarity with PyTorch and OpenVINO.
-   OpenVINO source code downloaded locally for modification.

## Step 1: Identify Unsupported Ops in TorchFX

### 1.1. Setup OpenVINO source code.
```bash
git clone https://github.com/openvinotoolkit/openvino.git
cd openvino
git submodule update --init --recursive
chmod +x install_build_dependencies.sh
sudo ./install_build_dependencies.sh
```
### 1.2. Modify code to print unsupported ops.

- Update the `is_node_supported()` function in [op_support.py](https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/python/src/openvino/frontend/pytorch/torchdynamo/op_support.py#L118) with the following which will print unsupported ops.

```python
    def is_node_supported(self, submodules: t.Mapping[str, Module], node: Node) -> bool:
        # OpenVINO FX subgraph should be purely functional
        if node.op not in CALLABLE_NODE_OPS:
            print(f":DEBUG - is_node_supported - target:{node.target}:supported:False")
            return False

        # ops in supported_dict doesn't have overload name
        # use overloadpacket's qualified_name for OpOverload
        if isinstance(node.target, OpOverload):
            target = _get_qualified_name(node.target.overloadpacket)

            if target in self._support_dict:
                print(f":DEBUG - is_node_supported - target:{node.target}:supported:True")
                return True
        res = super().is_node_supported(submodules, node)
        if res:
            print(f":DEBUG - is_node_supported - target:{node.target}:supported:True")
        else:
            print(f":DEBUG - is_node_supported - target:{node.target}:supported:False")
        return res
        # return super().is_node_supported(submodules, node)
```

### 1.3. Build OpenVINO
```bash
# Go to the openvino root folder.
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_PYTHON=ON -DENABLE_WHEEL=ON
make -j4 # here 4 is used, modify it based on num threads on your system.
cd wheels
pip install openvino-20*
```

```bash
# By disabling unused features like below, build process may be faster.
cmake .. \
-DCMAKE_BUILD_TYPE=Release \
-DENABLE_PYTHON=ON \
-DENABLE_WHEEL=ON \
-DENABLE_MULTI=OFF \
-DENABLE_AUTO=OFF \
-DENABLE_HETERO=OFF \
-DENABLE_INTEL_GPU=OFF \
-DENABLE_INTEL_GNA=OFF
```

### 1.4. Run the application
Run the model with `torch.compile(model, backend="openvino")`, now with the debug statements, Identify Unsupported Operations.
```bash
#To run with TorchFX backend, use env variable PYTORCH_TRACING_MODE
export PYTORCH_TRACING_MODE=TORCHFX
```

## Step 2: Modify op_table.cpp

### 2.1. Open [op_table.cpp](https://github.com/openvinotoolkit/openvino/blob/master/src/frontends/pytorch/src/op_table.cpp).

### 2.2. Locate the following functions:
   - `get_supported_ops_ts()` (returns supported ops for TorchScript)
   - `get_supported_ops_fx()` (returns supported ops for TorchFX)

-- If the unsupported op is available `get_supported_ops_ts()`, add it to `get_supported_ops_fx()`.

   Example:
   ```cpp
   // Inside get_supported_ops_fx()
   {"aten.bitwise_not.default", op::translate_bitwise_not}
   ```

-- If the unsupported operation is not present in `get_supported_ops_ts()`, proceed to Step 2.3 to implement its support.

### 2.3. Implement Unsupported Op Support
- If the unsupported op is not in `get_supported_ops_ts()`, implement its functionality in the appropriate file or create a new file in the [`op` directory](https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/pytorch/src/op/).
- Once the operation implementation is complete, define the corresponding OP_CONVERTOR in  [op_table.cpp](https://github.com/openvinotoolkit/openvino/blob/master/src/frontends/pytorch/src/op_table.cpp).
   ```cpp
   // Example for translate_transpose_fx
   OP_CONVERTER(translate_transpose_fx);
   ```
 - Finally, add  the newly added op in the `get_supported_ops_fx()` in [op_table.cpp] (https://github.com/openvinotoolkit/openvino/blob/master/src/frontends/pytorch/src/op_table.cpp) as mentioned in Step 2.2.

## Step 3: Add Op Support to op_support.py

### 3.1. Open [op_support.py](https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/python/src/openvino/frontend/pytorch/torchdynamo/op_support.py).

### 3.2. Add the newly supported op to the `op_support_dict` in the `__init__` function.

   Example:
   ```python
   # Inside __init__ function in op_support.py
   "aten.bitwise_not.default": op.translate_bitwise_not,
   ```

## Step 4: Rebuild OpenVINO

- Rebuild the OpenVINO  to apply the changes.
- The unsupported op should now be supported through `torch.compile(model, backend="openvino")`!

## Conclusion

Follow these steps to enable support for previously unsupported ops in the OpenVINO PyTorch frontend using TorchDynamo. Thoroughly test the modifications to ensure correct functionality.
