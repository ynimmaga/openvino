Latent Consistency Model using Optimum-Intel OpenVINO
=====================================================

This notebook provides instructions how to run Latent Consistency Model
(LCM). It allows to setup standard Hugging Face diffusers pipeline and
Optimum Intel pipeline optimized for Intel hardware including CPU and
GPU. Running inference on CPU and GPU it is easy to compare performance
and time required to generate an image for provided prompt. The notebook
can be also used on other Intel hardware with minimal or no
modifications.

|image0|

Optimum Intel is an interface from Hugging Face between both diffusers
and transformers libraries and various tools provided by Intel to
accelerate pipelines on Intel hardware. It allows to perform
quantization of the models hosted on Hugging Face. In this notebook
OpenVINO is used for AI-inference acceleration as a backend for Optimum
Intel!

For more details please refer to Optimum Intel repository
https://github.com/huggingface/optimum-intel

LCMs are the next generation of generative models after Latent Diffusion
Models (LDMs). They are proposed to overcome the slow iterative sampling
process of Latent Diffusion Models (LDMs), enabling fast inference with
minimal steps (from 2 to 4) on any pre-trained LDMs (e.g. Stable
Diffusion). To read more about LCM please refer to
https://latent-consistency-models.github.io/

Table of contents:
^^^^^^^^^^^^^^^^^^

-  `Prerequisites <#prerequisites>`__
-  `Full precision model on the
   CPU <#using-full-precision-model-in-cpu-with-latentconsistencymodelpipeline>`__
-  `Running inference using Optimum Intel
   OVLatentConsistencyModelPipeline <#running-inference-using-optimum-intel-ovlatentconsistencymodelpipeline>`__

.. |image0| image:: https://github.com/openvinotoolkit/openvino_notebooks/assets/10940214/1858dae4-72fd-401e-b055-66d503d82446

Prerequisites
~~~~~~~~~~~~~



Install required packages

.. code:: ipython3

    %pip install -q "openvino>=2023.3.0"
    %pip install -q "onnx>=1.11.0"
    %pip install -q "optimum-intel[diffusers]@git+https://github.com/huggingface/optimum-intel.git" "ipywidgets" "torch>=2.1" "transformers>=4.33.0" --extra-index-url https://download.pytorch.org/whl/cpu


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. code:: ipython3

    import warnings
    
    warnings.filterwarnings("ignore")

Showing Info Available Devices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



The ``available_devices`` property shows the available devices in your
system. The “FULL_DEVICE_NAME” option to ``ie.get_property()`` shows the
name of the device. Check what is the ID name for the discrete GPU, if
you have integrated GPU (iGPU) and discrete GPU (dGPU), it will show
``device_name="GPU.0"`` for iGPU and ``device_name="GPU.1"`` for dGPU.
If you just have either an iGPU or dGPU that will be assigned to
``"GPU"``

Note: For more details about GPU with OpenVINO visit this
`link <https://docs.openvino.ai/2024/get-started/configurations/configurations-intel-gpu.html>`__.
If you have been facing any issue in Ubuntu 20.04 or Windows 11 read
this
`blog <https://blog.openvino.ai/blog-posts/install-gpu-drivers-windows-ubuntu>`__.

.. code:: ipython3

    import openvino as ov
    
    core = ov.Core()
    devices = core.available_devices
    
    for device in devices:
        device_name = core.get_property(device, "FULL_DEVICE_NAME")
        print(f"{device}: {device_name}")


.. parsed-literal::

    CPU: Intel(R) Core(TM) i9-10920X CPU @ 3.50GHz


Using full precision model in CPU with ``LatentConsistencyModelPipeline``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Standard pipeline for the Latent Consistency Model(LCM) from Diffusers
library is used here. For more information please refer to
https://huggingface.co/docs/diffusers/en/api/pipelines/latent_consistency_models

.. code:: ipython3

    from diffusers import LatentConsistencyModelPipeline
    import gc
    
    pipeline = LatentConsistencyModelPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")


.. parsed-literal::

    2024-04-18 00:04:08.916564: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-04-18 00:04:08.953118: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.


.. parsed-literal::

    2024-04-18 00:04:09.550802: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT



.. parsed-literal::

    Loading pipeline components...:   0%|          | 0/7 [00:00<?, ?it/s]


.. code:: ipython3

    prompt = "A cute squirrel in the forest, portrait, 8k"
    
    image = pipeline(prompt=prompt, num_inference_steps=4, guidance_scale=8.0).images[0]
    image.save("image_standard_pipeline.png")
    image



.. parsed-literal::

      0%|          | 0/4 [00:00<?, ?it/s]




.. image:: latent-consistency-models-optimum-demo-with-output_files/latent-consistency-models-optimum-demo-with-output_8_1.png



.. code:: ipython3

    del pipeline
    gc.collect();

Select inference device for text-to-image generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    import ipywidgets as widgets
    
    core = ov.Core()
    
    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value="CPU",
        description="Device:",
        disabled=False,
    )
    
    device




.. parsed-literal::

    Dropdown(description='Device:', options=('CPU', 'AUTO'), value='CPU')



Running inference using Optimum Intel ``OVLatentConsistencyModelPipeline``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Accelerating inference of LCM using Intel Optimum with OpenVINO backend.
For more information please refer to
https://huggingface.co/docs/optimum/intel/inference#latent-consistency-models.
The pretrained model in this notebook is available on Hugging Face in
FP32 precision and in case if CPU is selected as a device, then
inference runs with full precision. For GPU accelerated AI-inference is
supported for FP16 data type and FP32 precision for GPU may produce high
memory footprint and latency. Therefore, default precision for GPU in
OpenVINO is FP16. OpenVINO GPU Plugin converts FP32 to FP16 on the fly
and there is no need to do it manually

.. code:: ipython3

    from optimum.intel.openvino import OVLatentConsistencyModelPipeline
    
    ov_pipeline = OVLatentConsistencyModelPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", export=True, compile=False)
    ov_pipeline.reshape(batch_size=1, height=768, width=768, num_images_per_prompt=1)
    ov_pipeline.save_pretrained("./openvino_ir")


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino


.. parsed-literal::

    Framework not specified. Using pt to export the model.


.. parsed-literal::

    Keyword arguments {'subfolder': '', 'trust_remote_code': False} are not expected by StableDiffusionPipeline and will be ignored.



.. parsed-literal::

    Loading pipeline components...:   0%|          | 0/7 [00:00<?, ?it/s]


.. parsed-literal::

    Using framework PyTorch: 2.2.2+cpu


.. parsed-literal::

    WARNING:tensorflow:Please fix your imports. Module tensorflow.python.training.tracking.base has been moved to tensorflow.python.trackable.base. The old module will be deleted in version 2.11.


.. parsed-literal::

    [ WARNING ]  Please fix your imports. Module %s has been moved to %s. The old module will be deleted in version %s.


.. parsed-literal::

    Using framework PyTorch: 2.2.2+cpu


.. parsed-literal::

    Using framework PyTorch: 2.2.2+cpu


.. parsed-literal::

    Using framework PyTorch: 2.2.2+cpu


.. code:: ipython3

    ov_pipeline.to(device.value)
    ov_pipeline.compile()


.. parsed-literal::

    Compiling the vae_decoder to CPU ...


.. parsed-literal::

    Compiling the unet to CPU ...


.. parsed-literal::

    Compiling the vae_encoder to CPU ...


.. parsed-literal::

    Compiling the text_encoder to CPU ...


.. code:: ipython3

    prompt = "A cute squirrel in the forest, portrait, 8k"
    
    image_ov = ov_pipeline(prompt=prompt, num_inference_steps=4, guidance_scale=8.0).images[0]
    image_ov.save("image_opt.png")
    image_ov



.. parsed-literal::

      0%|          | 0/4 [00:00<?, ?it/s]




.. image:: latent-consistency-models-optimum-demo-with-output_files/latent-consistency-models-optimum-demo-with-output_15_1.png



.. code:: ipython3

    del ov_pipeline
    gc.collect();
