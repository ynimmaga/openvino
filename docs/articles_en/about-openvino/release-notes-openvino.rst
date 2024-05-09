.. meta::
   :description: See what has changed in OpenVINO with the latest release, as well as all
                 previous releases in this year's cycle.

OpenVINO Release Notes
=============================

.. toctree::
   :maxdepth: 1
   :hidden:

   release-notes-openvino/system-requirements
   release-notes-openvino/release-policy



2024.1 - 24 April 2024
#############################

:doc:`System Requirements <./release-notes-openvino/system-requirements>` | :doc:`Release policy <./release-notes-openvino/release-policy>` | :doc:`Installation Guides <./../get-started/install-openvino>`


What's new
+++++++++++++++++++++++++++++

* More Gen AI coverage and framework integrations to minimize code changes.

  * Mixtral and URLNet models optimized for performance improvements on Intel® Xeon® processors.
  * Stable Diffusion 1.5, ChatGLM3-6B, and Qwen-7B models optimized for improved inference speed
    on Intel® Core™ Ultra processors with integrated GPU.
  * Support for Falcon-7B-Instruct, a GenAI Large Language Model (LLM) ready-to-use chat/instruct
    model with superior performance metrics.
  * New Jupyter Notebooks added: YOLO V9, YOLO V8 Oriented Bounding Boxes Detection (OOB), Stable
    Diffusion in Keras, MobileCLIP, RMBG-v1.4 Background Removal, Magika, TripoSR, AnimateAnyone,
    LLaVA-Next, and RAG system with OpenVINO and LangChain.

* Broader LLM model support and more model compression techniques.

  * LLM compilation time reduced through additional optimizations with compressed embedding.
    Improved 1st token performance of LLMs on 4th and 5th generations of Intel® Xeon® processors
    with Intel® Advanced Matrix Extensions (Intel® AMX).
  * Better LLM compression and improved performance with oneDNN, INT4, and INT8 support for
    Intel® Arc™ GPUs.
  * Significant memory reduction for select smaller GenAI models on Intel® Core™ Ultra processors
    with integrated GPU.

* More portability and performance to run AI at the edge, in the cloud, or locally.

  * The preview NPU plugin for Intel® Core™ Ultra processors is now available in the OpenVINO
    open-source GitHub repository, in addition to the main OpenVINO package on PyPI.
  * The JavaScript API is now more easily accessible through the npm repository, enabling
    JavaScript developers' seamless access to the OpenVINO API.
  * FP16 inference on ARM processors now enabled for the Convolutional Neural Network (CNN) by
    default.


OpenVINO™ Runtime
+++++++++++++++++++++++++++++

Common
-----------------------------

* Unicode file paths for cached models are now supported on Windows.
* Pad pre-processing API to extend input tensor on edges with constants.
* A fix for inference failures of certain image generation models has been implemented
  (fused I/O port names after transformation).
* Compiler's warnings-as-errors option is now on, improving the coding criteria and quality.
  Build warnings will not be allowed for new OpenVINO code and the existing warnings have been
  fixed.

AUTO Inference Mode
-----------------------------

* Returning the ov::enable_profiling value from ov::CompiledModel is now supported.

CPU Device Plugin
-----------------------------

* 1st token performance of LLMs has been improved on the 4th and 5th generations of Intel® Xeon®
  processors with Intel® Advanced Matrix Extensions (Intel® AMX).
* LLM compilation time and memory footprint have been improved through additional optimizations
  with compressed embeddings.
* Performance of MoE (e.g. Mixtral), Gemma, and GPT-J has been improved further.
* Performance has been improved significantly for a wide set of models on ARM devices.
* FP16 inference precision is now the default for all types of models on ARM devices.
* CPU architecture-agnostic build has been implemented, to enable unified binary distribution
  on different ARM devices.

GPU Device Plugin
-----------------------------

* LLM first token latency has been improved on both integrated and discrete GPU platforms.
* For the ChatGLM3-6B model, average token latency has been improved on integrated GPU platforms.
* For Stable Diffusion 1.5 FP16 precision, performance has been improved on Intel® Core™ Ultra
  processors.

NPU Device Plugin
-----------------------------

* NPU Plugin is now part of the OpenVINO GitHub repository. All the most recent plugin changes
  will be immediately available in the repo. Note that NPU is part of Intel® Core™ Ultra
  processors.
* New OpenVINO™ notebook “Hello, NPU!” introducing NPU usage with OpenVINO has been added.
* Version 22H2 or later is required for Microsoft Windows® 11 64-bit to run inference on NPU.

OpenVINO Python API
-----------------------------

* GIL-free creation of RemoteTensors is now used - holding GIL means that the process is not suited
  for multithreading and removing the GIL lock will increase performance which is critical for
  the concept of Remote Tensors.
* Packed data type BF16 on the Python API level has been added, opening a new way of supporting
  data types not handled by numpy.
* 'pad' operator support for ov::preprocess::PrePostProcessorItem has been added.
* ov.PartialShape.dynamic(int) definition has been provided.


OpenVINO C API
-----------------------------

* Two new pre-processing APIs for scale and mean have been added.

OpenVINO Node.js API
-----------------------------

* New methods to align JavaScript API with CPP API have been added, such as
  CompiledModel.exportModel(), core.import_model(), Core set/get property and Tensor.get_size(),
  and Model.is_dynamic().
* Documentation has been extended to help developers start integrating JavaScript applications
  with OpenVINO™.

TensorFlow Framework Support
-----------------------------

* `tf.keras.layers.TextVectorization tokenizer <https://www.tensorflow.org/api_docs/python/tf/keras/layers/TextVectorization>`__
  is now supported.
* Conversion of models with Variable and HashTable (dictionary) resources has been improved.
* 8 NEW operations have been added
  (`see the list here, marked as NEW <https://github.com/openvinotoolkit/openvino/blob/releases/2024/1/src/frontends/tensorflow/docs/supported_ops.md>`__).
* 10 operations have received complex tensor support.
* Input tensor names for TF1 models have been adjusted to have a single name per input.
* Hugging Face model support coverage has increased significantly, due to:

  * extraction of input signature of a model in memory has been fixed,
  * reading of variable values for a model in memory has been fixed.


PyTorch Framework Support
-----------------------------

* ModuleExtension, a new type of extension for PyTorch models is now supported
  (`PR #23536 <https://github.com/openvinotoolkit/openvino/pull/23536>`__).
* 22 NEW operations have been added.
* Experimental support for models produced by torch.export (FX graph) has been added
  (`PR #23815 <https://github.com/openvinotoolkit/openvino/pull/23815>`__).

ONNX Framework Support
-----------------------------
* 8 new operations have been added.


OpenVINO Model Server
+++++++++++++++++++++++++++++

* OpenVINO™ Runtime backend used is now 2024.1
* OpenVINO™ models with String data type on output are supported. Now, OpenVINO™ Model Server
  can support models with input and output of the String type, so developers can take advantage
  of the tokenization built into the model as the first layer. Developers can also rely on any
  postprocessing embedded into the model which returns text only. Check the
  `demo on string input data with the universal-sentence-encoder model <https://docs.openvino.ai/2024/ovms_demo_universal-sentence-encoder.html>`__
  and the
  `String output model demo <https://github.com/openvinotoolkit/model_server/tree/main/demos/image_classification_with_string_output>`__.
* MediaPipe Python calculators have been updated to support relative paths for all related
  configuration and Python code files. Now, the complete graph configuration folder can be
  deployed in an arbitrary path without any code changes.
* KServe REST API support has been extended to properly handle the string format in JSON body,
  just like the binary format compatible with NVIDIA Triton™.
* `A demo showcasing a full RAG algorithm <https://github.com/openvinotoolkit/model_server/tree/main/demos/python_demos/rag_chatbot>`__
  fully delegated to the model server has been added.


Neural Network Compression Framework
++++++++++++++++++++++++++++++++++++++++++

* Model subgraphs can now be defined in the ignored scope for INT8 Post-training Quantization,
  nncf.quantize(), which simplifies excluding accuracy-sensitive layers from quantization.
* A batch size of more than 1 is now partially supported for INT8 Post-training Quantization,
  speeding up the process. Note that it is not recommended for transformer-based models as it
  may impact accuracy. Here is an
  `example demo <https://github.com/openvinotoolkit/nncf/blob/develop/examples/quantization_aware_training/torch/resnet18/README.md>`__.
* Now it is possible to apply fine-tuning on INT8 models after Post-training Quantization to
  improve model accuracy and make it easier to move from post-training to training-aware
  quantization. Here is an
  `example demo <https://github.com/openvinotoolkit/nncf/blob/develop/examples/quantization_aware_training/torch/resnet18/README.md>`__.

OpenVINO Tokenizers
++++++++++++++++++++++++++++++++++++++++++

* TensorFlow support has been extended - TextVectorization layer translation:

  * Aligned existing ops with TF ops and added a translator for them.
  * Added new ragged tensor ops and string ops.

* A new tokenizer type, RWKV is now supported:

  * Added Trie tokenizer and Fuse op for ragged tensors.
  * A new way to get OV Tokenizers: build a vocab from file.

* Tokenizer caching has been redesigned to work with the OpenVINO™ model caching mechanism.


Other Changes and Known Issues
++++++++++++++++++++++++++++++++++++++++++

Jupyter Notebooks
-----------------------------

The default branch for the OpenVINO™ Notebooks repository has been changed from 'main' to
'latest'. The 'main' branch of the notebooks repository is now deprecated and will be maintained
until September 30, 2024.

The new branch, 'latest', offers a better user experience and simplifies maintenance due to
significant refactoring and an improved directory naming structure.

Use the local
`README.md <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/README.md>`__
file and OpenVINO™ Notebooks at
`GitHub Pages <https://openvinotoolkit.github.io/openvino_notebooks/>`__
to navigate through the content.


The following notebooks have been updated or newly added:

* `Grounded Segment Anything <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/grounded-segment-anything/grounded-segment-anything.ipynb>`__
* `Visual Content Search with MobileCLIP <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/mobileclip-video-search/mobileclip-video-search.ipynb>`__
* `YOLO V8 Oriented Bounding Box Detection Optimization <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/yolov8-optimization/yolov8-obb.ipynb>`__
* `Magika: AI-powered fast and efficient file type identification <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/magika-content-type-recognition/magika-content-type-recognition.ipynb>`__
* `Keras Stable Diffusion <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/stable-diffusion-keras-cv/stable-diffusion-keras-cv.ipynb>`__
* `RMBG background removal <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/rmbg-background-removal/rmbg-background-removal.ipynb>`__
* `AnimateAnyone: pose guided image to video generation <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/animate-anyone/animate-anyone.ipynb>`__
* `LLaVA-Next visual-language assistant <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/llava-next-multimodal-chatbot/llava-next-multimodal-chatbot.ipynb>`__
* `TripoSR: single image 3d reconstruction <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/triposr-3d-reconstruction/triposr-3d-reconstruction.ipynb>`__
* `RAG system with OpenVINO and LangChain <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/llm-rag-langchain/llm-rag-langchain.ipynb>`__


Known Issues
-----------------------------

| **Component - CPU Plugin**
| *ID* - N/A
| *Description:*
|   Default CPU pinning policy on Windows has been changed to follow Windows' policy
    instead of controlling the CPU pinning in the OpenVINO plugin. This brings certain dynamic or
    performance variance on Windows. Developers can use ov::hint::enable_cpu_pinning to enable
    or disable CPU pinning explicitly.

| **Component - Hardware Configuration**
| *ID* - N/A
| *Description:*
|   Reduced performance for LLMs may be observed on newer CPUs. To mitigate, modify the default settings in BIOS to
|   change the system into 2 NUMA node system:
|    1. Enter the BIOS configuration menu.
|    2. Select EDKII Menu -> Socket Configuration -> Uncore Configuration -> Uncore General Configuration ->  SNC.
|    3. The SNC setting is set to *AUTO* by default. Change the SNC setting to *disabled* to configure one NUMA node per processor socket upon boot.
|    4. After system reboot, confirm the NUMA node setting using: `numatcl -H`. Expect to see only nodes 0 and 1 on a
|    2-socket system with the following mapping:
|     Node - 0  -  1
|      0  - 10  -  21
|      1 -  21  -  10


Previous 2024 releases
+++++++++++++++++++++++++++++

.. dropdown:: 2024.0 - 06 March 2024
   :animate: fade-in-slide-down
   :color: secondary

   **What's new**

   * More Generative AI coverage and framework integrations to minimize code changes.

     * Improved out-of-the-box experience for TensorFlow sentence encoding models through the
       installation of OpenVINO™ toolkit Tokenizers.
     * New and noteworthy models validated:
       Mistral, StableLM-tuned-alpha-3b, and StableLM-Epoch-3B.
     * OpenVINO™ toolkit now supports Mixture of Experts (MoE), a new architecture that helps
       process more efficient generative models through the pipeline.
     * JavaScript developers now have seamless access to OpenVINO API. This new binding enables a
       smooth integration with JavaScript API.

   * Broader Large Language Model (LLM) support and more model compression techniques.

     * Broader Large Language Model (LLM) support and more model compression techniques.
     * Improved quality on INT4 weight compression for LLMs by adding the popular technique,
       Activation-aware Weight Quantization, to the Neural Network Compression Framework (NNCF).
       This addition reduces memory requirements and helps speed up token generation.
     * Experience enhanced LLM performance on Intel® CPUs, with internal memory state enhancement,
       and INT8 precision for KV-cache. Specifically tailored for multi-query LLMs like ChatGLM.
     * The OpenVINO™ 2024.0 release makes it easier for developers, by integrating more OpenVINO™
       features with the Hugging Face ecosystem. Store quantization configurations for popular
       models directly in Hugging Face to compress models into INT4 format while preserving
       accuracy and performance.

   * More portability and performance to run AI at the edge, in the cloud, or locally.

     * A preview plugin architecture of the integrated Neural Processor Unit (NPU) as part of
       Intel® Core™ Ultra processor (codename Meteor Lake) is now included in the main OpenVINO™
       package on PyPI.
     * Improved performance on ARM by enabling the ARM threading library. In addition, we now
       support multi-core ARM processors and enabled FP16 precision by default on MacOS.
     * New and improved LLM serving samples from OpenVINO Model Server for multi-batch inputs and
       Retrieval Augmented Generation (RAG).


   **OpenVINO™ Runtime**

   *Common*

   * The legacy API for CPP and Python bindings has been removed.
   * StringTensor support has been extended by operators such as ``Gather``, ``Reshape``, and
     ``Concat``, as a foundation to improve support for tokenizer operators and compliance with
     the TensorFlow Hub.
   * oneDNN has been updated to v3.3.
     (`see oneDNN release notes <https://github.com/oneapi-src/oneDNN/releases>`__).


   *CPU Device Plugin*

   * LLM performance on Intel® CPU platforms has been improved for systems based on AVX2 and
     AVX512, using dynamic quantization and internal memory state optimization, such as INT8
     precision for KV-cache. 13th and 14th generations of Intel® Core™ processors and Intel® Core™
     Ultra processors use AVX2 for CPU execution, and these platforms will benefit from speedup.
     Enable these features by setting ``"DYNAMIC_QUANTIZATION_GROUP_SIZE":"32"`` and
     ``"KV_CACHE_PRECISION":"u8"`` in the configuration file.
   * The ``ov::affinity`` API configuration is now deprecated and will be removed in release
     2025.0.
   * The following have been improved and optimized:

     * Multi-query structure LLMs (such as ChatGLM 2/3) for BF16 on the 4th and 5th generation
       Intel® Xeon® Scalable processors.
     * `Mixtral <https://huggingface.co/docs/transformers/model_doc/mixtral>`__ model performance.
     * 8-bit compressed LLM compilation time and memory usage, valuable for models with large
       embeddings like `Qwen <https://github.com/QwenLM/Qwen>`__.
     * Convolutional networks in FP16 precision on ARM processors.

   *GPU Device Plugin*

   * The following have been improved and optimized:

     * Average token latency for LLMs on integrated GPU (iGPU) platforms, using INT4-compressed
       models with large context size on Intel® Core™ Ultra processors.
     * LLM beam search performance on iGPU. Both average and first-token latency decrease may be
       expected for larger context sizes.
     * Multi-batch performance of YOLOv5 on iGPU platforms.

   * Memory usage for LLMs has been optimized, enabling '7B' models with larger context on
     16Gb platforms.

   *NPU Device Plugin (preview feature)*

   * The NPU plugin for OpenVINO™ is now available through PyPI (run “pip install openvino”).

   *OpenVINO Python API*

   * ``.add_extension`` method signatures have been aligned, improving API behavior for better
     user experience.

   *OpenVINO C API*

   * ov_property_key_cache_mode (C++ ov::cache_mode) now enables the ``optimize_size`` and
     ``optimize_speed`` modes to set/get model cache.
   * The VA surface on Windows exception has been fixed.

   *OpenVINO Node.js API*

   * OpenVINO - `JS bindings <https://docs.openvino.ai/2024/api/nodejs_api/nodejs_api.html>`__
     are consistent with the OpenVINO C++ API.
   * A new distribution channel is now available: Node Package Manager (npm) software registry
     (:doc:`check the installation guide <../get-started/install-openvino/install-openvino-npm>`).
   * JavaScript API is now available for Windows users, as some limitations for platforms other
     than Linux have been removed.

   *TensorFlow Framework Support*

   * String tensors are now natively supported, handled on input, output, and intermediate layers
     (`PR #22024 <https://github.com/openvinotoolkit/openvino/pull/22024>`__).

     * TensorFlow Hub universal-sentence-encoder-multilingual inferred out of the box
     * string tensors supported for ``Gather``, ``Concat``, and ``Reshape`` operations
     * integration with openvino-tokenizers module - importing openvino-tokenizers automatically
       patches TensorFlow FE with the required translators for models with tokenization

   * Fallback for Model Optimizer by operation to the legacy Frontend is no longer available.
     Fallback by .json config will remain until Model Optimizer is discontinued
     (`PR #21523 <https://github.com/openvinotoolkit/openvino/pull/21523>`__).
   * Support for the following has been added:

     * Mutable variables and resources such as HashTable*, Variable, VariableV2
       (`PR #22270 <https://github.com/openvinotoolkit/openvino/pull/22270>`__).
     * New tensor types: tf.u16, tf.u32, and tf.u64
       (`PR #21864 <https://github.com/openvinotoolkit/openvino/pull/21864>`__).
     * 14 NEW Ops*.
       `Check the list here (marked as NEW) <https://github.com/openvinotoolkit/openvino/blob/releases/2024/0/src/frontends/tensorflow/docs/supported_ops.md>`__.
     * TensorFlow 2.15
       (`PR #22180 <https://github.com/openvinotoolkit/openvino/pull/22180>`__).

   * The following issues have been fixed:

     * UpSampling2D conversion crashed when input type as int16
       (`PR #20838 <https://github.com/openvinotoolkit/openvino/pull/20838>`__).
     * IndexError list index for Squeeze
       (`PR #22326 <https://github.com/openvinotoolkit/openvino/pull/22326>`__).
     * Correct FloorDiv computation for signed integers
       (`PR #22684 <https://github.com/openvinotoolkit/openvino/pull/22684>`__).
     * Fixed bad cast error for tf.TensorShape to ov.PartialShape
       (`PR #22813 <https://github.com/openvinotoolkit/openvino/pull/22813>`__).
     * Fixed reading tf.string attributes for models in memory
       (`PR #22752 <https://github.com/openvinotoolkit/openvino/pull/22752>`__).


   *ONNX Framework Support*

   * ONNX Frontend now uses the OpenVINO API 2.0.

   *PyTorch Framework Support*

   * Names for outputs unpacked from dict or tuple are now clearer
     (`PR #22821 <https://github.com/openvinotoolkit/openvino/pull/22821>`__).
   * FX Graph (torch.compile) now supports kwarg inputs, improving data type coverage.
     (`PR #22397 <https://github.com/openvinotoolkit/openvino/pull/22397>`__).


   **OpenVINO Model Server**

   * OpenVINO™ Runtime backend used is now 2024.0.
   * Text generation demo now supports multi batch size, with streaming and unary clients.
   * The REST client now supports servables based on mediapipe graphs, including python pipeline
     nodes.
   * Included dependencies have received security-related updates.
   * Reshaping a model in runtime based on the incoming requests (auto shape and auto batch size)
     is deprecated and will be removed in the future. Using OpenVINO's dynamic shape models is
     recommended instead.


   **Neural Network Compression Framework (NNCF)**

   * The `Activation-aware Weight Quantization (AWQ) <https://arxiv.org/abs/2306.00978>`__
     algorithm for data-aware 4-bit weights compression is now available. It facilitates better
     accuracy for compressed LLMs with high ratio of 4-bit weights. To enable it, use the
     dedicated ``awq`` optional parameter of ``the nncf.compress_weights()`` API.
   * ONNX models are now supported in Post-training Quantization with Accuracy Control, through
     the ``nncf.quantize_with_accuracy_control()``, method. It may be used for models in the
     OpenVINO IR and ONNX formats.
   * A `weight compression example tutorial <https://github.com/openvinotoolkit/nncf/tree/develop/examples/llm_compression/openvino/tiny_llama_find_hyperparams>`__
     is now available, demonstrating how to find the appropriate hyperparameters for the TinyLLama
     model from the Hugging Face Transformers, as well as other LLMs, with some modifications.


   **OpenVINO Tokenizer**

   * Regex support has been improved.
   * Model coverage has been improved.
   * Tokenizer metadata has been added to rt_info.
   * Limited support for Tensorflow Text models has been added: convert MUSE for TF Hub with
     string inputs.
   * OpenVINO Tokenizers have their own repository now:
     `/openvino_tokenizers <https://github.com/openvinotoolkit/openvino_tokenizers>`__


   **Other Changes and Known Issues**

   *Jupyter Notebooks*

   The following notebooks have been updated or newly added:

   * `Mobile language assistant with MobileVLM <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/279-mobilevlm-language-assistant>`__
   * `Depth estimation with DepthAnything <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/280-depth-anything>`__
   * `Kosmos-2 <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/281-kosmos2-multimodal-large-language-model>`__
   * `Zero-shot Image Classification with SigLIP <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/282-siglip-zero-shot-image-classification>`__
   * `Personalized image generation with PhotoMaker <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/283-photo-maker>`__
   * `Voice tone cloning with OpenVoice <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/284-openvoice>`__
   * `Line-level text detection with Surya <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/285-surya-line-level-text-detection>`__
   * `InstantID: Zero-shot Identity-Preserving Generation using OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/286-instant-id>`__
   * `Tutorial for Big Image Transfer  (BIT) model quantization using NNCF <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/127-big-transfer-quantization>`__
   * `Tutorial for OpenVINO Tokenizers integration into inference pipelines <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/128-openvino-tokenizers>`__
   * `LLM chatbot <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/254-llm-chatbot/254-llm-chatbot.ipynb>`__ and
     `LLM RAG pipeline <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/254-llm-chatbot/254-rag-chatbot.ipynb>`__
     have received integration with new models: minicpm-2b-dpo, gemma-7b-it, qwen1.5-7b-chat, baichuan2-7b-chat


   *Known issues*

   | **Component - CPU Plugin**
   | *ID* - N/A
   | *Description:*
   |   Starting with 24.0, model inputs and outputs will no longer have tensor names, unless
       explicitly set to align with the PyTorch framework behavior.

   | **Component - GPU runtime**
   | *ID* - 132376
   | *Description:*
   |   First-inference latency slow down for LLMs on Intel® Core™ Ultra processors. Up to 10-20%
       drop may occur due to radical memory optimization for processing long sequences
       (about 1.5-2 GB reduced memory usage).

   | **Component - CPU runtime**
   | *ID* - N/A
   | *Description:*
   |   Performance results (first token latency) may vary from those offered by the previous OpenVINO version, for
       “latency” hint inference of LLMs with long prompts on Xeon platforms with 2 or more
       sockets. The reason is that all CPU cores of just the single socket running the application
       are employed, lowering the memory overhead for LLMs when numa control is not used.
   | *Workaround:*
   |   The behavior is expected but stream and thread configuration may be used to include cores
       from all sockets.









Deprecation And Support
+++++++++++++++++++++++++++++
Using deprecated features and components is not advised. They are available to enable a smooth
transition to new solutions and will be discontinued in the future. To keep using discontinued
features, you will have to revert to the last LTS OpenVINO version supporting them.
For more details, refer to the :doc:`OpenVINO Legacy Features and Components <../documentation/legacy-features>`
page.

Discontinued in 2024
-----------------------------

* Runtime components:

  * Intel® Gaussian & Neural Accelerator (Intel® GNA). Consider using the Neural Processing
    Unit (NPU) for low-powered systems like Intel® Core™ Ultra or 14th generation and beyond.
  * OpenVINO C++/C/Python 1.0 APIs (see
    `2023.3 API transition guide <https://docs.openvino.ai/2023.3/openvino_2_0_transition_guide.html>`__
    for reference).
  * All ONNX Frontend legacy API (known as ONNX_IMPORTER_API).
  * ``PerfomanceMode.UNDEFINED`` property as part of the OpenVINO Python API.

* Tools:

  * Deployment Manager. See :doc:`installation <../get-started/install-openvino>` and
    :doc:`deployment <../get-started/install-openvino>` guides for current distribution
    options.
  * `Accuracy Checker <https://docs.openvino.ai/2023.3/omz_tools_accuracy_checker.html>`__.
  * `Post-Training Optimization Tool <https://docs.openvino.ai/2023.3/pot_introduction.html>`__
    (POT). Neural Network Compression Framework (NNCF) should be used instead.
  * A `Git patch <https://github.com/openvinotoolkit/nncf/tree/develop/third_party_integration/huggingface_transformers>`__
    for NNCF integration with `huggingface/transformers <https://github.com/huggingface/transformers>`__.
    The recommended approach is to use `huggingface/optimum-intel <https://github.com/huggingface/optimum-intel>`__
    for applying NNCF optimization on top of models from Hugging Face.
  * Support for Apache MXNet, Caffe, and Kaldi model formats. Conversion to ONNX may be used
    as a solution.

Deprecated and to be removed in the future
--------------------------------------------

* The OpenVINO™ Development Tools package (pip install openvino-dev) will be removed from
  installation options and distribution channels beginning with OpenVINO 2025.
* Model Optimizer will be discontinued with OpenVINO 2025.0. Consider using the
  :doc:`new conversion methods <../openvino-workflow/model-preparation/convert-model-to-ir>`
  instead. For more details, see the
  :doc:`model conversion transition guide <../documentation/legacy-features/transition-legacy-conversion-api>`.
* OpenVINO property Affinity API will be discontinued with OpenVINO 2025.0.
  It will be replaced with CPU binding configurations (``ov::hint::enable_cpu_pinning``).
* OpenVINO Model Server components:

  * “auto shape” and “auto batch size” (reshaping a model in runtime) will be removed in the
    future. OpenVINO's dynamic shape models are recommended instead.

* The following notebooks have been deprecated and will be removed. For an up-to-date listing
  of available notebooks, refer to
  `OpenVINO™ Notebook index (openvinotoolkit.github.io) <https://openvinotoolkit.github.io/openvino_notebooks/>`__.

  * `Handwritten OCR with OpenVINO™ <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/handwritten-ocr>`__

    * See alternative: `Optical Character Recognition (OCR) with OpenVINO™ <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/optical-character-recognition>`__,
    * See alternative: `PaddleOCR with OpenVINO™ <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/paddle-ocr-webcam>`__,
    * See alternative: `Handwritten Text Recognition Demo <https://docs.openvino.ai/2024/omz_demos_handwritten_text_recognition_demo_python.html>`__

  * `Image In-painting with OpenVINO™ <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/image-inpainting>`__

    * See alternative: `Image Inpainting Python Demo <https://docs.openvino.ai/2024/omz_demos_image_inpainting_demo_python.html>`__

  * `Interactive Machine Translation with OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/machine-translation>`__

    * See alternative: `Machine Translation Python* Demo <https://docs.openvino.ai/2024/omz_demos_machine_translation_demo_python.html>`__

  * `Open Model Zoo Tools Tutorial <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/model-tools>`__

    * No alternatives, demonstrates deprecated tools.

  * `Super Resolution with OpenVINO™ <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/vision-superresolution>`__

    * See alternative: `Super Resolution with PaddleGAN and OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/vision-paddlegan-superresolution>`__
    * See alternative:  `Image Processing C++ Demo <https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/image_processing_demo/cpp/README.md>`__

  * `Image Colorization with OpenVINO Tutorial <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/vision-image-colorization>`__
  * `Interactive Question Answering with OpenVINO™ <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/question-answering>`__

    * See alternative: `BERT Question Answering Embedding Python* Demo <https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/bert_question_answering_embedding_demo/python/README.md>`__
    * See alternative:  `BERT Question Answering Python* Demo <https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/bert_question_answering_demo/python/README.md>`__

  * `Vehicle Detection And Recognition with OpenVINO™ <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/vehicle-detection-and-recognition>`__

    * See alternative: `Security Barrier Camera C++ Demo  <https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/security_barrier_camera_demo/cpp/README.md>`__

  * `The attention center model with OpenVINO™ <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/attention-center>`_
  * `Image Generation with DeciDiffusion <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/decidiffusion-image-generation>`_
  * `Image generation with DeepFloyd IF and OpenVINO™ <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/deepfloyd-if>`_
  * `Depth estimation using VI-depth with OpenVINO™ <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/depth-estimation-videpth>`_
  * `Instruction following using Databricks Dolly 2.0 and OpenVINO™ <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/dolly-2-instruction-following>`_

    * See alternative: `LLM Instruction-following pipeline with OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/llm-question-answering>`__

  * `Image generation with FastComposer and OpenVINO™ <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/fastcomposer-image-generation>`__
  * `Video Subtitle Generation with OpenAI Whisper  <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/whisper-subtitles-generation>`__

    * See alternative: `Automatic speech recognition using Distil-Whisper and OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/distil-whisper-asr/distil-whisper-asr.ipynb>`__

  * `Introduction to Performance Tricks in OpenVINO™ <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/performance-tricks>`__
  * `Speaker Diarization with OpenVINO™ <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/pyannote-speaker-diarization>`__
  * `Subject-driven image generation and editing using BLIP Diffusion and OpenVINO  <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/blip-diffusion-subject-generation>`__
  * `Text Prediction with OpenVINO™ <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/text-prediction>`__
  * `Training to Deployment with TensorFlow and OpenVINO™ <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/tensorflow-training-openvino>`__
  * `Speech to Text with OpenVINO™ <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/speech-to-text>`__
  * `Convert and Optimize YOLOv7 with OpenVINO™ <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/yolov7-optimization>`__
  * `Quantize Data2Vec Speech Recognition Model using NNCF PTQ API <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/speech-recognition-quantization/speech-recognition-quantization-data2vec.ipynb>`__

    * See alternative: `Quantize Speech Recognition Models with accuracy control using NNCF PTQ API <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/quantizing-model-with-accuracy-control/speech-recognition-quantization-wav2vec2.ipynb>`__

  * `Semantic segmentation with LRASPP MobileNet v3 and OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/torchvision-zoo-to-openvino/lraspp-segmentation.ipynb>`__
  * `Video Recognition using SlowFast and OpenVINO™ <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/slowfast-video-recognition>`__

    * See alternative: `Live Action Recognition with OpenVINO™ <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/action-recognition-webcam>`__

  * `Semantic Segmentation with OpenVINO™ using Segmenter <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/segmenter-semantic-segmentation>`__
  * `Programming Language Classification with OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/code-language-id>`__
  * `Stable Diffusion Text-to-Image Demo <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/stable-diffusion-v2/stable-diffusion-v2-text-to-image-demo.ipynb>`__

    * See alternative: `Stable Diffusion v2.1 using Optimum-Intel OpenVINO and multiple Intel Hardware <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/stable-diffusion-v2/stable-diffusion-v2-optimum-demo.ipynb>`__

  * `Text-to-Image Generation with Stable Diffusion v2 and OpenVINO™ <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/stable-diffusion-v2/stable-diffusion-v2-text-to-image.ipynb>`__

    * See alternative: `Stable Diffusion v2.1 using Optimum-Intel OpenVINO and multiple Intel Hardware <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/stable-diffusion-v2/stable-diffusion-v2-optimum-demo.ipynb>`__

  * `Image generation with Segmind Stable Diffusion 1B (SSD-1B) model and OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/stable-diffusion-xl/ssd-b1.ipynb>`__
  * `Data Preparation for 2D Medical Imaging <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/ct-segmentation-quantize/data-preparation-ct-scan.ipynb>`__
  * `Train a Kidney Segmentation Model with MONAI and PyTorch Lightning <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/ct-segmentation-quantize/pytorch-monai-training.ipynb>`__
  * `Live Inference and Benchmark CT-scan Data with OpenVINO™ <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/ct-segmentation-quantize/ct-scan-live-inference.ipynb>`__

    * See alternative: `Quantize a Segmentation Model and Show Live Inference <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/ct-segmentation-quantize/ct-segmentation-quantize-nncf.ipynb>`__

  * `Live Style Transfer with OpenVINO™ <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/style-transfer-webcam>`__



Legal Information
+++++++++++++++++++++++++++++++++++++++++++++

You may not use or facilitate the use of this document in connection with any infringement
or other legal analysis concerning Intel products described herein.

You agree to grant Intel a non-exclusive, royalty-free license to any patent claim
thereafter drafted which includes subject matter disclosed herein.

No license (express or implied, by estoppel or otherwise) to any intellectual property
rights is granted by this document.

All information provided here is subject to change without notice. Contact your Intel
representative to obtain the latest Intel product specifications and roadmaps.

The products described may contain design defects or errors known as errata which may
cause the product to deviate from published specifications. Current characterized errata
are available on request.

Intel technologies' features and benefits depend on system configuration and may require
enabled hardware, software or service activation. Learn more at
`http://www.intel.com/ <http://www.intel.com/>`__
or from the OEM or retailer.

No computer system can be absolutely secure.

Intel, Atom, Arria, Core, Movidius, Xeon, OpenVINO, and the Intel logo are trademarks
of Intel Corporation in the U.S. and/or other countries.

OpenCL and the OpenCL logo are trademarks of Apple Inc. used by permission by Khronos

Other names and brands may be claimed as the property of others.

Copyright © 2024, Intel Corporation. All rights reserved.

For more complete information about compiler optimizations, see our Optimization Notice.

Performance varies by use, configuration and other factors. Learn more at
`www.Intel.com/PerformanceIndex <www.Intel.com/PerformanceIndex>`__.





