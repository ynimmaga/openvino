Property addon
===================

.. meta::
   :description: Explore the modules of openvino-node in Node.js API and their implementation
                 in Intel® Distribution of OpenVINO™ Toolkit.

.. toctree::
   :maxdepth: 3
   :hidden:

   element <./openvino-node/enums/element>
   resizeAlgorithm <./openvino-node/enums/resizeAlgorithm>
   CompiledModel <./openvino-node/interfaces/CompiledModel>
   Core <./openvino-node/interfaces/Core>
   CoreConstructor <./openvino-node/interfaces/CoreConstructor>
   InferRequest <./openvino-node/interfaces/InferRequest>
   InputInfo <./openvino-node/interfaces/InputInfo>
   InputModelInfo <./openvino-node/interfaces/InputModelInfo>
   InputTensorInfo <./openvino-node/interfaces/InputTensorInfo>
   Model <./openvino-node/interfaces/Model>
   Output <./openvino-node/interfaces/Output>
   OutputInfo <./openvino-node/interfaces/OutputInfo>
   OutputTensorInfo <./openvino-node/interfaces/OutputTensorInfo>
   PartialShape <./openvino-node/interfaces/PartialShape>
   PartialShapeConstructor <./openvino-node/interfaces/PartialShapeConstructor>
   PrePostProcessor <./openvino-node/interfaces/PrePostProcessor>
   PrePostProcessorConstructor <./openvino-node/interfaces/PrePostProcessorConstructor>
   PreProcessSteps <./openvino-node/interfaces/PreProcessSteps>
   Tensor <./openvino-node/interfaces/Tensor>
   TensorConstructor <./openvino-node/interfaces/TensorConstructor>


The **openvino-node** package exports ``addon`` which contains the following properties:

.. rubric:: Interface NodeAddon

.. code-block:: ts

   interface NodeAddon {
       Core: CoreConstructor;
       PartialShape: PartialShapeConstructor;
       Tensor: TensorConstructor;
       element: typeof element;
       preprocess: {
           PrePostProcessor: PrePostProcessorConstructor;
           resizeAlgorithm: typeof resizeAlgorithm;
       };
   }

* **Defined in:**
  `addon.ts:192 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L192>`__


Properties
#####################


.. rubric:: Core

.. container:: m-4

   .. code-block:: ts

      Core: CoreConstructor

   * **Type declaration:**

     - CoreConstructor: :doc:`CoreConstructor <./openvino-node/interfaces/CoreConstructor>`

   -  **Defined in:**
      `addon.ts:193 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L193>`__


.. rubric:: PartialShape

.. container:: m-4

   .. code-block:: ts

      PartialShape: PartialShapeConstructor

   * **Type declaration:**

     - PartialShapeConstructor: :doc:`PartialShapeConstructor <./openvino-node/interfaces/PartialShapeConstructor>`

   -  **Defined in:**
      `addon.ts:195 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L195>`__

.. rubric:: Tensor

.. container:: m-4

   .. code-block:: ts

      Tensor: TensorConstructor

   * **Type declaration:**

     - TensorConstructor: :doc:`TensorConstructor <./openvino-node/interfaces/TensorConstructor>`

   -  **Defined in:**
      `addon.ts:194 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L194>`__


.. rubric:: element

.. container:: m-4

   .. code-block:: ts

      element: typeof element

   * **Type declaration:**

     - element: typeof :doc:`element <./openvino-node/enums/element>`

   -  **Defined in:**
      `addon.ts:201 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L201>`__


.. rubric:: preprocess

.. container:: m-4

   .. code-block:: ts

      preprocess: {
          PrePostProcessor: PrePostProcessorConstructor;
          resizeAlgorithm: typeof resizeAlgorithm;
      }

   * **Type declaration:**

     - PrePostProcessor: :doc:`PrePostProcessorConstructor <./openvino-node/interfaces/PrePostProcessorConstructor>`
     - resizeAlgorithm: typeof :doc:`resizeAlgorithm <./openvino-node/enums/resizeAlgorithm>`

   -  **Defined in:**
      `addon.ts:169 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L169>`__

