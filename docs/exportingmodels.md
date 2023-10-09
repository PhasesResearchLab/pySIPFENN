# Exporting pySIPFENN Models

## General Information

Whether you are the end-user of ML models and you want to integrate them into some other software that requires a
specific format, or you are a developer who is modifying pySIPFENN models to predict formation energy resulting from 
some particular DFT settings/pseudopotentials, or you retrain them on entirely different properties, you will probably need to export
the models at some point. This page provides information on how to do that using the 3 exporter classes we provide
in the `pysipfenn/core/modelExporters.py` module. Their API is very high-level and simple to use, so the process 
should be straightforward.

Please be aware that to use these exporters, you will need to install additional dependencies by installing the `dev`
extras with

    pip install "pysipfenn[dev]"

or for development installation

    pip install ".[dev]"


## ONNXExporter

The `ONNXExporter` class is used to export pySIPFENN models to the [ONNX](https://onnx.ai/) format. This is also the
same format they are shipped in by us! The reason we implemented this exporter is that:
1. Within pySIPFENN, models are actually stored as PyTorch models, and if they are modified, they need to be converted to 
ONNX format again. 
2. This exporter allows you to export models in different precision (float16) using its `toFP16` method. This is useful
if you want to use the models on devices with limited memory, such as mobile phones or embedded devices.
3. This exporter also allows you to simplify the model through the recent ONNX Optimizer package implementation, which 
could improve model performance and reduce its size.

To get started, you should take an initialized Calculator object and pass it to the exporter.

    from pysipfenn import ONNXExporter
    from pysipfenn import Calculator
    
    c = Calculator()
    onnx_exporter = ONNXExporter(c)

and then simply 

    onnx_exporter.export("MyModelNameGoesHere")

or to export all models in `c` at once

    onnx_exporter.exportAll()

and you should see new files like `MyModelNameGoesHere.onnx` in the current working directory. 

If you want to export the 
model in float16 precision, before you call `export` method, you should call

    onnx_exporter.toFP16("MyModelNameGoesHere")

or to convert all models in `c` at once

    onnx_exporter.toFP16All()

and similarly for simplification

    onnx_exporter.simplify("MyModelNameGoesHere")

or

    onnx_exporter.simplifyAll()

To summarize, if you want to export all models in `c` in float16 precision and simplified, you can do it with

    from pysipfenn import ONNXExporter
    from pysipfenn import Calculator
    
    c = Calculator()
    onnx_exporter = ONNXExporter(c)
    onnx_exporter.simplifyAll()
    onnx_exporter.toFP16All()
    onnx_exporter.exportAll()

and you should see new files like `MyModelNameGoesHere_simplified_fp16.onnx` in the current working directory.


## PyTorch

This is the simplest of the export methods because, as mentioned in [ONNXExporter](#onnxexporter) section, pySIPFENN 
models are already stored as PyTorch models; therefore, no conversion is needed. You can use it by simply calling 

    from pysipfenn import PyTorchExporter
    from pysipfenn import Calculator
    
    c = Calculator()
    torch_exporter = PyTorchExporter(c)
    torch_exporter.export("MyModelNameGoesHere")

or to export all models in `c` at once, replace the last line with
    
    torch_exporter.exportAll()

and you should see new files like `MyModelNameGoesHere.pt` in the current working directory.


## CoreML

CoreML is a format developed by Apple for use in their devices, where it provides the most seamless integration with
existing apps and can harvest very efficient Neural Engine hardware acceleration. At the same time, it can be used on
other platforms as well, such as Linux or Windows, through [coremltools](https://coremltools.readme.io/docs) toolset
from Apple used by this exporter.

Note that under the hood, CoreML uses the float16 precision, so the model predictions will numerically match those
exported with [ONNXExporter](#onnxexporter) in float16 precision rather than the default pySIPFENN models. This can
be useful if you want to use the models on devices with limited memory, such as mobile phones or embedded devices, and 
generally should not significantly affect the accuracy of the predictions.

You can use it by simply calling 

    from pysipfenn import CoreMLExporter
    from pysipfenn import Calculator
    
    c = Calculator()
    coreml_exporter = CoreMLExporter(c)
    coreml_exporter.export("MyModelNameGoesHere")

or to export all models in `c` at once, replace the last line with

    coreml_exporter.exportAll()

and you should see new files like `MyModelNameGoesHere.mlpackage` in the current working directory. If you are on MacOS and
have XCode installed, you can double-click on it to evaluate its integrity and benchmark its performance.

