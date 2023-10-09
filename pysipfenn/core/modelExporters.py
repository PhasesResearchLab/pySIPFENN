from pysipfenn import Calculator
import torch
import onnx
import io
from tqdm import tqdm

try:
    import coremltools as ct
    from onnxconverter_common import float16
    from onnxsim import simplify
except ModuleNotFoundError as e:
    print(f'Could not import {e.name}.\n')
    print('Dependencies for exporting to CoreML, Torch, and ONNX are not installed by default with pySIPFENN. You need '
          'to install pySIPFENN in "dev" mode like: pip install -e "pysipfenn[dev]", or like pip install -e ".[dev]" if'
          'you are cloned it. See pysipfenn.org for more details.')


class ONNXExporter:
    """Export models to the ONNX format (what they ship in by default) to allow (1) exporting modified pySIPFENN models,
    (2) simplify the models using ONNX optimizer, and (3) convert them to FP16 precision, cutting the size in half.

    Args:
        calculator: A calculator object with loaded models that has loaded PyTorch models (happens automatically
        when the autoLoad argument is kept to its default value of True when initializing the Calculator). During the
        initialization, the loaded PyTorch models are converted back to ONNX (in memory) to be then either adjusted or
        persisted to disk.

    Attributes:
        calculator: A calculator object with ONNX loaded models.
        simplifiedDict: A dictionary of models that have been simplified.
        fp16Dict: A dictionary of models that have been converted to FP16.
    """

    def __init__(self, calculator: Calculator):
        """Initialize the ONNXExporter using a calculator object."""
        self.simplifiedDict = {model: False for model in calculator.loadedModels.keys()}
        self.fp16Dict = {model: False for model in calculator.loadedModels.keys()}
        self.calculator = calculator
        assert len(self.calculator.loadedModels) > 0, 'No models loaded in calculator. Nothing to export.'
        print(f'Initialized ONNXExporter with PyTorch models: '
              f'{list(self.calculator.loadedModels.keys())}'
              f'\n Converting to ONNX models...')

        for model in calculator.loadedModels:
            print(f'Converting {model} to ONNX')
            assert 'descriptor' in self.calculator.models[model], f'{model} does not have a descriptor. Cannot export.'
            descriptorUsed = self.calculator.models[model]['descriptor']
            if descriptorUsed == 'Ward2017':
                dLen = 271
            elif descriptorUsed == 'KS2022':
                dLen = 256
            else:
                raise NotImplementedError(f'ONNX export for {descriptorUsed} not implemented yet.')

            assert model in self.calculator.loadedModels, f'{model} not loaded in calculator. Nothing to export.'
            loadedModel = self.calculator.loadedModels[model]
            loadedModel.eval()

            inputs_tracer = torch.zeros(dLen, )
            if 'OnnxDropoutDynamic()' in {str(module) for module in list(loadedModel._modules.values())}:
                inputs_tracer = (inputs_tracer, torch.zeros(1, ))

            temp = io.BytesIO()
            torch.onnx.export(
                loadedModel,
                inputs_tracer,
                temp,
                export_params=True,
                opset_version=16,
                do_constant_folding=True,
                input_names=[descriptorUsed],
                output_names=['property'],
            )
            temp.seek(0)
            self.calculator.loadedModels.update({
                model: onnx.load(temp)
            })
            del temp
        print(f'Initialized ONNXExporter with models: {list(self.calculator.loadedModels.keys())}')

    def simplify(self, model: str) -> None:
        """Simplify a loaded model using the ONNX optimizer.

        Args:
            model: The name of the model to simplify (must be loaded in the Calculator).

        Returns:
            None
        """
        print(f'Simplifying {model}')
        assert model in self.calculator.loadedModels, f'{model} not loaded in calculator. Nothing to simplify.'
        loadedModel = self.calculator.loadedModels[model]
        onnx_model_simp, check = simplify(loadedModel)
        assert check, "Simplified ONNX model could not be validated"
        self.calculator.loadedModels[model] = onnx_model_simp
        self.simplifiedDict[model] = True
        print(f'--> Simplified {model}', flush=True)

    def simplifyAll(self):
        """Simplify all loaded models with the simplify function."""
        for model in tqdm(self.calculator.loadedModels):
            self.simplify(model)
        print('*****  Done simplifying all models!  *****')

    def toFP16(self, model: str):
        """Convert a loaded model to FP16 precision.

        Args:
            model: The name of the model to convert to FP16 (must be loaded in the Calculator).

        Returns:
            None
        """
        print(f'Converting {model} to FP16')
        assert model in self.calculator.loadedModels, f'{model} not loaded in calculator. Nothing to convert to FP16.'
        loadedModel = self.calculator.loadedModels[model]
        # Convert to FP16
        onnx_model_fp16 = float16.convert_float_to_float16(loadedModel)
        self.calculator.loadedModels[model] = onnx_model_fp16
        self.fp16Dict[model] = True
        print(f'--> Converted {model} to FP16', flush=True)

    def toFP16All(self):
        """Convert all loaded models to FP16 precision with the toFP16 function."""
        for model in tqdm(self.calculator.loadedModels):
            self.toFP16(model)
        print('*****  Done converting all models to FP16!  *****')

    def export(self, model: str):
        """Export a loaded model to ONNX format.

        Args:
            model: The name of the model to export (must be loaded in the Calculator).

        Returns:
            None
        """
        print(f'Exporting {model} to ONNX')
        assert model in self.calculator.loadedModels, f'{model} not loaded in calculator. Nothing to export.'
        loadedModel = self.calculator.loadedModels[model]
        name = f"{model}"
        if self.simplifiedDict[model]:
            name += '_simplified'
        if self.fp16Dict[model]:
            name += '_fp16'
        name += '.onnx'
        onnx.save(loadedModel, name)
        print(f'--> Exported as {name}', flush=True)

    def exportAll(self):
        """Export all loaded models to ONNX format with the export function."""
        for model in tqdm(self.calculator.loadedModels):
            self.export(model)
        print('*****  Done exporting all models!  *****')


class TorchExporter:
    """Export models to the PyTorch PT format to allow for easy loading and inference in PyTorch in other projects.

    Args:
        calculator: A calculator object with loaded models.

    Attributes:
        calculator: A calculator object with loaded models.
    """
    def __init__(self, calculator: Calculator):
        """Initialize the TorchExporter with a calculator object that has loaded models."""
        self.calculator = calculator
        assert len(self.calculator.loadedModels) > 0, 'No models loaded in calculator. Nothing to export.'
        print(f'Initialized TorchExporter with models: {list(self.calculator.loadedModels.keys())}')

    def export(self, model: str):
        """Export a loaded model to PyTorch PT format. Models are exported in eval mode (no dropout) and saved in the
        current working directory.

        Args:
            model: The name of the model to export (must be loaded in the Calculator) and it must have a descriptor
                (Ward2017 or KS2022) defined in the calculator.models dictionary created when the Calculator was
                initialized.

        Returns:
            None
        """
        print(f'Exporting {model} to PyTorch PT format')

        assert model in self.calculator.loadedModels, f'{model} not loaded in calculator. Nothing to export.'
        loadedModel = self.calculator.loadedModels[model]

        assert 'descriptor' in self.calculator.models[model], f'{model} does not have a descriptor. Cannot export.'
        descriptorUsed = self.calculator.models[model]['descriptor']
        if descriptorUsed == 'Ward2017':
            dLen = 271
        elif descriptorUsed == 'KS2022':
            dLen = 256
        else:
            raise NotImplementedError(f'TorchExporter export for {descriptorUsed} not implemented yet.')

        loadedModel.eval()
        inputs_tracer = torch.zeros(dLen, )
        if 'OnnxDropoutDynamic()' in {str(module) for module in list(loadedModel._modules.values())}:
            inputs_tracer = (inputs_tracer, torch.zeros(1, ))

        tracedModel = torch.jit.trace(loadedModel, inputs_tracer)

        name = f"{model}.pt"
        tracedModel.save(name)
        print(f'--> Exported as {name}', flush=True)

    def exportAll(self):
        """Export all loaded models to PyTorch PT format with the export function."""
        for model in tqdm(self.calculator.loadedModels):
            self.export(model)
        print('*****  Done exporting all models!  *****')


class CoreMLExporter:
    """Export models to the CoreML format to allow for easy loading and inference in CoreML in other projects,
    particularly valuable for Apple devices, as pySIPFENN models can be run using the Neural Engine accelerator
    with minimal power consumption and neat optimizations.

    Args:
        calculator: A calculator object with loaded models.

    Attributes:
        calculator: A calculator object with loaded models.
    """
    def __init__(self, calculator: Calculator):
        self.calculator = calculator
        assert len(self.calculator.loadedModels)>0, 'No models loaded in calculator. Nothing to export.'
        print(f'Initialized CoreMLExporter with models: {list(self.calculator.loadedModels.keys())}')

    def export(self, model: str):
        """Export a loaded model to CoreML format. Models will be saved as {model}.mlpackage in the current working
        directory. Models will be annotated with the feature vector name (Ward2017 or KS2022) and the output will be
        named "property". The latter behavior will be adjusted in the future when model output name and unit will be
        added to the model JSON metadata.

        Args:
            model: The name of the model to export (must be loaded in the Calculator) and it must have a descriptor
                (Ward2017 or KS2022) defined in the calculator.models dictionary created when the Calculator was
                initialized.

        Returns:
            None
        """
        print(f'Exporting {model} to CoreML')
        assert model in self.calculator.loadedModels, f'{model} not loaded in calculator. Nothing to export.'
        loadedModel = self.calculator.loadedModels[model]
        assert 'descriptor' in self.calculator.models[model], f'{model} does not have a descriptor. Cannot export.'
        descriptorUsed = self.calculator.models[model]['descriptor']
        if descriptorUsed == 'Ward2017':
            dLen = 271
        elif descriptorUsed == 'KS2022':
            dLen = 256
        else:
            raise NotImplementedError(f'CoreML export for {descriptorUsed} not implemented yet.')

        loadedModel.eval()

        inputs_converter = [ct.TensorType(name=descriptorUsed, shape=(dLen,))]
        inputs_tracer = torch.zeros(dLen,)

        if 'OnnxDropoutDynamic()' in {str(module) for module in list(loadedModel._modules.values())}:
            inputs_tracer = (inputs_tracer, torch.zeros(1,))
            inputs_converter.append(ct.TensorType(name='DropoutMode', shape=(1,)))

        tracedModel = torch.jit.trace(loadedModel, inputs_tracer)

        coreml_model = ct.convert(
            model=tracedModel,
            convert_to='mlprogram',
            inputs=inputs_converter,
            outputs=[ct.TensorType(name='property')]
        )
        name = f"{model}.mlpackage"
        coreml_model.save(name)
        print(f'--> Exported as {name}', flush=True)

    def exportAll(self):
        """Export all loaded models to CoreML format with the export function."""
        for model in tqdm(self.calculator.loadedModels):
            self.export(model)
        print('*****  Done exporting all models!  *****')