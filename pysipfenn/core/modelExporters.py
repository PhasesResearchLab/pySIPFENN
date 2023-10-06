from pysipfenn import Calculator
import torch
import onnx
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
        calculator: A calculator object with loaded models.

    Attributes:
        calculator: A calculator object with loaded models.
        simplifiedDict: A dictionary of models that have been simplified.
        fp16Dict: A dictionary of models that have been converted to FP16.
    """

    def __init__(self, calculator: Calculator):
        """Initialize the ONNXExporter with a calculator object that has loaded models."""
        self.simplifiedDict = {}
        self.fp16Dict = {}
        self.calculator = calculator
        assert len(self.calculator.loadedModels) > 0, 'No models loaded in calculator. Nothing to export.'
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
        """Export a loaded model to PyTorch PT format

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
    def __init__(self, calculator: Calculator):
        self.calculator = calculator
        assert len(self.calculator.loadedModels)>0, 'No models loaded in calculator. Nothing to export.'
        print(f'Initialized CoreMLExporter with models: {list(self.calculator.loadedModels.keys())}')

    def export(self, model: str):
        print(f'Exporting {model} to CoreML')
        loadedModel = self.calculator.loadedModels[model]

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
            outputs=[ct.TensorType(name='Ef_eV')]
        )
        name = f"{model}.mlpackage"
        coreml_model.save(name)
        print(f'--> Exported as {name}', flush=True)

    def exportAll(self):
        for model in tqdm(self.calculator.loadedModels):
            self.export(model)
        print('*****  Done exporting all models!  *****')