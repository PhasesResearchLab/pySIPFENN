from pysipfenn import Calculator
import torch
import coremltools as ct
from onnxconverter_common import float16
from onnxsim import simplify
import onnx
from tqdm import tqdm
from importlib import resources


class ONNXExporter:
    def __init__(self, calculator: Calculator):
        self.simplifiedDict = {}
        self.fp16Dict = {}
        self.calculator = calculator
        if len(self.calculator.loadedModels) == 0:
            print(f'No models loaded in calculator. '
                  f'Reloading models into ONNX: {self.calculator.network_list_available}')
            with resources.files('pysipfenn.modelsSIPFENN') as modelPath:
                for net in tqdm(self.calculator.network_list_available):
                    self.calculator.loadedModels.update({
                        net: onnx.load(f'{modelPath}/{net}.onnx')
                    })
        print(f'Initialized ONNXExporter with models: {list(self.calculator.loadedModels.keys())}')

    def simplify(self, model: str):
        print(f'Simplifying {model}')
        loadedModel = self.calculator.loadedModels[model]
        # Simplify
        onnx_model_simp, check = simplify(loadedModel)
        assert check, "Simplified ONNX model could not be validated"
        self.calculator.loadedModels[model] = onnx_model_simp
        self.simplifiedDict[model] = True
        print(f'--> Simplified {model}', flush=True)

    def simplifyAll(self):
        for model in tqdm(self.calculator.loadedModels):
            self.simplify(model)
        print('*****  Done simplifying all models!  *****')

    def toFP16(self, model: str):
        print(f'Converting {model} to FP16')
        loadedModel = self.calculator.loadedModels[model]
        # Convert to FP16
        onnx_model_fp16 = float16.convert_float_to_float16(loadedModel)
        self.calculator.loadedModels[model] = onnx_model_fp16
        self.fp16Dict[model] = True
        print(f'--> Converted {model} to FP16', flush=True)

    def toFP16All(self):
        for model in tqdm(self.calculator.loadedModels):
            self.toFP16(model)
        print('*****  Done converting all models to FP16!  *****')

    def export(self, model: str):
        print(f'Exporting {model} to ONNX')
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
        for model in tqdm(self.calculator.loadedModels):
            self.export(model)
        print('*****  Done exporting all models!  *****')