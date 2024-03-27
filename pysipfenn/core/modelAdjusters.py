import os
from typing import Union, Literal

import numpy as np
import torch
import plotly.express as px
from pysipfenn.core.pysipfenn import Calculator

class LocalAdjuster:
    """
    Adjuster class taking a ``Calculator`` and operating on local data provided to model as a pair of descriptor data
    (provided in several ways) and target values (provided in several ways). It can then adjust the model with some predefined
    hyperparameters or run a fairly typical grid search, which can be interpreted manually or uploaded to the ClearML
    platform. Can use CPU, CUDA, or MPS (Mac M1) devices for training.

    Args:
        calculator: Instance of the ``Calculator`` class with the model to be adjusted, defined and loaded. It can
            contain the descriptor data already in it, so that it does not have to be provided separately.
        model: Name of the model to be adjusted in the ``Calculator``. E.g., ``SIPFENN_Krajewski2022_NN30``.
        targetData: Target data to be used for training the model. It can be provided as a path to a NumPy ``.npy``/
            ``.NPY`` or CSV ``.csv``/``.CSV`` file, or directly as a NumPy array. It has to be the same length as the
            descriptor data.
        descriptorData: Descriptor data to be used for training the model. It can be left unspecified (``None``) to
            use the data in the ``Calculator``, or provided as a path to a NumPy ``.npy``/``.NPY`` or CSV ``.csv``/
            ``.CSV`` file, or directly as a NumPy array. It has to be the same length as the target data. Default is
            ``None``.
        device: Device to be used for training the model. It Has to be one of the following: ``"cpu"``, ``"cuda"``, or
            ``"mps"``. Default is ``"cpu"``.
        descriptor: Name of the feature vector provided in the descriptorData. It can be optionally provided to
            check if the descriptor data is compatible.

    Attributes:
        calculator: Instance of the ``Calculator`` class being operated on.
        model: The original model to be adjusted.
        adjustedModel: A PyTorch model after the adjustment. Initially set to ``None``.
        descriptorData: NumPy array with descriptor data to use as input for the model.
        targetData: NumPy array with target data to use as output for the model.
    """

    def __init__(
            self,
            calculator: Calculator,
            model: str,
            targetData: Union[str, np.ndarray],
            descriptorData: Union[None, str, np.ndarray] = None,
            device: Literal["cpu", "cuda", "mps"] = "cpu",
            descriptor: Literal["Ward2017", "KS2022"] = None
    ) -> None:
        self.adjustedModel = None

        assert isinstance(calculator, Calculator), "The calculator must be an instance of the Calculator class."
        self.calculator = calculator

        self.device = torch.device(device)

        assert isinstance(model, str), "The model must be a string pointing to the model to be adjusted in the Calculator."
        assert model in self.calculator.models, "The model must be one of the models in the Calculator."
        assert model in self.calculator.loadedModels, "The model must be loaded in the Calculator."
        self.model = self.calculator.loadedModels[model]
        self.model = self.model.to(device=self.device)

        if descriptorData is None:
            assert self.calculator.descriptorData is not None, "The descriptor data can be inferred from the data in the Calculator, but no data is present."
            self.descriptorData = self.calculator.descriptorData
        elif isinstance(descriptorData, np.ndarray):
            self.descriptorData = descriptorData
        elif isinstance(descriptorData, str):
            # Path to NPY file with data
            if (descriptorData.endswith(".npy") or descriptorData.endswith(".NPY")) and os.path.exists(descriptorData):
                self.descriptorData = np.load(descriptorData)
            # Path to CSV file with data
            elif (descriptorData.endswith(".csv") or descriptorData.endswith(".CSV")) and os.path.exists(descriptorData):
                self.descriptorData = np.loadtxt(descriptorData, delimiter=",")
            else:
                raise ValueError("If a string is provided as descriptor data parameter, it must be a path to a npy/NPY or csv/CSV file.")
        else:
            raise ValueError("The descriptor data must be either (a) None to use the data in the Calculator,"
                             "(b) a path to a npy/NPY file, or (c) a path to a csv/CSV file.")

        if isinstance(targetData, np.ndarray):
            self.targetData = targetData
        elif isinstance(targetData, str):
            # Path to NPY file with data
            if (targetData.endswith(".npy") or targetData.endswith(".NPY")) and os.path.exists(targetData):
                self.targetData = np.load(targetData)
            # Path to CSV file with data
            elif (targetData.endswith(".csv") or targetData.endswith(".CSV")) and os.path.exists(targetData):
                self.targetData = np.loadtxt(targetData, delimiter=",")
            else:
                raise ValueError("If a string is provided as target data parameter, it must be a path to a npy/NPY or csv/CSV file.")
        else:
            raise ValueError("The target data must be either a path to a npy/NPY file or a path to a csv/CSV file.")

        assert len(self.descriptorData) == len(self.targetData), "The descriptor and target data must have the same length."

        if descriptor is not None:
            if descriptor == "Ward2017":
                assert self.descriptorData.shape[1] == 271, "The descriptor must have 271 features for the Ward2017 descriptor."
            elif descriptor == "KS2022":
                assert self.descriptorData.shape[1] == 256, "The descriptor must have 256 features for the KS2022 descriptor."
            else:
                raise NotImplementedError("The descriptor must be either 'Ward2017' or 'KS2022'. Others will be added in the future.")

    def plotStarting(self) -> None:
        """
        Plot the starting model (before adjustment) on the target data.
        """
        self.model.eval()
        with torch.no_grad():
            dataIn = torch.from_numpy(np.array(self.descriptorData)).to(device=self.device).float()
            predictions = self.model(dataIn, None).detach().cpu().numpy().flatten()
        fig = px.scatter(
            x=self.targetData.flatten(),
            y=predictions,
            labels={
                "x": "Target Data", "y": "Predictions"},
            title="Starting (Unadjusted) Model Predictions")
        fig.show()

    def plotAdjusted(self) -> None:
        """
        Plot the adjusted model on the target data.
        """
        assert self.adjustedModel is not None, "The model must be adjusted before plotting. It is currently None."
        self.adjustedModel.eval()
        with torch.no_grad():
            dataIn = torch.from_numpy(np.array(self.descriptorData)).to(device=self.device).float()
            predictions = self.adjustedModel(dataIn, None).detach().cpu().numpy().flatten()
        fig = px.scatter(
            x=self.targetData.flatten(),
            y=predictions,
            labels={
                "x": "Target Data", "y": "Predictions"},
            title="Adjusted Model Predictions")
        fig.show()


class OPTIMADEAdjuster(LocalAdjuster):
    """
    Adjuster class operating on data provided by the OPTIMADE API. Primarily geared towards tuning or retraining of the
    models based on other atomistic databases, or their subsets, accessed through OPTIMADE, to adjust the model to a
    different domain, which in the context of DFT datasets could mean adjusting the model to predict properties with DFT
    settings used by that database or focusing its attention to specific chemistry like, for instance, all compounds of
    Sn and all perovskites. It accepts OPTIMADE query as an input and then operates based on the ``LocalAdjuster`` class.
    """
