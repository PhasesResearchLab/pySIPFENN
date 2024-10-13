# Standard library imports
import os
from typing import Union, Literal, Tuple, List, Dict
from copy import deepcopy
import gc
from functools import reduce
import operator
from random import shuffle
import time

# Default 3rd party imports
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from pysipfenn.core.pysipfenn import Calculator
from pymatgen.core import Structure, Composition

import plotly.express as px
import plotly.graph_objects as go


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
        useClearML: Whether to use the ClearML platform for logging the training process. Default is ``False``.
        taskName: Name of the task to be used. Default is ``"LocalFineTuning"``.

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
            descriptor: Literal["Ward2017", "KS2022"] = None,
            useClearML: bool = False,
            taskName: str = "LocalFineTuning"
    ) -> None:
        self.adjustedModel = None
        self.useClearML = useClearML
        self.useClearMLMessageDisplayed = False
        self.taskName = taskName

        assert isinstance(calculator, Calculator), "The calculator must be an instance of the Calculator class."
        self.calculator = calculator

        self.device = torch.device(device)

        assert isinstance(model, str), "The model must be a string pointing to the model to be adjusted in the Calculator."
        assert model in self.calculator.models, "The model must be one of the models in the Calculator."
        assert model in self.calculator.loadedModels, "The model must be loaded in the Calculator."
        self.modelName = model
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
                self.descriptorData = np.loadtxt(descriptorData, delimiter=",", skiprows=1)[:, 1:]
            else:
                raise ValueError("If a string is provided as descriptor data parameter, it must be a path to a npy/NPY or csv/CSV file.")
        else:
            print(descriptorData)
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
                # Skip the first row if it is a header
                self.targetData = np.loadtxt(targetData, delimiter=",", skiprows=1)[:, 1:]
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

        self.comps: List[str] = []
        self.names: List[str] = []
        self.validationLabels: List[str] = []

        print("Initialized Adjuster instance!\n")

    def plotStarting(self) -> None:
        """
        Plot the starting model (before adjustment) on the target data. By default, it will plot in your browser.
        """
        reference = self.targetData.flatten()
        assert len(reference) == len(self.descriptorData), "The target data and descriptor data must have the same length."
        assert len(reference) != 0, "The target data must not be empty for plotting."
        self.model.eval()
        print("Running the STARTING model on the data and plotting the results...")
        with torch.no_grad():
            dataIn = torch.from_numpy(np.array(self.descriptorData)).float().to(device=self.device)
            predictions = self.model(dataIn, None).detach().cpu().numpy().flatten()
        minVal = min(min(reference), min(predictions))
        maxVal = max(max(reference), max(predictions))

        if self.names:
            fig = px.scatter(
                x=reference,
                y=predictions,
                hover_name=self.names,
                labels={"x": "Target Data", "y": "Predictions"},
                title="Starting (Unadjusted) Model Predictions (Hover for Name)"
                )
        else:
            fig = px.scatter(
                x=reference,
                y=predictions,
                labels = {"x": "Target Data", "y": "Predictions"},
                title = "Starting (Unadjusted) Model Predictions"
            )
        # If the validation labels are set, color the points as blue for training, green for validation, and red for
        # any other label, just in case advanced users want to use this method for other purposes.
        if self.validationLabels:
            print("Overlaying the training and validation labels on the plot.")
            fig.update_traces(
                marker=dict(
                    color=[(
                        "blue" if label == "Training" else
                        "green" if label == "Validation" else
                        "red") for label in self.validationLabels],
                    symbol='circle-dot',
                    opacity=0.5,
                    size=12
                )
            )
        else:
            fig.update_traces(
                marker=dict(
                    symbol='circle-dot',
                    opacity=0.5,
                    size=12
                )
            )
        fig.add_trace(
            go.Scatter(
                x=[minVal, maxVal],
                y=[minVal, maxVal],
                mode='lines',
                line=dict(color='gray'),
                name='x=y'
            )
        )
        fig.show()

    def plotAdjusted(self) -> None:
        """
        Plot the adjusted model on the target data. By default, it will plot in your browser.
        """
        assert self.adjustedModel is not None, "The model must be adjusted before plotting. It is currently None."
        self.adjustedModel.eval()
        reference = self.targetData.flatten()
        assert len(reference) == len(self.descriptorData), "The target data and descriptor data must have the same length."
        assert len(reference) != 0, "The target data must not be empty for plotting."
        print("Running the ADJUSTED model on the data and plotting the results...")
        with torch.no_grad():
            dataIn = torch.from_numpy(np.array(self.descriptorData)).float().to(device=self.device)
            predictions = self.adjustedModel(dataIn, None).detach().cpu().numpy().flatten()
        minVal = min(min(reference), min(predictions))
        maxVal = max(max(reference), max(predictions))

        if self.names:
            fig = px.scatter(
                x=reference,
                y=predictions,
                hover_name=self.names,
                labels={"x": "Target Data", "y": "Predictions"},
                title="Adjusted Model Predictions (Hover for Name)"
                )
        else:
            fig = px.scatter(
                x=reference,
                y=predictions,
                labels = {"x": "Target Data", "y": "Predictions"},
                title = "Adjusted Model Predictions"
            )
        # If the validation labels are set, color the points as blue for training, green for validation, and red for
        # any other label, just in case advanced users want to use this method for other purposes.
        if self.validationLabels:
            print("Overlaying the training and validation labels on the plot.")
            fig.update_traces(
                marker=dict(
                    color=[(
                        "blue" if label == "Training" else
                        "green" if label == "Validation" else
                        "red") for label in self.validationLabels],
                    symbol='circle-dot',
                    opacity=0.5,
                    size=12
                )
            )
        else:
            fig.update_traces(
                marker=dict(
                    symbol='circle-dot',
                    opacity=0.5,
                    size=12
                )
            )
        fig.add_trace(
            go.Scatter(
                x=[minVal, maxVal],
                y=[minVal, maxVal],
                mode='lines',
                line=dict(color='gray'),
                name='x=y'
            )
        )
        fig.show()

    def adjust(
            self,
            validation: float = 0.2,
            learningRate: float = 1e-5,
            epochs: int = 50,
            batchSize: int = 32,
            optimizer: Literal["Adam", "AdamW", "Adamax", "RMSprop"] = "Adam",
            weightDecay: float = 1e-5,
            lossFunction: Literal["MSE", "MAE"] = "MAE",
            verbose: bool = True
    ) -> Tuple[torch.nn.Module, List[float], List[float]]:
        """
        Takes the original model, copies it, and adjusts the model on the provided data. The adjusted model is stored in
        the ``adjustedModel`` attribute of the class and can be then persisted to the original ``Calculator`` or used
        for plotting. The default hyperparameters are selected for fine-tuning the model rather than retraining it, as
        to slowly adjust it (1% of the typical learning rate) and not overfit it (50 epochs).

        Args:
            learningRate: The learning rate to be used for the adjustment. Default is ``1e-5`` that is 1% of a typical
                learning rate of ``Adam`` optimizer.
            epochs: The number of times to iterate over the data, i.e., how many times the model will see the data.
                Default is ``50``, which is on the higher side for fine-tuning. If the model does not retrain fast enough
                but already converged, consider lowering this number to reduce the time and possibly overfitting to the
                training data.
            batchSize: The number of points passed to the model at once. Default is ``32``, which is a typical batch size for
                smaller datasets. If the dataset is large, consider increasing this number to speed up the training.
            optimizer: Algorithm to be used for optimization. Default is ``Adam``, which is a good choice for most models
                and one of the most popular optimizers. Other options are
            lossFunction: Loss function to be used for optimization. Default is ``MAE`` (Mean Absolute Error / L1) that is
                more robust to outliers than ``MSE`` (Mean Squared Error).
            validation: Fraction of the data to be used for validation. Default is the common ``0.2`` (20% of the data).
                If set to ``0``, the model will be trained on the whole dataset without validation, and you will not be able
                to check for overfitting or gauge the model's performance on unseen data.
            weightDecay: Weight decay to be used for optimization. Default is ``1e-5`` that should work well if data is
                abundant enough relative to the model complexity. If the model is overfitting, consider increasing this
                number to regularize the model more.
            verbose: Whether to print information, such as loss, during the training. Default is ``True``.

        Returns:
            A tuple with 3 elements: (1) the adjusted model, (2) training loss list of floats, and (3) validation loss
            list of floats. The adjusted model is also stored in the ``adjustedModel`` attribute of the class.
        """

        if verbose:
            print("Loading the data...")
        assert len(self.descriptorData) != 0, "The descriptor data must not be empty for the adjustment process."
        assert len(self.targetData) != 0, "The target data must not be empty for the adjustment process."
        assert len(self.descriptorData) == len(self.targetData), "The descriptor and target data must have the same length."

        ddTensor = torch.from_numpy(self.descriptorData).float().to(device=self.device)
        tdTensor = torch.from_numpy(self.targetData).float().to(device=self.device)
        if validation > 0:
            split = int(len(ddTensor) * (1 - validation))
            self.validationLabels = ["Training"]*split + ["Validation"]*(len(ddTensor)-split)
            ddTrain, ddVal = ddTensor[:split], ddTensor[split:]
            tdTrain, tdVal = tdTensor[:split], tdTensor[split:]
        else:
            self.validationLabels = ["Training"]*len(ddTensor)
            ddTrain, ddVal = ddTensor, None
            tdTrain, tdVal = tdTensor, None

        datasetTrain = TensorDataset(ddTrain, tdTrain)
        dataloaderTrain = DataLoader(datasetTrain, batch_size=batchSize, shuffle=True)

        if verbose:
            print(f'LR: {learningRate} |  Optimizer: {optimizer}  |  Weight Decay: {weightDecay} |  Loss: {lossFunction}')
        # Training a logging platform. Completely optional and does not affect the training.
        if self.useClearML:
            if verbose and not self.useClearMLMessageDisplayed:
                print("Using ClearML for logging. Make sure to have (1) their Python package installed and (2) the API key"
                      " set up according to their documentation. Otherwise you will get an error.")
                self.useClearMLMessageDisplayed = True
            from clearml import Task
            task = Task.create(project_name=self.taskName,
                               task_name=f'LR:{learningRate} OPT:{optimizer} WD:{weightDecay} LS:{lossFunction}')
            task.set_parameters({'lr': learningRate,
                                 'epochs': epochs,
                                 'batch_size': batchSize,
                                 'weight_decay': weightDecay,
                                 'loss': lossFunction,
                                 'optimizer': optimizer,
                                 'model': self.modelName})
        if verbose:
            print("Copying and initializing the model...")
        model = deepcopy(self.model)
        model.train()
        if verbose:
            print("Setting up the training...")
        if optimizer == "Adam":
            optimizerClass = torch.optim.Adam
        elif optimizer == "AdamW":
            optimizerClass = torch.optim.AdamW
        elif optimizer == "Adamax":
            optimizerClass = torch.optim.Adamax
        elif optimizer == "RMSprop":
            optimizerClass = torch.optim.RMSprop
        else:
            raise NotImplementedError("The optimizer must be one of the following: 'Adam', 'AdamW', 'Adamax', 'RMSprop'.")
        optimizerInstance = optimizerClass(model.parameters(), lr=learningRate, weight_decay=weightDecay)

        if lossFunction == "MSE":
            loss = torch.nn.MSELoss()
        elif lossFunction == "MAE":
            loss = torch.nn.L1Loss()
        else:
            raise NotImplementedError("The loss function must be one of the following: 'MSE', 'MAE'.")

        transferLosses = [float(loss(model(ddTrain, None), tdTrain))]
        if validation > 0:
            validationLosses = [float(loss(model(ddVal, None), tdVal))]
            if verbose:
                print(
                    f'Train: {transferLosses[-1]:.4f} | Validation: {validationLosses[-1]:.4f} | Epoch: 0/{epochs}')
        else:
            validationLosses = []
            if verbose:
                print(f'Train: {transferLosses[-1]:.4f} | Epoch: 0/{epochs}')

        for epoch in range(epochs):
            model.train()
            for data, target in dataloaderTrain:
                optimizerInstance.zero_grad()
                output = model(data, None)
                lossValue = loss(output, target)
                lossValue.backward()
                optimizerInstance.step()
            transferLosses.append(float(loss(model(ddTrain, None), tdTrain)))

            if validation > 0:
                model.eval()
                validationLosses.append(float(loss(model(ddVal, None), tdVal)))
                model.train()
                if self.useClearML:
                    task.get_logger().report_scalar(
                        title='Loss',
                        series='Validation',
                        value=validationLosses[-1],
                        iteration=epoch+1)
                if verbose:
                    print(
                        f'Train: {transferLosses[-1]:.4f} | Validation: {validationLosses[-1]:.4f} | Epoch: {epoch + 1}/{epochs}')
            else:
                if verbose:
                    print(f'Train: {transferLosses[-1]:.4f} | Epoch: {epoch + 1}/{epochs}')

            if self.useClearML:
                task.get_logger().report_scalar(
                    title='Loss',
                    series='Training',
                    value=transferLosses[-1],
                    iteration=epoch+1)

        print("Training finished!")
        if self.useClearML:
            task.close()
        model.eval()
        self.adjustedModel = model
        del model
        del optimizerInstance
        del loss
        gc.collect()
        print("All done!")

        return self.adjustedModel, transferLosses, validationLosses

    def matrixHyperParameterSearch(
            self,
            validation: float = 0.2,
            epochs: int = 20,
            batchSize: int = 64,
            lossFunction: Literal["MSE", "MAE"] = "MAE",
            learningRates: List[float] = (1e-6, 1e-5, 1e-4),
            optimizers: List[Literal["Adam", "AdamW", "Adamax", "RMSprop"]] = ("Adam", "AdamW", "Adamax"),
            weightDecays: List[float] = (1e-5, 1e-4, 1e-3),
            verbose: bool = True,
            plot: bool = True
    ) -> Tuple[torch.nn.Module, Dict[str, Union[float, str]]]:
        """
        Performs a grid search over the hyperparameters provided to find the best combination. By default, it will
        plot the training history with plotly in your browser, and (b) print the best hyperparameters found. If the
        ClearML platform was set to be used for logging (at the class initialization), the results will be uploaded
        there as well. If the default values are used, it will test 27 combinations of learning rates, optimizers, and
        weight decays. The method will then adjust the model to the best hyperparameters found, corresponding to the
        lowest validation loss if validation is used, or the lowest training loss if validation is not used
        (``validation=0``). Note that the validation is used by default.

        Args:
            validation: Same as in the ``adjust`` method. Default is ``0.2``.
            epochs: Same as in the ``adjust`` method. Default is ``20`` to keep the search time reasonable on most
                CPU-only machines (around 1 hour). For most cases, a good starting number of epochs is 100-200, which
                should complete in 10-30 minutes on most modern GPUs or Mac M1-series machines (w. device set to MPS).
            batchSize: Same as in the ``adjust`` method. Default is ``32``.
            lossFunction: Same as in the ``adjust`` method. Default is ``MAE``, i.e. Mean Absolute Error or L1 loss.
            learningRates: List of floats with the learning rates to be tested. Default is ``(1e-6, 1e-5, 1e-4)``. See
                the ``adjust`` method for more information.
            optimizers: List of strings with the optimizers to be tested. Default is ``("Adam", "AdamW", "Adamax")``. See
                the ``adjust`` method for more information.
            weightDecays: List of floats with the weight decays to be tested. Default is ``(1e-5, 1e-4, 1e-3)``. See
                the ``adjust`` method for more information.
            verbose: Same as in the ``adjust`` method. Default is ``True``.
            plot: Whether to plot the training history after all the combinations are tested. Default is ``True``.
        """
        nTasks = len(learningRates) * len(optimizers) * len(weightDecays)
        if verbose:
            print("Starting the hyperparameter search...")
            print(f"{nTasks} combinations will be tested.\n")

        bestModel: torch.nn.Module = None
        bestTrainingLoss: float = np.inf
        bestValidationLoss: float = np.inf
        bestHyperparameters: Dict[str, Union[float, str, None]] = {
            "learningRate": None,
            "optimizer": None,
            "weightDecay": None,
            "epochs": None
        }

        trainLossHistory: List[List[float]] = []
        validationLossHistory: List[List[float]] = []
        labels: List[str] = []
        tasksDone = 0
        t0 = time.perf_counter()

        for learningRate in learningRates:
            for optimizer in optimizers:
                for weightDecay in weightDecays:
                    labels.append(f"LR: {learningRate} | OPT: {optimizer} | WD: {weightDecay}")
                    model, trainingLoss, validationLoss = self.adjust(
                        validation=validation,
                        learningRate=learningRate,
                        epochs=epochs,
                        batchSize=batchSize,
                        optimizer=optimizer,
                        weightDecay=weightDecay,
                        lossFunction=lossFunction,
                        verbose=True
                    )
                    trainLossHistory.append(trainingLoss)
                    validationLossHistory.append(validationLoss)
                    if validation > 0:
                        localBestValidationLoss, bestEpoch = min((val, idx) for idx, val in enumerate(validationLoss))
                        if localBestValidationLoss < bestValidationLoss:
                            print(f"New best model found with LR: {learningRate}, OPT: {optimizer}, WD: {weightDecay}, "
                                  f"Epoch: {bestEpoch + 1}/{epochs} | Train: {trainingLoss[bestEpoch]:.4f} | "
                                  f"Validation: {localBestValidationLoss:.4f}")
                            del bestModel
                            gc.collect()
                            bestModel = model
                            bestTrainingLoss = trainingLoss[bestEpoch]
                            bestValidationLoss = localBestValidationLoss
                            bestHyperparameters["learningRate"] = learningRate
                            bestHyperparameters["optimizer"] = optimizer
                            bestHyperparameters["weightDecay"] = weightDecay
                            bestHyperparameters["epochs"] = bestEpoch + 1
                        else:
                            print(f"Model with LR: {learningRate}, OPT: {optimizer}, WD: {weightDecay} did not improve.")
                    else:
                        localBestTrainingLoss, bestEpoch = min((val, idx) for idx, val in enumerate(trainingLoss))
                        if localBestTrainingLoss < bestTrainingLoss:
                            print(f"New best model found with LR: {learningRate}, OPT: {optimizer}, WD: {weightDecay}, "
                                  f"Epoch: {bestEpoch + 1}/{epochs} | Train: {localBestTrainingLoss:.4f}")
                            del bestModel
                            gc.collect()
                            bestModel = model
                            bestTrainingLoss = localBestTrainingLoss
                            bestHyperparameters["learningRate"] = learningRate
                            bestHyperparameters["optimizer"] = optimizer
                            bestHyperparameters["weightDecay"] = weightDecay
                            bestHyperparameters["epochs"] = bestEpoch + 1
                        else:
                            print(f"Model with LR: {learningRate}, OPT: {optimizer}, WD: {weightDecay} did not improve.")

                    tasksDone += 1
                    pastTimePerTask = ((time.perf_counter() - t0)/60) / tasksDone
                    print(f"Task {tasksDone}/{nTasks} done. Estimated time left: {pastTimePerTask * (nTasks - tasksDone):.2f} minutes.\n")


        if verbose:
            print(f"\n\nBest model found with LR: {bestHyperparameters['learningRate']}, OPT: {bestHyperparameters['optimizer']}, "
                  f"WD: {bestHyperparameters['weightDecay']}, Epoch: {bestHyperparameters['epochs']}")
            if validation > 0:
                print(f"Train: {bestTrainingLoss:.4f} | Validation: {bestValidationLoss:.4f}\n")
            else:
                print(f"Train: {bestTrainingLoss:.4f}\n")
        assert bestModel is not None, "The best model was not found. Something went wrong during the hyperparameter search."
        self.adjustedModel = bestModel
        del bestModel
        gc.collect()

        if plot:
            fig1 = go.Figure()
            for idx, label in enumerate(labels):
                fig1.add_trace(
                    go.Scatter(
                        x=np.arange(epochs+1),
                        y=trainLossHistory[idx],
                        mode='lines+markers',
                        name=label)

            )
            fig1.update_layout(
                title="Training Loss History",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                legend_title="Hyperparameters",
                showlegend=True,
                template="plotly_white"
            )
            fig1.show()
            if validation > 0:
                fig2 = go.Figure()
                for idx, label in enumerate(labels):
                    fig2.add_trace(
                        go.Scatter(
                            x=np.arange(epochs+1),
                            y=validationLossHistory[idx],
                            mode='lines+markers',
                            name=label)
                    )
                fig2.update_layout(
                    title="Validation Loss History",
                    xaxis_title="Epoch",
                    yaxis_title="Loss",
                    legend_title="Hyperparameters",
                    showlegend=True,
                    template="plotly_white"
                )
                fig2.show()

        return self.adjustedModel, bestHyperparameters

    def highlightPoints(
            self,
            pointsIndices: List[int]
    ) -> None:
        """
        Highlights data points at certain indices, so that they can be distinguished at later steps. They will be plotted in red by ``plotStarting()``
        and ``plotAdjusted()``. Please note that this will be overwriten the next time you make a call to the ``adjust()``, so you may need to perform 
        it again.

        Args:
            pointsIndices: A list of point indices to highlight. Please note that in Python lists indices start from ``0``.
        """

        if not self.validationLabels:
            print("No validation labels set yet. Please note highlights will be overwriten by the next adjustemnt call.")
        for p in pointsIndices:
            assert p < len(self.validationLabels), "The index of the point to be highlighted is out of bounds."
            self.validationLabels[p] = "Highlight"

    def highlightCompositions(
            self,
            compositions: List[str]
    ) -> None:
        """
        Highlights data points that correspond to certain chemical compositions, so that they can be distinguished at later steps. The strings you 
        provide will be interpreted when matching to the data, so ``HfMo``, ``Hf1Mo1``, ``Hf2Mo2``, and ``Hf50 Mo50`` will all be considered equal.
        They will be plotted in red by ``plotStarting()`` and ``plotAdjusted()``. Please note that this will be overwriten the next time you make 
        a call to the ``adjust()``, so you may need to perform it again.

        Args:
            compositions: A list of strings with chemical formulas. They will be interpreted, so any valid formula pointing to the same composition
                will be parsed in the same fashion. Currently, the composition needs to be exact, i.e. ``Hf33 Mo33 Ni33`` will match to ``HfMoNi`` 
                but ``Hf28.6 Mo71.4`` will not match to ``Hf2 Mo5``. This can be implemented if there is interest.
        """
        
        if not self.validationLabels:
            print("No validation labels set yet. Please note highlights will be overwriten by the next adjustemnt call.")
        assert self.comps, "The compositions must be set before highlighting them. If you use ``OPTIMADEAdjuster``, this is done automatically, but with ``LocalAdjuster``, you have to set them manually."
        reducedFormulas = set([Composition(c).reduced_formula for c in compositions])
        for idx, comp in enumerate(self.comps):
            if comp in reducedFormulas:
                self.validationLabels[idx] = "Highlight"


class OPTIMADEAdjuster(LocalAdjuster):
    """
    Adjuster class operating on data provided by the OPTIMADE API. Primarily geared towards tuning or retraining of the
    models based on other atomistic databases, or their subsets, accessed through OPTIMADE, to adjust the model to a
    different domain, which in the context of DFT datasets could mean adjusting the model to predict properties with DFT
    settings used by that database or focusing its attention to specific chemistry like, for instance, all compounds of
    Sn and all perovskites. It accepts OPTIMADE query as an input and then operates based on the ``LocalAdjuster`` class.

    It will set up the environment for the adjustment, letting you progressively build up the training dataset by
    OPTIMADE queries which get featurized and their results will be concatenated, i.e., you can make one big query or
    several smaller ones and then adjust the model on the whole dataset when you are ready.

    For details on more advanced uses of the OPTIMADE API client, please refer to 
    `the documentation <https://www.optimade.org/optimade-python-tools/latest/getting_started/client/>`_.

    Args:
        calculator: Instance of the ``Calculator`` class with the model to be adjusted, defined and loaded. Unlike in the
            ``LocalAdjuster``, the descriptor data will not be passed, since it will be fetched from the OPTIMADE API.
        model: Name of the model to be adjusted in the ``Calculator``. E.g., ``SIPFENN_Krajewski2022_NN30``.
        provider: Strings with the name of the provider to be used for the OPTIMADE queries. The type-hinting
            gives a list of providers available at the time of writing this code, but it is by no means limited to them.
            For the up-to-date list, along with their current status, please refer to the
            `OPTIMADE Providers Dashboard <https://optimade.org/providers-dashboard>`_. The default is ``"mp"`` which
            stands for the Materials Project, but we do not recommend any particular provider over any other. One has to
            be picked to work out of the box. Your choice should be based on the data you are interested in.
        targetPath: List of strings with the path to the target data in the OPTIMADE response. This will be dependent
            on the provider you choose, and you will need to identify it by looking at the response. The easiest way to
            do this is by going to their endpoint, like
            `this, very neat one, for JARVIS <https://jarvis.nist.gov/optimade/jarvisdft/v1/structures/>`_,
            `this one for Alexandria PBEsol <https://alexandria.icams.rub.de/pbesol/v1/structures>`_,
            `this one for MP <https://optimade.materialsproject.org/v1/structures>`_,
            or `this one for our in-house MPDD <https://optimade.mpdd.org/v1/structures>`_. Examples include
            ``('attributes', '_mp_stability', 'gga_gga+u', 'formation_energy_per_atom')`` for GGA+U formation energy
            per atom in MP, or ``('attributes', '_alexandria_scan_formation_energy_per_atom')`` for the `SCAN` formation
            energy per atom in Alexandria, or ``('attributes', '_alexandria_formation_energy_per_atom')`` for the
            ``GGAsol`` formation energy per atom in Alexandria, or ``('attributes', '_jarvis_formation_energy_peratom')``
            for the `optb88vdw` formation energy per atom in JARVIS, or ``('attributes',
            '_mpdd_formationenergy_sipfenn_krajewski2020_novelmaterialsmodel')`` for the formation energy predicted
            by the SIPFENN_Krajewski2020_NovelMaterialsModel for every structure in MPDD. Default is the MP example.
        targetSize: The length of the target data to be fetched from the OPTIMADE API. This is typically ``1`` for a single
            scalar property, but it can be more. Default is ``1``.
        device: Same as in the ``LocalAdjuster``. Default is ``"cpu"`` which is available on all systems. If you have a
            GPU, you can set it to ``"cuda"``, or to ``"mps"`` if you are using a Mac M1-series machine, in order to
            speed up the training process by orders of magnitude.
        descriptor: *Not* the same as in the ``LocalAdjuster``. Since the descriptor data will be calculated for each
            structure fetched from the OPTIMADE API, this parameter is needed to specify which descriptor to use. At the
            time of writing this code, it can be either ``"Ward2017"`` or ``"KS2022"``. Special versions of ``KS2022``
            cannot be used since assumptions cannot be made about the data fetched from the OPTIMADE API and only general
            symmetry-based optimizations can be applied. Default is ``"KS2022"``.
        useClearML: Same as in the ``LocalAdjuster``. Default is ``False``.
        taskName: Same as in the ``LocalAdjuster``. Default is ``"OPTIMADEFineTuning"``, and you are encouraged to change
            it, especially if you are using the ClearML platform.
        maxResults: The maximum number of results to be fetched from the OPTIMADE API for a given query. Default is
            ``10000`` which is a very high number for most re-training tasks. If you are fetching a lot of data, it's
            possible the query is too broad, and you should consider narrowing it down.
        endpointOverride: List of URL strings with the endpoint to be used for the OPTIMADE queries. This is an advanced
            option allowing you to ignore the ``provider`` parameter and directly specify the endpoint to be used. It is
            useful if you want to use a specific version of the provider's endpoint or narrow down the query to a
            sub-database (Alexandria has two different endpoints for PBEsol and SCAN, for instance). You can also use it
            to query unofficial endpoints. Make sure to (a) include protocol (``http://`` or ``https://``) and (b) not
            include version (``/v1/``), nor the specific endpoint (``/structures``) as the client will add them. I.e.,
            you want ``https://alexandria.icams.rub.de/pbesol`` rather than
            ``alexandria.icams.rub.de/pbesol/v1/structures``. Default is ``None`` which has no effect.

    Attributes:
        reference: List of lists of strings with the references to the data fetched from the OPTIMADE API by looking up
            relations of a given data point. This is not given by most of the providers (as of Fall 2024) but can be
            very useful for some users to track the provenance of the data when possible. If no references are found, the
            list will be a list of empty lists.
    """

    def __init__(
            self,
            calculator: Calculator,
            model: str,
            provider:
                Literal[
                    "aiida",
                    "aflow",
                    "alexandria",
                    "cod",
                    "ccpnc",
                    "cmr",
                    "httk",
                    "matcloud",
                    "mcloud",
                    "mcloudarchive",
                    "mp",
                    "mpdd",
                    "mpds",
                    "mpod",
                    "nmd",
                    "odbx",
                    "omdb",
                    "oqmd",
                    "jarvis",
                    "pcod",
                    "tcod",
                    "twodmatpedia"
                ] = "mp",
            targetPath: List[str] = ('attributes', '_mp_stability', 'gga_gga+u', 'formation_energy_per_atom'),
            targetSize: int = 1,
            device: Literal["cpu", "cuda", "mps"] = "cpu",
            descriptor: Literal["Ward2017", "KS2022"] = "KS2022",
            useClearML: bool = False,
            taskName: str = "OPTIMADEFineTuning",
            maxResults: int = 10000,
            endpointOverride: List[str] = None
    ) -> None:
        from optimade.client import OptimadeClient

        assert isinstance(calculator, Calculator), "The calculator must be an instance of the Calculator class."
        assert isinstance(model, str), "The model must be a string with the name of the model to be adjusted."
        assert isinstance(provider, str), "The provider must be a string with the name of the provider to be used."
        assert len(provider) != 0, "The provider must not be an empty string."
        assert targetPath and isinstance(targetPath, list) or isinstance(targetPath, tuple), "The target path must be a list of strings pointing to the target data in the OPTIMADE response."
        assert len(targetPath) > 0, "The target path must not be empty, i.e., it cannot point to no data."
        if provider != "mp" and targetPath == ('attributes', '_mp_stability', 'gga_gga+u', 'formation_energy_per_atom'):
            raise ValueError("You are utilizing the default (example) property target path specific to the Materials "
                             "Project but you are connecting to a different provider. You must adjust the target path "
                             "to receive data from the provider you are connecting to based on what they serve through "
                             "their provider-specific OPTIMADE endpoint fields. See targetPath docstring for more info.")

        super().__init__(
            calculator=calculator,
            model=model,
            targetData=np.array([]),
            descriptorData=np.array([]),
            device=device,
            descriptor=None,
            useClearML=useClearML,
            taskName=taskName,
        )

        self.descriptor = descriptor
        self.targetPath = targetPath
        self.provider = provider
        if endpointOverride is None:
            self.client = OptimadeClient(
                use_async=False,
                include_providers=[provider],
                max_results_per_provider=maxResults
            )
        else:
            assert isinstance(endpointOverride, list) or isinstance(endpointOverride, tuple), "The endpoint override must be a list of strings."
            assert len(endpointOverride) != 0, "The endpoint override must not be an empty list."
            self.client = OptimadeClient(
                use_async=False,
                base_urls=endpointOverride,
                max_results_per_provider=maxResults
            )

        if self.descriptor == "Ward2017":
            self.descriptorData: np.ndarray = np.empty((0, 271))
        elif self.descriptor == "KS2022":
            self.descriptorData: np.ndarray = np.empty((0, 256))
        else:
            raise NotImplementedError("The descriptor must be either 'Ward2017' or 'KS2022'. Others will be added in the future.")

        self.targetData: np.ndarray = np.empty((0, targetSize))

        self.references: List[List[str]] = []

        print("Initialized Adjuster instance!\n")

    def fetchAndFeturize(
            self,
            query: str,
            parallelWorkers: int = 1,
            verbose: bool = True
    ) -> None:
        """
        Automatically (1) fetches data from ``OPTIMADE API`` provider specified in the given ``OPTIMADEAdjuster`` instance, (2)
        filters the data by checking if the target property is available, and (3) featurizes the incoming data with the
        descriptor calculator selected for ``OPTIMADEAdjuster`` instance (``KS2022`` by default). It effectively prepares everything
        for the adjustments to be made as if local data was loaded and some metadata was added on top of that.

        Args:
            query: A valid ``OPTIMADE API`` query as defined at `the specification page for the OPTIMADE consortium <https://www.optimade.org/>`_.
                These can be made very elaborate by stacking several filters together, but generally retain good readability and are easy to
                interpret thanks to explicit structure written in English. Here are two quick examples: 
                ``'elements HAS "Hf" AND elements HAS "Mo" AND NOT elements HAS ANY "O","C","F","Cl","S"'`` / 
                ``'elements HAS "Hf" AND elements HAS "Mo" AND elements HAS "Zr"'``.
            parallelWorkers: How many workers to use at the featurization step. See ``KS2022`` for more details. On most machines, ``4``-``12`` should 
                be the optimal number.
            verbose: Prints information about progress and results of the process. It is set to ``True`` by default.
        """
        from optimade.adapters.structures import pymatgen as pymatgen_adapter
        from optimade.models import StructureResource

        response = self.client.get(query)
        providerResponse = response['structures'][query]
        respondingProviderURL = list(providerResponse.keys())[0]
        data = providerResponse[respondingProviderURL]['data']

        targetDataStage: List[List[float]] = []
        structs: List[Structure] = []
        comps: List[str] = []
        names: List[str] = []
        missing: List[str] = []
        references: List[List[str]] = []

        if verbose:
            print(f"Obtained {len(data)} structures from the OPTIMADE API.")
            print("Extracting the data...")

        for datapoint in data:
            # OPTIMADE Standard Data
            comp = Composition(datapoint['attributes']['chemical_formula_reduced']).reduced_formula
            name = comp + '-' + datapoint['id']

            # Database-specific payload existing at a specific target path (e.g., formation energy per atom in MP)
            try:
                targetDataStage.append([reduce(operator.getitem, self.targetPath, datapoint)])
            except KeyError:
                missing.append(name)
                continue

            comps.append(comp)
            names.append(name)

            # References for the data. Not present for most providers but we want to capture them if they are.
            try:
                reference = []
                for ref in datapoint['relationships']['references']['data']:
                    if ref['type'] == "references":
                        reference.append(ref['id'])
                references.append(reference)
            except:
                references.append([])
            # Stage for featurization of the received data
            structs.append(pymatgen_adapter.get_pymatgen(StructureResource(**datapoint)))

        if missing:
            print(f"\nCould not find the target data at the provided path: {self.targetPath}\nfor {len(missing)} "
                  f"structures:\n{missing}\n")

        dataIn = list(zip(comps, names, references, structs, targetDataStage))
        assert len(dataIn) != 0, "No data was fetched from the OPTIMADE API. Please check both the query and the provider."
        shuffle(dataIn)
        comps, names, references, structs, targetDataStage = zip(*dataIn)

        self.comps.extend(comps)
        self.names.extend(names)
        self.references.extend(references)

        print(f"Extracted {len(targetDataStage)} datapoints (composition+structure+target) from the OPTIMADE API.")
        self.targetData = np.concatenate((self.targetData, np.array(targetDataStage)), axis=0)

        if verbose:
            print("Featurizing the structures...")

        if self.descriptor == "Ward2017":
            self.calculator.calculate_Ward2017(structs, mode="parallel", max_workers=parallelWorkers)
            self.descriptorData = np.concatenate((self.descriptorData, self.calculator.descriptorData), axis=0)

        elif self.descriptor == "KS2022":
            self.calculator.calculate_KS2022(structs, mode="parallel", max_workers=parallelWorkers)
            self.descriptorData = np.concatenate((self.descriptorData, self.calculator.descriptorData), axis=0)

        else:
            raise NotImplementedError("The descriptor must be either 'Ward2017' or 'KS2022'. Others will be added in the future.")

        self.validationLabels = ["Training"]*len(self.descriptorData)

        if verbose:
            print("Featurization complete!")
            print(f"Current dataset size: "
                  f"{len(self.names)} with "
                  f"{len(set(self.names))} unique IDs belonging to "
                  f"{len(set(self.comps))} unique compositions.\n")
            if len(self.names) > len(set(self.names)):
                print("Please note that there are duplicate IDs in the dataset. Such degenerate dataset can be used "
                      "without issues for training (in some occasions may be even desirable to bias the model to areas "
                      "matching multiple criteria), but the validation error may be underestimated since some data"
                      "may be in both training and validation set.")


if __name__ == '__main__':
    pass
