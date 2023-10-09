# General Imports
import os

import natsort
from pySmartDL import SmartDL
import csv
import numpy as np
from pymatgen.core import Structure
import json
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from time import perf_counter

from importlib import resources

import torch
import onnx2torch
import onnx

from typing import List, Union, Dict

# Descriptor Generators
from pysipfenn.descriptorDefinitions import Ward2017, KS2022, KS2022_dilute

# - add new ones here if extending the code

__version__ = '0.11.0post1'
__authors__ = [["Adam Krajewski", "ak@psu.edu"],
               ["Jonathan Siegel", "jwsiegel@tamu.edu"]]
__name__ = 'pysipfenn'


class Calculator:
    """
        pySIPFENN Calculator automatically initializes all functionalities including identification and loading
        of all available models defined statically in models.json file. It exposes methods for calculating predefined
        structure-informed descriptors (feature vectors) and predicting properties using models that utilize them.

        Args:
            autoLoad: Automatically load all available models. Default: True.

        Attributes:
            models: Dictionary with all model information based on the models.json file in the modelsSIPFENN
                directory. The keys are the network names and the values are dictionaries with the model information.
            loadedModels: Dictionary with all loaded models. The keys are the network names and the values
                are the loaded pytorch models.
            descriptorData: List of all descriptor data created during the last predictions run. The order
                of the list corresponds to the order of atomic structures given to models as input. The order of the
                list of descriptor data for each structure corresponds to the order of networks in the toRun list.
            predictions: List of all predictions created during the last predictions run. The order of the
                list corresponds to the order of atomic structures given to models as input. The order of the list
                of predictions for each structure corresponds to the order of networks in the toRun list.
            inputFiles: List of all input file names used during the last predictions run. The order of the list
                corresponds to the order of atomic structures given to models as input.
    """

    def __init__(self,
                 autoLoad: bool = True,
                 verbose: bool = True):
        if verbose:
            print('*********  Initializing pySIPFENN Calculator  **********')
        # dictionary with all model information
        with resources.files('pysipfenn.modelsSIPFENN').joinpath('models.json').open('r') as f:
            if verbose:
                print(f'Loading model definitions from: {f.name}')
            self.models = json.load(f)
        # networks list
        self.network_list = list(self.models.keys())
        if verbose:
            print(f'Found {len(self.network_list)} network definitions in models.json')
        # network names
        self.network_list_names = [self.models[net]['name'] for net in self.network_list]
        self.network_list_available = []
        self.updateModelAvailability()

        self.loadedModels = {}
        if autoLoad:
            print(f'Loading all available models (autoLoad=True)')
            self.loadModels()
        else:
            print(f'Skipping model loading (autoLoad=False)')

        self.toRun = []
        self.descriptorData = []
        self.predictions = []
        self.inputFiles = []
        if verbose:
            print(f'*********  pySIPFENN Successfully Initialized  **********')

    def __str__(self):
        '''Prints the status of the Calculator object.'''
        printOut = f'pySIPFENN Calculator Object. Version: {__version__}\n'
        printOut += f'Models are located in:\n{resources.files("pysipfenn.modelsSIPFENN")}\n{"-" * 80}\n'
        printOut += f'Loaded Networks: {list(self.loadedModels.keys())}\n'
        if len(self.inputFiles) > 0:
            printOut += f'Last files selected as input: {len(self.inputFiles)}\n'
            if len(self.inputFiles) > 4:
                printOut += f'{self.inputFiles[:2]} ... [{len(self.inputFiles) - 4}] ... {self.inputFiles[-2:]}\n'
        if len(self.descriptorData) > 0:
            printOut += f'Last feature calculation run on: {len(self.descriptorData)} structures\n'
        if len(self.toRun) > 0:
            printOut += f'Last Prediction Run Using: {self.toRun}\n'
        if len(self.predictions) > 0:
            printOut += f'Last prediction run on: {len(self.predictions)} structures\n'
            printOut += f'                        {len(self.predictions[0])} predictions/structure\n'
        return printOut

    def updateModelAvailability(self) -> None:
        """
            Updates availability of models based on the pysipfenn.modelsSIPFENN directory contents. Works only for
            current ONNX model definitions.
        """
        with resources.files('pysipfenn.modelsSIPFENN') as p:
            all_files = os.listdir(p)
        detectedNets = []
        for net, netName in zip(self.network_list, self.network_list_names):
            if all_files.__contains__(net + '.onnx'):
                detectedNets.append(net)
                print('\u2714 ' + netName)
            else:
                print('\u292B ' + netName)
        self.network_list_available = detectedNets

    def downloadModels(self, network: str = 'all') -> None:
        """Downloads ONNX models. By default, all available models are downloaded. If a model is already available
        on disk, it is skipped. If a specific network is given, only that network is downloaded possibly overwriting
        the existing one. If the networks name is not recognized message is printed.

        Args:
            network: Name of the network to download. Defaults to 'all'.

        """
        with resources.files('pysipfenn.modelsSIPFENN') as modelPath:
            # Fetch all
            if network == 'all':
                print('Fetching all networks!')
                for net in self.network_list:
                    if net not in self.network_list_available:
                        if 'URL_ONNX' in self.models[net]:
                            print(f'Fetching: {net}')
                            downloadObject = SmartDL(self.models[net]['URL_ONNX'],
                                                     f'{modelPath}/{net}.onnx',
                                                     threads=16)
                            downloadObject.start()
                            print('\nONNX Network Successfully Fetched.')
                        else:
                            print(f'{net} not detected on disk and ONNX URL has not been provided.')
                    else:
                        print(f'{net} detected on disk. Ready to use.')
                if self.network_list == self.network_list_available:
                    print('All networks available!')
                else:
                    print('Problem occurred.')

            # Fetch single
            elif network in self.network_list:
                print(f'Fetching: {network}')
                downloadObject = SmartDL(self.models[network]['URL_ONNX'],
                                         f'{modelPath}/{network}.onnx',
                                         threads=16)
                downloadObject.start()
                print('\nONNX Network Successfully Fetched.')
            # Not recognized
            else:
                print('Network name not recognized')
        self.updateModelAvailability()

    def calculate_Ward2017(self,
                           structList: List[Structure],
                           mode: str = 'serial',
                           max_workers: int = 4) -> list:
        """Calculates Ward2017 descriptors for a list of structures. The calculation can be done in serial or parallel
        mode. In parallel mode, the number of workers can be specified. The results are stored in the descriptorData
        attribute. The function returns the list of descriptors as well.

        Args:
            structList: List of structures to calculate descriptors for. The structures must be
                initialized with the pymatgen Structure class.
            mode: Mode of calculation. Defaults to 'serial'. Options are 'serial' and 'parallel'.
            max_workers: Number of workers to use in parallel mode. Defaults to 4.

        Returns:
            List of Ward2017 descriptor (feature vector) for each structure.

        """
        if mode == 'serial':
            descList = [Ward2017.generate_descriptor(s) for s in tqdm(structList)]
            print('Done!')
            self.descriptorData = descList
            return descList
        elif mode == 'parallel':
            descList = process_map(Ward2017.generate_descriptor, structList, max_workers=max_workers)
            print('Done!')
            self.descriptorData = descList
            return descList

    def calculate_KS2022(self,
                         structList: List[Structure],
                         mode: str = 'serial',
                         max_workers: int = 8) -> list:
        """Calculates KS2022 descriptors for a list of structures. The calculation can be done in serial or parallel
        mode. In parallel mode, the number of workers can be specified. The results are stored in the descriptorData
        attribute. The function returns the list of descriptors as well.

        Args:
            structList: List of structures to calculate descriptors for. The structures must be
                initialized with the pymatgen Structure class.
            mode: Mode of calculation. Defaults to 'serial'. Options are 'serial' and 'parallel'.
            max_workers: Number of workers to use in parallel mode. Defaults to 8.

        Returns:
            List of KS2022 descriptor (feature vector) for each structure.

        """
        if mode == 'serial':
            descList = [KS2022.generate_descriptor(s) for s in tqdm(structList)]
            print('Done!')
            self.descriptorData = descList
            return descList
        elif mode == 'parallel':
            descList = process_map(KS2022.generate_descriptor, structList, max_workers=max_workers)
            print('Done!')
            self.descriptorData = descList
            return descList

    def calculate_KS2022_dilute(self,
                                structList: List[Structure],
                                baseStruct: Union[str, List[Structure]] = 'pure',
                                mode: str = 'serial',
                                max_workers: int = 8) -> list:
        """Calculates KS2022 descriptors for a list of dilute structures (either based on pure elements and on custom
        base structures, e.g. TCP endmember configurations) that contain a single alloying atom. Speed increases are
        substantial compared to the KS2022 descriptor, which is more general and can be used on any structure. The
        calculation can be done in serial or parallel mode. In parallel mode, the number of workers can be specified.
        The results are stored in the descriptorData attribute. The function returns the list of descriptors as well.

        Args:
            structList: List of structures to calculate descriptors for. The structures must be
                dilute structures (either based on pure elements and on custom base structures, e.g. TCP endmember
                configurations) that contain a single alloying atom. The structures must be initialized with the
                pymatgen Structure class.
            baseStruct: Non-diluted references for the dilute structures. Defaults to 'pure', which assumes that the structures
                are based on pure elements and generates references automatically. Alternatively, a list of structures
                can be provided, which can be either pure elements or custom base structures (e.g. TCP endmember
                configurations).
            mode: Mode of calculation. Defaults to 'serial'. Options are 'serial' and 'parallel'.
            max_workers: Number of workers to use in parallel mode. Defaults to 8.

        Returns:
            List of KS2022 descriptor (feature vector) for each structure.
        """

        if baseStruct == 'pure' or isinstance(baseStruct, Structure):
            if mode == 'serial':
                descList = [KS2022_dilute.generate_descriptor(s, baseStruct=baseStruct) for s in tqdm(structList)]
                print('Done!')
                self.descriptorData = descList
                return descList
            elif mode == 'parallel':
                descList = process_map(KS2022_dilute.generate_descriptor(baseStruct=baseStruct),
                                       structList,
                                       max_workers=max_workers)
                print('Done!')
                self.descriptorData = descList
                return descList

        elif isinstance(baseStruct, List) and len(baseStruct) == len(structList):
            if mode == 'serial':
                descList = [KS2022_dilute.generate_descriptor(s, bs) for s, bs in tqdm(zip(structList, baseStruct))]
                print('Done!')
                self.descriptorData = descList
                return descList
            elif mode == 'parallel':
                descList = process_map(KS2022_dilute.generate_descriptor,
                                       structList, baseStruct, max_workers=max_workers)
                print('Done!')
                self.descriptorData = descList
                return descList

    def loadModels(self, network: str = 'all') -> None:
        """
            Load model/models into memory of the Calculator class. The models are loaded from the modelsSIPFENN directory inside
            the package. It's location can be seen by calling print() on the Calculator. The models are stored in the
            loadedModels attribute as a dictionary with the network string as key and the PyTorch model as value.

            Note:
                This function only works with models that are stored in the modelsSIPFENN directory inside the package,
                are in ONNX format, and have corresponding entries in models.json. For all others, you will need to use
                loadModelCustom().

            Args:
                network: Default is 'all', which loads all models detected as available. Alternatively, a specific model
                    can be loaded by its corresponding key in models.json. E.g. 'SIPFENN_Krajewski2020_NN9' or
                    'SIPFENN_Krajewski2022_NN30'. The key is the same as the network argument in downloadModels().
        """
        with resources.files('pysipfenn.modelsSIPFENN') as modelPath:
            if network == 'all':
                print('Loading models:')
                for net in tqdm(self.network_list_available):
                    self.loadedModels.update({
                        net: onnx2torch.convert(onnx.load(f'{modelPath}/{net}.onnx')).float()
                    })
            elif network in self.network_list_available:
                print('Loading model: ', network)
                self.loadedModels.update({
                    network: onnx2torch.convert(onnx.load(f'{modelPath}/{network}.onnx')).float()
                })
            else:
                raise ValueError(
                    'Network not available. Please check the network name for typos or run downloadModels() '
                    'to download the models. Currently available models are: ', self.network_list_available)

    def loadModelCustom(self, networkName: str, modelName: str, descriptor: str, modelDirectory: str = '.') -> None:
        """
            Load a custom ONNX model from a custom directory specified by the user. The primary use case for this
            function is to load models that are not included in the package and cannot be placed in the package
            directory because of write permissions (e.g. on restrictive HPC systems) or storage allocations.

            Args:
                modelDirectory: Directory where the model is located. Defaults to the current directory.
                networkName: Name of the network. This is the name used to refer to the ONNX network. It has to be
                    unique, not contain any spaces, and correspond to the name of the ONNX file (excluding the .onnx
                    extension).
                modelName: Name of the model. This is the name that will be displayed in the model selection menu. It
                    can be any string desired.
                descriptor: Descriptor/feature vector used by the model. pySIPFENN currently supports the following
                    descriptors: KS2022, and Ward2017.
        """

        self.loadedModels.update({
            networkName: onnx2torch.convert(onnx.load(f'{modelDirectory}/{networkName}.onnx')).float()
        })
        self.models.update({
            networkName: {
                'name': modelName,
                'descriptor': descriptor
            }})
        self.network_list.append(networkName)
        self.network_list_names.append(modelName)
        self.network_list_available.append(networkName)
        print(f'Loaded model {modelName} ({networkName}) from {modelDirectory}')

    def makePredictions(self,
                        models: Dict[str, torch.nn.Module],
                        toRun: List[str],
                        dataInList: List[Union[List[float], np.array]]) -> List[list]:
        """Makes predictions using PyTorch networks listed in toRun and provided in models dictionary.

        Args:
            models: Dictionary of models to use. Keys are network names and values are PyTorch models.
            toRun: List of networks to run. Must be a subset of models.keys().
            dataInList: List of data to make predictions for. Each element of the list should be a descriptor accepted
                by all networks in toRun. Can be a list of lists of floats or a list of numpy arrays.

        Returns:
            List of predictions. Each element of the list is a list of predictions for all ran network. The order of the
            predictions is the same as the order of the networks in toRun.
        """
        dataOuts = []
        print('Making predictions...')
        # Run for each network
        dataIn = torch.from_numpy(np.array(dataInList)).float()
        for net in toRun:
            t0 = perf_counter()
            model = models[net]
            model.eval()
            if 'OnnxDropoutDynamic()' in {str(module) for module in list(model._modules.values())}:
                tempOut = model(dataIn, None)
            else:
                tempOut = model(dataIn)
            t1 = perf_counter()
            dataOuts.append(tempOut.cpu().detach().numpy())
            print(f'Prediction rate: {round(len(tempOut) / (t1 - t0), 1)} pred/s')
            print(f'Obtained {len(tempOut)} predictions from:  {net}')

        # Transpose and round the predictions
        dataOuts = np.array(dataOuts).T.tolist()[0]
        self.predictions = dataOuts
        return dataOuts

    def findCompatibleModels(self, descriptor: str) -> List[str]:
        """Finds all models compatible with a given descriptor based on the descriptor definitions loaded from the
        models.json file.

        Args:
            descriptor: Descriptor to use. Must be one of the available descriptors. See pysipfenn.descriptorDefinitions
                to see available modules or add yours. Available default descriptors are: 'Ward2017', 'KS2022'.

        Returns:
            List of compatible models.
        """

        compatibleList = []
        for net in self.models:
            if descriptor in self.models[net]['descriptor']:
                compatibleList.append(net)
        return compatibleList

    def runModels(self,
                  descriptor: str,
                  structList: List[Structure],
                  mode: str = 'serial',
                  max_workers: int = 4) -> List[list]:
        """Runs all loaded models on a list of Structures using specified descriptor. Supports serial and parallel
        computation modes. If parallel is selected, max_workers determines number of processes handling the
        featurization of structures (90-99+% of computational intensity) and models are then run in series.

        Args:
            descriptor: Descriptor to use. Must be one of the available descriptors. See pysipfenn.descriptorDefinitions
                to see available modules or add yours. Available default descriptors are: 'Ward2017', 'KS2022'.
            structList: List of pymatgen Structure objects to run the models on.
            mode: Computation mode. 'serial' or 'parallel'. Default is 'serial'. Parallel mode is not recommended for
                small datasets.
            max_workers: Number of workers to use in parallel mode. Default is 4. Ignored in serial mode. If set to
                None, will use all available cores. If set to 0, will use 1 core.

        Returns:
            List of predictions. Each element of the list is a list of predictions for all ran networks. The
            order of the predictions is the same as the order of the input structures. The order of the networks is
            the same as the order of the networks in self.network_list_available. If a network is not available, it
            will not be included in the list. If a network is not compatible with the selected descriptor, it will
            not be included in the list.
        """

        self.toRun = list(set(self.findCompatibleModels(descriptor)).intersection(set(self.network_list_available)))
        self.toRun = natsort.natsorted(self.toRun)
        if len(self.toRun) == 0:
            print('The list of models to run is empty. This may be caused by selecting a descriptor not '
                  'defined/available, or if the selected descriptor does not correspond to any available network. '
                  'Check spelling and invoke the downloadModels() function if you are using base models.')
            raise AssertionError
        else:
            print(f'\nModels that will be run: {self.toRun}')

        print('Calculating descriptors...')
        if descriptor == 'Ward2017':
            self.descriptorData = self.calculate_Ward2017(structList=structList,
                                                          mode=mode,
                                                          max_workers=max_workers)
        elif descriptor == 'KS2022':
            self.descriptorData = self.calculate_KS2022(structList=structList,
                                                        mode=mode,
                                                        max_workers=max_workers)
        else:
            print('Descriptor handing not implemented. Check spelling.')
            raise AssertionError

        self.predictions = self.makePredictions(models=self.loadedModels,
                                                toRun=self.toRun,
                                                dataInList=self.descriptorData)

        return self.predictions

    def runModels_dilute(self,
                         descriptor: str,
                         structList: List[Structure],
                         baseStruct: Union[str, List[Structure]] = 'pure',
                         mode: str = 'serial',
                         max_workers: int = 4) -> List[list]:
        """Runs all loaded models on a list of Structures using specified descriptor. A critical difference
        from runModels() is that this function will call dilute-specific featurizer, e.g. KS2022_dilute when KS2022 is
        provided as input, which can only be used on dilute structures (both based on pure elements and on custom base
        structures, e.g. TCP endmember configurations) that contain a single alloying atom. Speed increases are
        substantial compared to the KS2022 descriptor, which is more general and can be used on any structure.
        Supports serial and parallel modes in the same way as runModels().

        Args:
            descriptor: Descriptor to use for predictions. Must be one of the descriptors which support the dilute
                structures (i.e. *_dilute). See pysipfenn.descriptorDefinitions to see available modules or add yours
                here. Available default dilute descriptors are now: 'KS2022'. The 'KS2022' can also be called from
                runModels() function, but is not recommended for dilute alloys, as it negates the speed increase of the
                dilute structure featurizer.
            structList: List of pymatgen Structure objects to run the models on. Must be dilute structures as described
                above.
            baseStruct: Non-diluted references for the dilute structures. Defaults to 'pure', which assumes that the
                structures are based on pure elements and generates references automatically. Alternatively, a list of
                structures can be provided, which can be either pure elements or custom base structures (e.g. TCP
                endmember configurations).
            mode: Computation mode. 'serial' or 'parallel'. Default is 'serial'. Parallel mode is not recommended for
                small datasets.
            max_workers: Number of workers to use in parallel mode. Default is 4. Ignored in serial mode. If set to
                None, will use all available cores. If set to 0, will use 1 core.

        Returns:
            List of predictions. Each element of the list is a list of predictions for all ran networks. The
            order of the predictions is the same as the order of the input structures. The order of the networks
            is the same as the order of the networks in self.network_list_available. If a network is not available,
            it will not be included in the list. If a network is not compatible with the selected descriptor, it
            will not be included in the list.
        """

        self.toRun = list(set(self.findCompatibleModels(descriptor)).intersection(set(self.network_list_available)))
        if len(self.toRun) == 0:
            print('The list of models to run is empty. This may be caused by selecting a descriptor not '
                  'defined/available, or if the selected descriptor does not correspond to any available network. '
                  'Check spelling and invoke the downloadModels() function if you are using base models.')
            raise AssertionError
        else:
            print(f'Running {self.toRun} models')

        print('Calculating descriptors...')
        if descriptor == 'KS2022':
            self.descriptorData = self.calculate_KS2022_dilute(structList=structList,
                                                               baseStruct=baseStruct,
                                                               mode=mode,
                                                               max_workers=max_workers)
        else:
            print('Descriptor handing not implemented. Check spelling.')
            raise AssertionError

        self.predictions = self.makePredictions(models=self.loadedModels,
                                                toRun=self.toRun,
                                                dataInList=self.descriptorData)

        return self.predictions

    def get_resultDicts(self) -> List[dict]:
        """Returns a list of dictionaries with the predictions for each network. The keys of the dictionaries are the
        names of the networks. The order of the dictionaries is the same as the order of the input structures passed
        through runModels() functions.

        Returns:
            List of dictionaries with the predictions.
        """
        return [dict(zip(self.toRun, pred)) for pred in self.predictions]

    def get_resultDictsWithNames(self) -> List[dict]:
        """Returns a list of dictionaries with the predictions for each network. The keys of the dictionaries are the
        names of the networks and the names of the input structures. The order of the dictionaries is the same as the
        order of the input structures passed through runModels() functions. Note that this function requires
        self.inputFiles to be set, which is done automatically when using runFromDirectory() or
        runFromDirectory_dilute() but not when using runModels() or runModels_dilute(), as the input structures are
        passed directly to the function and names have to be provided separately by assigning them to self.inputFiles.

        Returns:
            List of dictionaries with the predictions.
        """
        assert self.inputFiles is not []
        assert len(self.inputFiles) == len(self.predictions)
        return [
            dict(zip(['name'] + self.toRun, [name] + pred))
            for name, pred in
            zip(self.inputFiles, self.predictions)]

    def runFromDirectory(self,
                         directory: str,
                         descriptor: str,
                         mode: str = 'serial',
                         max_workers: int = 4
                         ) -> List[list]:
        """Runs all loaded models on a list of Structures it automatically imports from a specified directory. The
        directory must contain only atomic structures in formats such as 'poscar', 'cif', 'json', 'mcsqs', etc.,
        or a mix of these. The structures are automatically sorted using natsort library, so the order of the
        structures in the directory, as defined by the operating system, is not important. Natural sorting,
        for example, will sort the structures in the following order: '1-Fe', '2-Al', '10-xx', '11-xx', '20-xx',
        '21-xx', '11111-xx', etc. This is useful when the structures are named using a numbering system. The order of
        the predictions is the same as the order of the input structures. The order of the networks in a prediction
        is the same as the order of the networks in self.network_list_available. If a network is not available,
        it will not be included in the list.

        Args:
            directory: Directory containing the structures to run the models on. The directory must contain only atomic
                structures in formats such as 'poscar', 'cif', 'json', 'mcsqs', etc., or a mix of these. The structures
                are automatically sorted as described above.
            descriptor: Descriptor to use. Must be one of the available descriptors. See pysipgenn.descriptorDefinitions
                for a list of available descriptors.
            mode: Computation mode. 'serial' or 'parallel'. Default is 'serial'. Parallel mode is not recommended for
                small datasets.
            max_workers: Number of workers to use in parallel mode. Default is 4. Ignored in serial mode. If set to
                None, will use all available cores. If set to 0, will use 1 core.

        Returns:
            List of predictions. Each element of the list is a list of predictions for all ran networks. The order of
            the predictions is the same as the order of the input structures. The order of the networks is the same as
            the order of the networks in self.network_list_available. If a network is not available, it will not be
            included in the list.
        """

        print('Importing structures...')
        self.inputFiles = os.listdir(directory)
        self.inputFiles = natsort.natsorted(self.inputFiles)
        structList = [Structure.from_file(f'{directory}/{eif}') for eif in tqdm(self.inputFiles)]
        self.runModels(descriptor=descriptor, structList=structList, mode=mode, max_workers=max_workers)
        print('Done!')

        return self.predictions

    def runFromDirectory_dilute(self,
                                directory: str,
                                descriptor: str,
                                baseStruct: str = 'pure',
                                mode: str = 'serial',
                                max_workers: int = 8) -> None:
        """Runs all loaded models on a list of dilute Structures it automatically imports from a specified directory.
        The directory must contain only atomic structures in formats such as 'poscar', 'cif', 'json', 'mcsqs', etc.,
        or a mix of these. The structures are automatically sorted using natsort library, so the order of the
        structures in the directory, as defined by the operating system, is not important. Natural sorting,
        for example, will sort the structures in the following order: '1-Fe', '2-Al', '10-xx', '11-xx', '20-xx',
        '21-xx', '11111-xx', etc. This is useful when the structures are named using a numbering system. The order of
        the predictions is the same as the order of the input structures. The order of the networks in a prediction
        is the same as the order of the networks in self.network_list_available. If a network is not available,
        it will not be included in the list.

        Args:
            directory: Directory containing the structures to run the models on. The directory must contain only atomic
                structures in formats such as 'poscar', 'cif', 'json', 'mcsqs', etc., or a mix of these. The structures
                are automatically sorted as described above. The structures must be dilute structures, i.e. they must
                contain only one alloying element.
            descriptor: Descriptor to use. Must be one of the available descriptors. See pysipgenn.descriptorDefinitions
                for a list of available descriptors.
            baseStruct: Non-diluted references for the dilute structures. Defaults to 'pure', which assumes that the
                structures are based on pure elements and generates references automatically. Alternatively, a list of
                structures can be provided, which can be either pure elements or custom base structures (e.g. TCP
                endmember configurations).
            mode: Computation mode. 'serial' or 'parallel'. Default is 'serial'. Parallel mode is not recommended for
                small datasets.
            max_workers: Number of workers to use in parallel mode. Default is 8. Ignored in serial mode. If set to
                None, will use all available cores. If set to 0, will use 1 core.

        Returns:
            List of predictions. Each element of the list is a list of predictions for all ran networks. The order of
            the predictions is the same as the order of the input structures. The order of the networks is the same as
            the order of the networks in self.network_list_available. If a network is not available, it will not be
            included in the list.

        """
        print('Importing structures...')
        self.inputFiles = os.listdir(directory)
        self.inputFiles = natsort.natsorted(self.inputFiles)
        structList = [Structure.from_file(f'{directory}/{eif}') for eif in tqdm(self.inputFiles)]
        self.runModels_dilute(descriptor=descriptor,
                              structList=structList,
                              baseStruct=baseStruct,
                              mode=mode,
                              max_workers=max_workers)
        print('Done!')

    def writeResultsToCSV(self, file: str) -> None:
        """Writes the results to a CSV file. The first column is the name of the structure. If the self.inputFiles
        attribute is populated automatically by runFromDirectory() or set manually, the names of the structures will
        be used. Otherwise, the names will be '1', '2', '3', etc. The remaining columns are the predictions for each
        network. The order of the columns is the same as the order of the networks in self.network_list_available.

        Args:
            file: Name of the file to write the results to. If the file already exists, it will be overwritten. If the
                file does not exist, it will be created. The file must have a '.csv' extension to be recognized
                correctly.
        """

        assert self.toRun is not []
        with open(file, 'w+', encoding="utf-8") as f:
            f.write('Name,' + ','.join(self.toRun) + '\n')
            if len(self.inputFiles) == len(self.predictions):
                for name, pred in zip(self.inputFiles, self.predictions):
                    assert len(pred) == len(self.toRun)
                    f.write(f'{name},{",".join(str(v) for v in pred)}\n')
            else:
                i = 1
                for pred in self.predictions:
                    f.write(f'{i},{",".join(str(v) for v in pred)}\n')
                    i += 1

    def writeDescriptorsToCSV(self, descriptor: str, file: str = 'descriptorData.csv') -> None:
        """Writes the descriptor data to a CSV file. The first column is the name of the structure. If the
        self.inputFiles attribute is populated automatically by runFromDirectory() or set manually, the names of the
        structures will be used. Otherwise, the names will be '1', '2', '3', etc. The remaining columns are the
        descriptor values. The order of the columns is the same as the order of the labels in the descriptor
        definition file.

        Args:
            descriptor: Descriptor to use. Must be one of the available descriptors. See pysipgenn.descriptorDefinitions
                for a list of available descriptors, such as 'KS2022' and 'Ward2017'.
            file: Name of the file to write the results to. If the file already exists, it will be overwritten. If the
                file does not exist, it will be created. The file must have a '.csv' extension to be recognized
                correctly.
        """

        # Load descriptor labels
        with resources.files('pysipfenn').joinpath(f'descriptorDefinitions/labels_{descriptor}.csv') as labelsCSV:
            with open(labelsCSV, 'r') as f:
                reader = csv.reader(f)
                labels = [v[0] for v in list(reader)]

        # Write descriptor data
        with open(file, 'w+', encoding="utf-8") as f:
            f.write(f'Name,{",".join(labels)}\n')
            if len(self.inputFiles) == len(self.descriptorData):
                for name, dd in zip(self.inputFiles, self.descriptorData):
                    assert len(dd) == len(labels)
                    f.write(f'{name},{",".join(str(v) for v in dd.tolist())}\n')
            else:
                i = 1
                for dd in self.descriptorData:
                    f.write(f'{i},{",".join(str(v) for v in dd)}\n')
                    i += 1


def ward2ks2022(ward2017: np.ndarray) -> np.ndarray:
    """Converts a Ward 2017 descriptor to a KS2022 descriptor (which is its subset).

    Args:
        ward2017: Ward2017 descriptor. Must be a 1D NumPy array of length 271.

    Returns:
        KS2022 descriptor array.
    """

    assert isinstance(ward2017, np.ndarray)
    assert ward2017.shape == (271,)
    ward2017split = np.split(ward2017, [12, 15, 121, 126, 258, 264, 268, 269, 271])
    ks2022 = np.concatenate((
        ward2017split[0],
        ward2017split[2],
        ward2017split[4],
        ward2017split[6],
        ward2017split[8]
    ), axis=-1, dtype=np.float32)

    return ks2022
