# Standard Library Imports
import os
import gc
import csv
import json
from time import perf_counter
from typing import List, Union, Dict
from importlib import resources

# Helper Imports
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import natsort
from pysmartdl2 import SmartDL
from colorama import Fore, Style

# Scientific Computing Imports
import numpy as np
from pymatgen.core import Structure, Composition

# Machine Learning Imports
import torch
import onnx2torch
import onnx

# YAML Handling Imports and Configuration
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import LiteralScalarString

# Descriptor Generators
from pysipfenn.descriptorDefinitions import (
    Ward2017, KS2022, KS2022_dilute, KS2022_randomSolutions
)

# - add new ones here if extending the code

__version__ = '0.15.0'
__authors__ = [["Adam Krajewski", "ak@psu.edu"],
               ["Jonathan Siegel", "jwsiegel@tamu.edu"]]
__name__ = 'pysipfenn'

# *********************************  CALCULATION HIGH-LEVEL ENVIRONMENT  *********************************
class Calculator:
    """pySIPFENN Calculator automatically initializes all functionalities including identification and loading
    of all available models defined statically in the ``models.json`` file. It exposes methods for calculating predefined
    structure-informed descriptors (feature vectors) and predicting properties using models that utilize them.

    Args:
        autoLoad: Automatically load all available ML models based on the ``models.json`` file. This `will` require
            significant memory and time if they are available, so for featurization and other non-model-requiring
            tasks, it is recommended to set this to ``False``. Defaults to ``True``.
        verbose: Print initialization messages and several other non-critical messages during runtime procedures.
            Defaults to True.

    Attributes:
        models: Dictionary with all model information based on the ``models.json`` file in the modelsSIPFENN
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
        """Initializes the pySIPFENN Calculator object."""
        if verbose:
            print('\n*********  Initializing pySIPFENN Calculator  **********')
            self.verbose = verbose
        # dictionary with all model information
        with resources.files('pysipfenn.modelsSIPFENN').joinpath('models.json').open('r') as f:
            if verbose:
                print(f'Loading model definitions from: {Fore.BLUE}{f.name}{Style.RESET_ALL}')
            self.models = json.load(f)
        # networks list
        self.network_list = list(self.models.keys())
        if verbose:
            print(f'Found {Fore.BLUE}{len(self.network_list)} network definitions in models.json{Style.RESET_ALL}')
        # network names
        self.network_list_names = [self.models[net]['name'] for net in self.network_list]
        self.network_list_available = []
        self.updateModelAvailability()

        self.loadedModels = {}
        if autoLoad:
            print(f'Loading all available models ({Fore.BLUE}autoLoad=True{Style.RESET_ALL})')
            self.loadModels()
        else:
            print(f'Skipping model loading ({Fore.BLUE}autoLoad=False{Style.RESET_ALL})')

        self.prototypeLibrary = {}
        self.parsePrototypeLibrary(verbose=verbose)

        self.toRun = []
        self.descriptorData = []
        self.predictions = []
        self.metas = {
            'RSS': []
        }
        self.inputFiles = []
        if verbose:
            print(f'{Fore.GREEN}**********      Successfully Initialized      **********{Style.RESET_ALL}')

    def __str__(self):
        """Prints the status of the ``Calculator`` object."""
        printOut = f'pySIPFENN Calculator Object. Version: {__version__}\n'
        printOut += f'Models are located in:\n   {resources.files("pysipfenn.modelsSIPFENN")}\n'
        printOut += f'Auxiliary files (incl. structure prototypes):\n   {resources.files("pysipfenn.misc")}\n{"-" * 80}\n'
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

    # *********************************  PROTOTYPE HANDLING  *********************************
    def parsePrototypeLibrary(self,
                              customPath: str = "default",
                              verbose: bool = False,
                              printCustomLibrary: bool = False) -> None:
        """Parses the prototype library YAML file in the ``misc`` directory, interprets them into pymatgen ``Structure``
        objects, and stores them in the ``self.prototypeLibrary`` dict attribute of the ``Calculator`` object. You can use it
        also to temporarily append a custom prototype library (by providing a path) which will live as long as the
        ``Calculator``. For permanent changes, use ``appendPrototypeLibrary()``.

        Args:
            customPath: Path to the prototype library YAML file. Defaults to the magic string ``"default"``, which loads the
                default prototype library included in the package in the ``misc`` directory.
            verbose: If True, it prints the number of prototypes loaded. Defaults to ``False``, but note that ``Calculator``
                class automatically initializes with ``verbose=True``.
            printCustomLibrary: If True, it prints the name and POSCAR of each prototype being added to the prototype
                library. Has no effect if ``customPath`` is ``'default'``. Defaults to ``False``.

        Returns:
            None
        """
        yaml_safeLoader = YAML(typ='safe')

        if customPath == 'default':
            with resources.files('pysipfenn.misc').joinpath('prototypeLibrary.yaml').open('r') as f:
                prototypes = yaml_safeLoader.load(f)
        else:
            with open(customPath, 'r') as f:
                prototypes = yaml_safeLoader.load(f)
                if printCustomLibrary:
                    for prototype in prototypes:
                        print(f'{prototype["name"]}:\n{prototype["POSCAR"]}')
        for prototype in prototypes:
            assert isinstance(prototype['name'], str), 'Prototype name must be a string.'
            assert isinstance(prototype['POSCAR'], str), 'Prototype POSCAR must be a string.'
            assert isinstance(prototype['origin'], str), 'Prototype origin must be a string.'
            struct = Structure.from_str(prototype['POSCAR'], fmt='poscar')
            assert struct.is_valid(), f'Invalid structure for prototype {prototype["name"]}'
            assert struct.is_ordered, f'Unordered structure for prototype {prototype["name"]}. Make sure that the ' \
                                        f'POSCAR file is in the direct format and that no prior randomization has ' \
                                        f'been applied to the structure occupancies.'
            self.prototypeLibrary.update({
                prototype['name']: {
                    'POSCAR': prototype['POSCAR'],
                    'structure': struct,
                    'origin': prototype['origin']
                }
            })
        if verbose:
            protoLen = len(self.prototypeLibrary)
            if protoLen == 0:
                print(f"{Style.DIM}No prototypes were loaded into the prototype library.{Style.RESET_ALL}")
            else:
                print(f"Loaded {Fore.GREEN}{protoLen} prototypes {Style.RESET_ALL}into the library.")
            

    def appendPrototypeLibrary(self, customPath: str) -> None:
        """Parses a custom prototype library YAML file and permanently appends it into the internal prototypeLibrary
        of the pySIPFENN package. They will be persisted for future use and, by default, they will be loaded
        automatically when instantiating the ``Calculator`` object, similar to your custom models.

        Args:
            customPath: Path to the prototype library YAML file to be appended to the internal ``self.prototypeLibrary``
                of the ``Calculator`` object.

        Returns:
            None
        """

        self.parsePrototypeLibrary(customPath=customPath, printCustomLibrary=True, verbose=True)
        print(f'Now, {len(self.prototypeLibrary)} prototype structures are present into the prototype library. '
              f'Persisting them for future use.')
        overwritePrototypeLibrary(self.prototypeLibrary)

    # *********************************  MODEL HANDLING  *********************************
    def updateModelAvailability(self) -> None:
        """Updates availability of models based on the pysipfenn.modelsSIPFENN directory contents. Works only for
        current ONNX model definitions."""
        with resources.files('pysipfenn.modelsSIPFENN') as p:
            all_files = os.listdir(p)
        detectedNets = []
        for net, netName in zip(self.network_list, self.network_list_names):
            if all_files.__contains__(net + '.onnx'):
                detectedNets.append(net)
                print(f"{Fore.GREEN}✔ {netName}{Style.RESET_ALL}")
            else:
                print(f"{Style.DIM}x {netName}{Style.RESET_ALL}")
        self.network_list_available = detectedNets

    def downloadModels(self, network: str = 'all') -> None:
        """Downloads ONNX models. By default, all available models are downloaded. If a model is already available
        on disk, it is skipped. If a specific ``network`` is given, only that network is downloaded, possibly overwriting
        the existing one. If the ``network`` name is not recognized, the message will be printed.

        Args:
            network: Name of the network to download. Defaults to ``'all'``.

        """
        with resources.files('pysipfenn.modelsSIPFENN') as modelPath:
            # Fetch all
            if network == 'all':
                print('Fetching all networks!')
                downloadableNets = [net for net in self.network_list if 'URL_ONNX' in self.models[net]]
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
                if downloadableNets == self.network_list_available:                
                    print('All downloadable networks are now available!')
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

    def loadModels(self, network: str = 'all') -> None:
        """Load model/models into memory of the ``Calculator`` class. The models are loaded from the ``modelsSIPFENN`` directory
        inside the package. Its location can be seen by calling ``print()`` on the ``Calculator``. The models are stored in the
        ``self.loadedModels`` attribute as a dictionary with the network string as key and the PyTorch model as value.

         Note:
            This function only works with models that are stored in the ``modelsSIPFENN`` directory inside the package,
            are in ONNX format, and have corresponding entries in ``models.json``. For all others, you will need to use
            ``loadModelCustom()``.

        Args:
            network: Default is ``'all'``, which loads all models detected as available. Alternatively, a specific model
                can be loaded by its corresponding key in models.json. E.g. ``'SIPFENN_Krajewski2020_NN9'`` or
                ``'SIPFENN_Krajewski2022_NN30'``. The key is the same as the network argument in ``downloadModels()``.

        Raises:
            ValueError: If the network name is not recognized or if the model is not available in the ``modelsSIPFENN``
                directory.

        Returns:
            None. It updates the loadedModels attribute of the Calculatorclass.
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

    def loadModelCustom(
            self,
            networkName: str,
            modelName: str,
            descriptor: str,
            modelDirectory: str = '.'
    ) -> None:
        """Load a custom ONNX model from a custom directory specified by the user. The primary use case for this
        function is to load models that are not included in the package and cannot be placed in the package
        directory because of write permissions (e.g. on restrictive HPC systems) or storage allocations.

        Args:
            modelDirectory: Directory where the model is located. Defaults to the current directory.
            networkName: Name of the network. This is the name used to refer to the ONNX network. It has to be
                unique, not contain any spaces, and correspond to the name of the ONNX file (excluding the ``.onnx``
                extension).
            modelName: Name of the model. This is the name that will be displayed in the model selection menu. It
                can be any string desired.
            descriptor: Descriptor/feature vector used by the model. pySIPFENN currently supports the following
                descriptors: ``'KS2022'``, and ``'Ward2017'``.
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

    def findCompatibleModels(self, descriptor: str) -> List[str]:
        """Finds all models compatible with a given descriptor based on the descriptor definitions loaded from the
        ``models.json`` file.

        Args:
            descriptor: Descriptor to use. Must be one of the available descriptors. See ``pysipfenn.descriptorDefinitions``
                to see available modules or add yours. Available default descriptors are: ``'Ward2017'``, ``'KS2022'``.

        Returns:
            List of strings corresponding to compatible models.
        """

        compatibleList = []
        for net in self.models:
            if descriptor in self.models[net]['descriptor']:
                compatibleList.append(net)
        return compatibleList

    # *******************************  DESCRIPTOR HANDLING (MID-LEVEL API) *******************************
    def calculate_Ward2017(
            self,
            structList: List[Structure],
            mode: str = 'serial',
            max_workers: int = 4
    ) -> list:
        """Calculates ``Ward2017`` descriptors for a list of structures. The calculation can be done in serial or parallel
        mode. In parallel mode, the number of workers can be specified. The results are stored in the ``self.descriptorData``
        attribute. The function returns the list of descriptors as well.

        Args:
            structList: List of structures to calculate descriptors for. The structures must be
                initialized with the pymatgen ``Structure`` class.
            mode: Mode of calculation. Defaults to 'serial'. Options are ``'serial'`` and ``'parallel'``.
            max_workers: Number of workers to use in parallel mode. Defaults to ``4``. If ``None``, the number of workers
                will be set to the number of available CPU cores. If set to ``0``, 1 worker will be used.

        Returns:
            List of ``Ward2017`` descriptor (feature vector) for each structure.

        """
        if mode == 'serial':
            descList = [Ward2017.generate_descriptor(s) for s in tqdm(structList)]
            if self.verbose: print('Done!')
            self.descriptorData = descList
            return descList
        elif mode == 'parallel':
            descList = process_map(Ward2017.generate_descriptor, structList, max_workers=max_workers)
            if self.verbose: print('Done!')
            self.descriptorData = descList
            return descList

    def calculate_KS2022(
            self,
            structList: List[Structure],
            mode: str = 'serial',
            max_workers: int = 8
    ) -> list:
        """Calculates ``KS2022`` descriptors for a list of structures. The calculation can be done in serial or parallel
        mode. In parallel mode, the number of workers can be specified. The results are stored in the descriptorData
        attribute. The function returns the list of descriptors as well.

        Args:
            structList: List of structures to calculate descriptors for. The structures must be
                initialized with the pymatgen ``Structure`` class.
            mode: Mode of calculation. Defaults to 'serial'. Options are ``'serial'`` and ``'parallel'``.
            max_workers: Number of workers to use in parallel mode. Defaults to ``8``. If ``None``, the number of workers
                will be set to the number of available CPU cores. If set to ``0``, 1 worker will be used.

        Returns:
            List of ``KS2022`` descriptor (feature vector) for each structure.

        """
        if mode == 'serial':
            descList = [KS2022.generate_descriptor(s) for s in tqdm(structList)]
            if self.verbose: print('Done!')
            self.descriptorData = descList
            return descList
        elif mode == 'parallel':
            descList = process_map(KS2022.generate_descriptor, structList, max_workers=max_workers)
            if self.verbose: print('Done!')
            self.descriptorData = descList
            return descList

    def calculate_KS2022_dilute(
            self,
            structList: List[Structure],
            baseStruct: Union[str, List[Structure]] = 'pure',
            mode: str = 'serial',
            max_workers: int = 8
    ) -> List[np.ndarray]:
        """Calculates ``KS2022`` descriptors for a list of dilute structures (either based on pure elements and on custom
        base structures, e.g. TCP endmember configurations) that contain a single alloying atom. Speed increases are
        substantial compared to the ``KS2022`` descriptor, which is more general and can be used on any structure. The
        calculation can be done in serial or parallel mode. In parallel mode, the number of workers can be specified.
        The results are stored in the ``self.descriptorData`` attribute. The function returns the list of descriptors as well.

        Args:
            structList: List of structures to calculate descriptors for. The structures must be
                dilute structures (either based on pure elements and on custom base structures, e.g. TCP endmember
                configurations) that contain a single alloying atom. The structures must be initialized with the
                pymatgen ``Structure`` class.
            baseStruct: Non-diluted references for the dilute structures. Defaults to ``'pure'``, which assumes that the structures
                are based on pure elements and generates references automatically. Alternatively, a list of structures
                can be provided, which can be either pure elements or custom base structures (e.g. TCP endmember
                configurations).
            mode: Mode of calculation. Defaults to ``'serial'``. Options are ``'serial'`` and ``'parallel'``.
            max_workers: Number of workers to use in parallel mode. Defaults to ``8``. If ``None``, the number of workers
                will be set to the number of available CPU cores. If set to ``0``, 1 worker will be used.

        Returns:
            List of ``KS2022`` descriptor (feature vector) ``np.ndarray`` for each structure.
        """

        if baseStruct == 'pure' or isinstance(baseStruct, Structure):
            if mode == 'serial':
                descList = [KS2022_dilute.generate_descriptor(s, baseStruct=baseStruct) for s in tqdm(structList)]
                if self.verbose:
                    print('Done!')
                self.descriptorData = descList
                return descList
            elif mode == 'parallel':
                pairedInput = list(zip(structList, [baseStruct] * len(structList)))
                descList = process_map(wrapper_KS2022_dilute_generate_descriptor,
                                       pairedInput,
                                       max_workers=max_workers)
                if self.verbose:
                    print('Done!')
                self.descriptorData = descList
                return descList

        elif isinstance(baseStruct, List) and len(baseStruct) == len(structList):
            if mode == 'serial':
                descList = [KS2022_dilute.generate_descriptor(s, bs) for s, bs in tqdm(zip(structList, baseStruct))]
                if self.verbose:
                    print('Done!')
                self.descriptorData = descList
                return descList
            elif mode == 'parallel':
                pairedInput = list(zip(structList, baseStruct))
                descList = process_map(wrapper_KS2022_dilute_generate_descriptor,
                                       pairedInput, max_workers=max_workers)
                if self.verbose:
                    print('Done!')
                self.descriptorData = descList
                return descList
            else:
                raise ValueError('`baseStruct` must be (1) `pure`, (2) `Structure` or a list of them.')

    def calculate_KS2022_randomSolutions(
            self,
            baseStructList: Union[str, Structure, List[str], List[Structure], List[Union[Composition, str]]],
            compList: Union[str, List[str], Composition, List[Composition], List[Union[Composition, str]]],
            minimumSitesPerExpansion: int = 50,
            featureConvergenceCriterion: float = 0.005,
            compositionConvergenceCriterion: float = 0.01,
            minimumElementOccurrences: int = 10,
            plotParameters: bool = False,
            printProgress: bool = False,
            mode: str = 'serial',
            max_workers: int = 8
    ) -> List[np.ndarray]:
        """Calculates ``KS2022`` descriptors corresponding to random solid solutions occupying base structure / lattice
        sites for a list of compositions through method described in ``descriptorDefinitions.KS2022_randomSolutions``
        submodule. The results are stored in the descriptorData attribute. The function returns the list of descriptors
        in numpy format as well.

        Args:
            baseStructList: The base structure to generate a random solid solution (RSS). It does _not_ need to be a simple
                Bravis lattice, such as BCC lattice, but can be any ``Structure`` object or a list of them, if you need to
                define them on per-case basis. In addition to `Structure` objects, you can use "magic" strings
                corresponding to one of the structures in the library you can find under ``pysipfenn.misc`` directory or
                loaded under ``self.prototypeLibrary`` attribute. The magic strings include, but are not limited to:
                ``'BCC'``, ``'FCC'``, ``'HCP'``, ``'DHCP'``, ``'Diamond'``, and so on. You can invoke them by their name, e.g. ``BCC``, or
                by passing ``self.prototypeLibrary['BCC']['structure']`` directly. If you pass a list to ``baseStruct``,
                you are allowed to mix-and-match ``Structure`` objects and magic strings.
            compList: The composition to populate the supercell with until KS2022 descriptor converges. You can use
                pymatgen's ``Composition`` objects or strings of valid chemical formulas (symbol - atomic fraction pairs),
                like ``'Fe0.5Ni0.3Cr0.2'``, ``'Fe50 Ni30 Cr20'``, or ``'Fe5 Ni3 Cr2'``. You can either pass a single entity, in
                which case it will be used for all structures (use to run the same composition for different base
                structures), or a list of entities, in which case pairs will be used in the order of the list. If you
                pass a list to ``compList``, you are allowed to mix-and-match ``Composition`` objects and composition
                strings.
            minimumSitesPerExpansion: The minimum number of sites that the base structure will be expanded to (doubling
                dimension-by-dimension) before it is used as expansion step/batch in each iteration of adding local
                chemical environment information to the global ensemble.
                The optimal value will depend on the number of species and their relative fractions in the composition.
                Generally, low values (<20ish) will result in a slower convergence, as some extreme local chemical
                environments will have strong influence on the global ensemble, and too high values (>150ish) will
                result in a needlessly slow computation for not-complex compositions, as at least two iterations will
                be processed. The default value is ``50`` and works well for simple cases.
            featureConvergenceCriterion: The maximum difference between any feature belonging to the current iteration
                (statistics based on the global ensemble of local chemical environments) and the previous iteration
                (before last expansion) expressed as a fraction of the maximum value of each feature found in the OQMD
                database at the time of SIPFENN creation (see ``KS2022_randomSolutions.maxFeaturesInOQMD`` array).
                The default value is ``0.01``, corresponding to 1% of the maximum value.
            compositionConvergenceCriterion: The maximum average difference between any element fraction belonging to
                the current composition (net of all expansions) and the target composition (``comp``). The default value
                is ``0.01``, corresponding to 1% deviation, which interpretation will depend on the number of elements
                in the composition.
            minimumElementOccurrences: The minimum number of times all elements must occur in the composition before it
                is considered converged. This setting prevents the algorithm from converging before very dilute elements
                like C in low-carbon steel, have had a chance to occur. The default value is ``10``.
            plotParameters: If True, the convergence history will be plotted using plotly. The default value is ``False``,
                but tracking them is recommended and will be accessible in the `metas` attribute of the Calculator under
                the key ``'RSS'``.
            printProgress: If True, the progress will be printed to the console. The default value is False.
            mode: Mode of calculation. Options are ``serial`` (default) and ``parallel``.
            max_workers: Number of workers to use in parallel mode. Defaults to ``8``.

        Returns:
            A list of ``numpy.ndarray``s containing the ``KS2022`` descriptor, just like the ordinary ``KS2022``. **Please note
            the stochastic nature of this algorithm**. The result will likely vary slightly between runs and parameters,
            so if convergence is critical, verify it with a test matrix of ``minimumSitesPerExpansion``,
            ``featureConvergenceCriterion``, and ``compositionConvergenceCriterion`` values.
        """
        # LIST-LIST: Assert that if both baseStruct and compList are lists, they have the same length
        if isinstance(baseStructList, list) and isinstance(compList, list):
            assert len(baseStructList) == len(compList), \
                'baseStruct and compList must have the same length if both are lists. If you want to use the same ' \
                'entity for all calculations, do not wrap it.'

        # STRING / STRUCT handling and extension
        if isinstance(baseStructList, str) or isinstance(baseStructList, Structure):
            baseStructList = [baseStructList]
            if isinstance(compList, list) and len(compList) > 1:
                baseStructList = baseStructList * len(compList)
        else:
            assert isinstance(baseStructList, list), 'baseStruct must be a list if it is not a string or Structure.'

        if isinstance(compList, str) or isinstance(compList, Composition):
            compList = [compList]
            if isinstance(baseStructList, list) and len(baseStructList) > 1:
                compList = compList * len(baseStructList)
        else:
            assert isinstance(compList, list), 'compList must be a list if it is not a string or Composition.'

        # LISTS of STRING / STRUCT
        for i in range(len(baseStructList)):
            assert isinstance(baseStructList[i], (str, Structure)), \
                'baseStruct must be a list of strings or Structure objects.'
            if isinstance(baseStructList[i], str):
                baseStructList[i] = string2prototype(self, baseStructList[i])

        for i in range(len(compList)):
            assert isinstance(compList[i], (str, Composition)), \
                'compList must be a list of strings or Composition objects.'
            if isinstance(compList[i], str):
                c = Composition(compList[i])
                assert c.valid, f'Unrecognized composition string: {compList}. Please provide a valid composition ' \
                                f'string, e.g. "Fe0.5Ni0.3Cr0.2", "Fe50 Ni30 Cr20", or "Fe5 Ni3 Cr2".'
                compList[i] = c

        assert len(baseStructList) == len(compList), 'baseStruct and compList must have the same length at this point.'
        pairedInputAndSettings, descList, metaList = [], [], []

        for i in range(len(baseStructList)):
            pairedInputAndSettings.append(
                (baseStructList[i],
                 compList[i],
                 minimumSitesPerExpansion,
                 featureConvergenceCriterion,
                 compositionConvergenceCriterion,
                 minimumElementOccurrences,
                 plotParameters,
                 printProgress,
                 True))

        if mode == 'serial':
            for base, comp, *settings in tqdm(pairedInputAndSettings):
                desc, meta = KS2022_randomSolutions.generate_descriptor(base, comp, *settings)
                descList.append(desc)
                metaList.append(meta)

        elif mode == 'parallel':
            print(pairedInputAndSettings)
            descList, metaList = zip(*process_map(
                wrapper_KS2022_randomSolutions_generate_descriptor,
                pairedInputAndSettings,
                max_workers=max_workers
            ))
        else:
            raise ValueError('Incorrect calculation mode selected. Must be either `serial` or `parallel`.')

        if self.verbose:
            print('Done!')
        self.descriptorData = descList
        self.metas['RSS'] = metaList
        return descList

    # *******************************  PREDICTION RUNNERS (MID-LEVEL API) *******************************
    def makePredictions(
            self,
            models: Dict[str, torch.nn.Module],
            toRun: List[str],
            dataInList: List[Union[List[float], np.array]]
    ) -> List[list]:
        """Makes predictions using PyTorch networks listed in toRun and provided in models dictionary. Shared among all
        "predict" functions.

        Args:
            models: Dictionary of models to use. Keys are network names and values are PyTorch models loaded from ONNX
                with ``loadModels()`` / ``loadModelCustom()`` or manually (fairly simple!).
            toRun: List of networks to run. It must be a subset of ``models.keys()``.
            dataInList: List of data to make predictions for. Each element of the list should be a descriptor accepted
                by all networks in toRun. Can be a list of lists of floats or a list of numpy ``nd.array``s.

        Returns:
            List of predictions. Each element of the list is a list of predictions for all run networks. The order of the
            predictions is the same as the order of the networks in ``toRun``.
        """
        dataOuts = []
        if self.verbose:
            print('Making predictions...')
        # Run for each network
        dataIn = torch.from_numpy(np.array(dataInList)).float()
        assert set(toRun).issubset(set(models.keys())), 'Some networks to run are not available in the models.'
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
            if self.verbose:
                print(f'Prediction rate: {round(len(tempOut) / (t1 - t0), 1)} pred/s')
                print(f'Obtained {len(tempOut)} predictions from:  {net}')

        # Transpose and round the predictions
        dataOuts = np.array(dataOuts).T.tolist()[0]
        self.predictions = dataOuts
        return dataOuts

    # *******************************  TOP-LEVEL API  *******************************
    def runModels(
            self,
            descriptor: str,
            structList: List[Structure],
            mode: str = 'serial',
            max_workers: int = 4
    ) -> List[List[float]]:
        """Runs all loaded models on a list of Structures using specified descriptor. Supports serial and parallel
        computation modes. If parallel is selected, max_workers determines number of processes handling the
        featurization of structures (90-99+% of computational intensity) and models are then run in series.

        Args:
            descriptor: Descriptor to use. Must be one of the available descriptors. See ``pysipfenn.descriptorDefinitions``
                to see available modules or add yours. Available default descriptors are: ``'Ward2017'``, ``'KS2022'``.
            structList: List of pymatgen Structure objects to run the models on.
            mode: Computation mode. ``'serial'`` or ``'parallel'``. Default is ``'serial'``. Parallel mode is not recommended for
                small datasets.
            max_workers: Number of workers to use in parallel mode. Default is ``4``. Ignored in serial mode. If set to
                ``None``, will use all available cores. If set to ``0``, will use ``1`` core.

        Returns:
            List of predictions. Each element of the list is a list of predictions for all ran networks. The
            order of the predictions is the same as the order of the input structures. The order of the networks is
            the same as the order of the networks in ``self.network_list_available``. If a network is not available, it
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
            self.descriptorData = self.calculate_Ward2017(
                structList=structList,
                mode=mode,
                max_workers=max_workers
            )
        elif descriptor == 'KS2022':
            self.descriptorData = self.calculate_KS2022(
                structList=structList,
                mode=mode,
                max_workers=max_workers
            )
        else:
            print('Descriptor handing not implemented. Check spelling.')
            raise AssertionError

        self.makePredictions(
            models=self.loadedModels,
            toRun=self.toRun,
            dataInList=self.descriptorData
        )

        return self.predictions

    def runModels_dilute(
            self,
            descriptor: str,
            structList: List[Structure],
            baseStruct: Union[str, List[Structure]] = 'pure',
            mode: str = 'serial',
            max_workers: int = 4
    ) -> List[List[float]]:
        """Runs all loaded models on a list of Structures using specified descriptor. A critical difference
        from runModels() is that this function will call dilute-specific featurizer, e.g. ``KS2022_dilute`` when ``'KS2022'`` is
        provided as input, which can only be used on dilute structures (both based on pure elements and on custom base
        structures, e.g. TCP endmember configurations) that contain a single alloying atom. Speed increases are
        substantial compared to the KS2022 descriptor, which is more general and can be used on any structure.
        Supports serial and parallel modes in the same way as ``runModels()``.

        Args:
            descriptor: Descriptor to use for predictions. Must be one of the descriptors which support the dilute
                structures (i.e. `*_dilute`). See ``pysipfenn.descriptorDefinitions`` to see available modules or add yours
                here. Available default dilute descriptors are now: ``'KS2022'``. The ``'KS2022'`` can also be called from
                ``runModels()`` function, but is not recommended for dilute alloys, as it negates the speed increase of the
                dilute structure featurizer.
            structList: List of pymatgen ``Structure`` objects to run the models on. Must be dilute structures as described
                above.
            baseStruct: Non-diluted references for the dilute structures. Defaults to 'pure', which assumes that the
                structures are based on pure elements and generates references automatically. Alternatively, a list of
                structures can be provided, which can be either pure elements or custom base structures (e.g. TCP
                endmember configurations).
            mode: Computation mode. ``'serial'`` or ``'parallel'``. Default is ``'serial'``. Parallel mode is not recommended for
                small datasets.
            max_workers: Number of workers to use in parallel mode. Default is ``4``. Ignored in serial mode. If set to
                ``None``, will use all available cores. If set to ``0``, will use ``1`` core.

        Returns:
            List of predictions. Each element of the list is a list of predictions for all run networks. The
            order of the predictions is the same as the order of the input structures. The order of the networks
            is the same as the order of the networks in ``self.network_list_available``. If a network is not available,
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
            self.descriptorData = self.calculate_KS2022_dilute(
                structList=structList,
                baseStruct=baseStruct,
                mode=mode,
                max_workers=max_workers
            )
        else:
            print('Descriptor handing not implemented. Check spelling.')
            raise AssertionError

        self.makePredictions(
            models=self.loadedModels,
            toRun=self.toRun,
            dataInList=self.descriptorData
        )

        return self.predictions

    def runModels_randomSolutions(
            self,
            descriptor: str,
            baseStructList: Union[str, Structure, List[str], List[Structure], List[Union[Composition, str]]],
            compList: Union[str, List[str], Composition, List[Composition], List[Union[Composition, str]]],
            minimumSitesPerExpansion: int = 50,
            featureConvergenceCriterion: float = 0.005,
            compositionConvergenceCriterion: float = 0.01,
            minimumElementOccurrences: int = 10,
            plotParameters: bool = False,
            printProgress: bool = False,
            mode: str = 'serial',
            max_workers: int = 8,
        ) -> List[List[float]]:
        """A top-level convenience wrapper for the ``calculate_KS2022_randomSolutions`` function. It passes all the
        arguments to that function directly (except for ``descriptor`` and uses its result to run all applicable models.
        The result is a list of predictions for all run networks.

        Args:
            descriptor: Descriptor to use for predictions. Must be one of the descriptors which support the random
            solid solution structures (i.e. `*_randomSolutions`). See ``pysipfenn.descriptorDefinitions`` to see
            available modules or add yours here. As of v0.15.0, the only available descriptor is
            ``'KS2022'`` through its ``KS2022_randomSolutions`` submodule.
            baseStructList: See ``calculate_KS2022_randomSolutions`` for details. You can mix-and-match ``Structure``
                objects and magic strings, either individually (to use the same entity for all calculations) or in a
                list.
            compList: See ``calculate_KS2022_randomSolutions`` for details. You can mix-and-match ``Composition``
                objects and composition strings, either individually (to use the same entity for all calculations)
                or in a list.
            minimumSitesPerExpansion: See ``calculate_KS2022_randomSolutions``.
            featureConvergenceCriterion: See ``calculate_KS2022_randomSolutions``.
            compositionConvergenceCriterion: See ``calculate_KS2022_randomSolutions``.
            minimumElementOccurrences: See ``calculate_KS2022_randomSolutions``.
            plotParameters: See ``calculate_KS2022_randomSolutions``.
            printProgress: See ``calculate_KS2022_randomSolutions``.
            mode: Computation mode. ``'serial'`` or ``'parallel'``. Default is ``'serial'``. Parallel mode is not
                recommended for small datasets.

        Returns:
            List of predictions. They will correspond to the order of the networks in ``self.toRun`` established by the
            ``findCompatibleModels()`` function. If a network is not available, it will not be included in the list.
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
            self.descriptorData = self.calculate_KS2022_randomSolutions(
                baseStructList=baseStructList,
                compList=compList,
                minimumSitesPerExpansion=minimumSitesPerExpansion,
                featureConvergenceCriterion=featureConvergenceCriterion,
                compositionConvergenceCriterion=compositionConvergenceCriterion,
                minimumElementOccurrences=minimumElementOccurrences,
                plotParameters=plotParameters,
                printProgress=printProgress,
                mode=mode,
                max_workers=max_workers
            )
        else:
            print('Descriptor handing not implemented. Check spelling.')
            raise AssertionError

        self.makePredictions(
            models=self.loadedModels,
            toRun=self.toRun,
            dataInList=self.descriptorData)

        return self.predictions

    def runFromDirectory(
            self,
            directory: str,
            descriptor: str,
            mode: str = 'serial',
            max_workers: int = 4
    ) -> List[list]:
        """Runs all loaded models on a list of Structures it automatically imports from a specified directory. The
        directory must contain only atomic structures in formats such as ``'poscar'``, ``'cif'``, ``'json'``, ``'mcsqs'``, etc.,
        or a mix of these. The structures are automatically sorted using natsort library, so the order of the
        structures in the directory, as defined by the operating system, is not important. Natural sorting,
        for example, will sort the structures in the following order: ``'1-Fe'``, ``'2-Al'``, ``'10-xx'``, ``'11-xx'``, ``'20-xx'``,
        ``'21-xx'``, ``'11111-xx'``, etc. This is useful when the structures are named using a numbering system. The order of
        the predictions is the same as the order of the input structures. The order of the networks in a prediction
        is the same as the order of the networks in ``self.network_list_available``. If a network is not available,
        it will not be included in the list.

        Args:
            directory: Directory containing the structures to run the models on. The directory must contain only atomic
                structures in formats such as ``'poscar'``, ``'cif'``, ``'json'``, ``'mcsqs'``, etc., or a mix of these. The structures
                are automatically sorted as described above.
            descriptor: Descriptor to use. Must be one of the available descriptors. See ``pysipgenn.descriptorDefinitions``
                for a list of available descriptors.
            mode: Computation mode. ``'serial'`` or ``'parallel'``. Default is ``'serial'``. Parallel mode is not recommended for
                small datasets.
            max_workers: Number of workers to use in parallel mode. Default is ``4``. Ignored in serial mode. If set to
                ``None``, will use all available cores. If set to ``0``, will use 1 core.

        Returns:
            List of predictions. Each element of the list is a list of predictions for all run networks. The order of
            the predictions is the same as the order of the input structures. The order of the networks is the same as
            the order of the networks in ``self.network_list_available``. If a network is not available, it will not be
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
        The directory must contain only atomic structures in formats such as ``'poscar'``, ``'cif'``, ``'json'``, ``'mcsqs'``, etc.,
        or a mix of these. The structures are automatically sorted using natsort library, so the order of the
        structures in the directory, as defined by the operating system, is not important. Natural sorting,
        for example, will sort the structures in the following order: ``'1-Fe'``, ``'2-Al'``, ``'10-xx'``, ``'11-xx'``, ``'20-xx'``,
        ``'21-xx'``, ``'11111-xx'``, etc. This is useful when the structures are named using a numbering system. The order of
        the predictions is the same as the order of the input structures. The order of the networks in a prediction
        is the same as the order of the networks in self.network_list_available. If a network is not available,
        it will not be included in the list.

        Args:
            directory: Directory containing the structures to run the models on. The directory must contain only atomic
                structures in formats such as ``'poscar'``, ``'cif'``, ``'json'``, ``'mcsqs'``, etc., or a mix of these. The structures
                are automatically sorted as described above. The structures must be dilute structures, i.e. they must
                contain only one alloying element.
            descriptor: Descriptor to use. Must be one of the available descriptors. See ``pysipfenn.descriptorDefinitions``
                for a list of available descriptors.
            baseStruct: Non-diluted references for the dilute structures. Defaults to ``'pure'``, which assumes that the
                structures are based on pure elements and generates references automatically. Alternatively, a list of
                structures can be provided, which can be either pure elements or custom base structures (e.g. TCP
                endmember configurations).
            mode: Computation mode. ``'serial'`` or ``'parallel'``. Default is ``'serial'``. Parallel mode is not recommended for
                small datasets.
            max_workers: Number of workers to use in parallel mode. Default is ``8``. Ignored in serial mode. If set to
                ``None``, will use all available cores. If set to ``0``, will use 1 core.

        Returns:
            List of predictions. Each element of the list is a list of predictions for all run networks. The order of
            the predictions is the same as the order of the input structures. The order of the networks is the same as
            the order of the networks in ``self.network_list_available``. If a network is not available, it will not be
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


    # *******************************  POST-PROCESSING  *******************************
    def get_resultDicts(self) -> List[dict]:
        """Returns a list of dictionaries with the predictions for each network. The keys of the dictionaries are the
        names of the networks. The order of the dictionaries is the same as the order of the input structures passed
        through ``runModels()`` functions.

        Returns:
            List of dictionaries with the predictions.
        """
        return [dict(zip(self.toRun, pred)) for pred in self.predictions]

    def get_resultDictsWithNames(self) -> List[dict]:
        """Returns a list of dictionaries with the predictions for each network. The keys of the dictionaries are the
        names of the networks and the names of the input structures. The order of the dictionaries is the same as the
        order of the input structures passed through ``runModels()`` functions. Note that this function requires
        ``self.inputFiles`` to be set, which is done automatically when using ``runFromDirectory()`` or
        ``runFromDirectory_dilute()`` but not when using ``runModels()`` or ``runModels_dilute()``, as the input structures are
        passed directly to the function and names have to be provided separately by assigning them to ``self.inputFiles``.

        Returns:
            List of dictionaries with the predictions.
        """
        assert self.inputFiles is not []
        assert len(self.inputFiles) == len(self.predictions)
        return [
            dict(zip(['name'] + self.toRun, [name] + pred))
            for name, pred in
            zip(self.inputFiles, self.predictions)]


    def writeResultsToCSV(self, file: str) -> None:
        """Writes the results to a CSV file. The first column is the name of the structure. If the ``self.inputFiles``
        attribute is populated automatically by ``runFromDirectory()`` or set manually, the names of the structures will
        be used. Otherwise, the names will be ``'1'``, ``'2'``, ``'3'``, etc. The remaining columns are the predictions for each
        network. The order of the columns is the same as the order of the networks in ``self.network_list_available``.

        Args:
            file: Name of the file to write the results to. If the file already exists, it will be overwritten. If the
                file does not exist, it will be created. The file must have a ``'.csv'`` extension to be recognized
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
        ``self.inputFiles`` attribute is populated automatically by runFromDirectory() or set manually, the names of the
        structures will be used. Otherwise, the names will be ``'1'``, ``'2'``, ``'3'``, etc. The remaining columns are the
        descriptor values. The order of the columns is the same as the order of the labels in the descriptor
        definition file.

        Args:
            descriptor: Descriptor to use. Must be one of the available descriptors. See ``pysipgenn.descriptorDefinitions``
                for a list of available descriptors, such as ``'KS2022'`` and ``'Ward2017'``. It provides the labels for the
                descriptor values.
            file: Name of the file to write the results to. If the file already exists, it will be overwritten. If the
                file does not exist, it will be created. The file must have a ``'.csv'`` extension to be recognized
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

    def destroy(self) -> None:
            """Deallocates all loaded models and clears all data from the Calculator object."""
            self.loadedModels.clear()
            self.toRun.clear()
            self.descriptorData.clear()
            self.predictions.clear()
            self.inputFiles.clear()
            gc.collect()
            print("Calculator and all loaded models deallocated. All data cleared.")
            del self


# ************************  SATELLITE FUNCTIONS  ************************
            
def ward2ks2022(ward2017: np.ndarray) -> np.ndarray:
    """Converts a ``Ward2017`` descriptor to a ``KS2022`` descriptor (which is its subset).

    Args:
        ward2017: ``Ward2017`` descriptor. Must be a 1D ``np.ndarray`` of length ``271``.

    Returns:
        ``KS2022`` descriptor array.
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

def overwritePrototypeLibrary(prototypeLibrary: dict) -> None:
    """Destructively overwrites the prototype library with a custom one. Used by the ``appendPrototypeLibrary()`` function
    to persist its changes. The other main use it to restore the default one to the original state based on a backup
    made earlier (see tests for an example)."""
    yaml_customDumper = YAML()
    yaml_customDumper.top_level_colon_align = True

    with resources.files('pysipfenn.misc').joinpath('prototypeLibrary.yaml').open('w+') as f:
        # Restructure the prototype library back to the original format of a list of dictionaries
        print(prototypeLibrary)
        prototypeList = [
            {
                'name': key,
                'origin': value['origin'],
                'POSCAR': LiteralScalarString(str(value['POSCAR']))
            }
            for key, value in prototypeLibrary.items()]
        print(prototypeList)
        # Persist the prototype library
        yaml_customDumper.dump(prototypeList, f)
        print(f'Updated prototype library persisted to {f.name}')

# *** HELPERS ***
def string2prototype(c: Calculator, prototype: str) -> Structure:
    """Converts a prototype string to a pymatgen ``Structure`` object.

    Args:
        c: ``Calculator`` object with the ``prototypeLibrary``.
        prototype: Prototype string.

    Returns:
        ``Structure`` object.
    """
    assert isinstance(prototype, str), 'Prototype string must be a string.'
    assert prototype in c.prototypeLibrary, \
        f'Unrecognized magic string for baseStruct: {prototype}. Please use one of the recognized magic ' \
        f'strings: {list(c.prototypeLibrary.keys())} or provide a Structure object.'
    s: Structure = c.prototypeLibrary[prototype]['structure']
    assert s.is_valid(), f'Invalid structure: {s}'
    return s

# *** WRAPPERS ***
def wrapper_KS2022_dilute_generate_descriptor(args):
    """Wraps the ``KS2022_dilute.generate_descriptor`` function for parallel processing."""
    return KS2022_dilute.generate_descriptor(*args)

def wrapper_KS2022_randomSolutions_generate_descriptor(args):
    """Wraps the ``KS2022_randomSolutions.generate_descriptor`` function for parallel processing."""
    return KS2022_randomSolutions.generate_descriptor(*args)
