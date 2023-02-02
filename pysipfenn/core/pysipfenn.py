# General Imports
import os

import natsort
import wget
import csv
import numpy as np
from pymatgen.core import Structure
import json
from concurrent import futures
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from time import perf_counter

from importlib import resources

import torch
import onnx2torch
import onnx

from typing import List

# Descriptor Generators
from pysipfenn.descriptorDefinitions import Ward2017, KS2022, KS2022_dilute
# - add new ones here if extending the code


class Calculator:
    def __init__(self):

        self.thread_pool_executor = futures.ThreadPoolExecutor(max_workers=4)
        self.process_pool_executor = futures.ProcessPoolExecutor(max_workers=12)
        self.descriptor_thread_executor = futures.ThreadPoolExecutor(max_workers=13)

        # dictionary with all model information
        with resources.files('pysipfenn.modelsSIPFENN').joinpath('models.json').open('r') as f:
            self.models = json.load(f)
        # networks list
        self.network_list = list(self.models.keys())
        # network names
        self.network_list_names = [self.models[net]['name'] for net in self.network_list]
        self.network_list_available = []
        self.updateModelAvailability()

        self.loadedModels = {}
        self.loadModels()

        self.toRun = []
        self.descriptorData = []
        self.predictions = []
        self.inputFiles = []
        print(f'*********  PySIPFENN Successfully Initialized  **********')

    def updateModelAvailability(self):
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

    def downloadModels_legacyMxNet(self, network='all'):
        with resources.files('pysipfenn.modelsSIPFENN') as modelPath:
            # Fetch all
            if network == 'all':
                print('Fetching all networks!')
                for net in self.network_list:
                    if net not in self.network_list_available:
                        print(f'Fetching: {net}')
                        wget.download(self.models[net]['URLjson'], f'{modelPath}/{net}.json')
                        print('\nArchitecture Successfully Fetched.')
                        print('Downloading the Network Parameters. This process can take a few minutes.')
                        wget.download(self.models[net]['URLparams'], f'{modelPath}/{net}.params')
                        print('\nNetwork Parameters Fetched.')
                    else:
                        print(f'{net} detected on disk. Ready to use.')

                if self.network_list == self.network_list_available:
                    print('All networks available!')
                else:
                    print('Problem occurred.')

            # Fetch single
            elif network in self.network_list:
                print(f'Fetching: {network}')
                wget.download(self.models[network]['URLjson'], f'{modelPath}/{network}.json')
                print('\nArchitecture Successfully Fetched.')
                print('Downloading the Network Parameters. This process can take a few minutes.')
                wget.download(self.models[network]['URLparams'], f'{modelPath}/{network}.params')
                print('\nNetwork Parameters Fetched.')
            # Not recognized
            else:
                print('Network name not recognized')

    def downloadModels(self, network='all'):
        with resources.files('pysipfenn.modelsSIPFENN') as modelPath:
            # Fetch all
            if network == 'all':
                print('Fetching all networks!')
                for net in self.network_list:
                    if net not in self.network_list_available:
                        if 'URL_ONNX' in self.models[net]:
                            print(f'Fetching: {net}')
                            wget.download(self.models[net]['URL_ONNX'], f'{modelPath}/{net}.onnx')
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
                wget.download(self.models[network]['URL_ONNX'], f'{modelPath}/{network}.onnx')
                print('\nONNX Network Successfully Fetched.')
            # Not recognized
            else:
                print('Network name not recognized')
        self.updateModelAvailability()

    def calculate_Ward2017(self, structList: List[Structure], mode='serial', max_workers=10):

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

    def calculate_KS2022(self, structList: List[Structure], mode='serial', max_workers=10):

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

    def calculate_KS2022_dilute(self, structList: List[Structure], baseStruct='pure', mode='serial', max_workers=10):
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

    # Create available models dictionary with loaded model neural networks
    def loadModels(self):
        with resources.files('pysipfenn.modelsSIPFENN') as modelPath:
            print('Loading models:')
            for net in tqdm(self.network_list_available):
                self.loadedModels.update({
                    net: onnx2torch.convert(onnx.load(f'{modelPath}/{net}.onnx')).float()
                })

    def makePredictions_legacyMxNet(self, mxnet_networks, dataInList):
        # Import MxNet
        import mxnet as mx
        from mxnet import nd
        from mxnet import gluon
        # Create a context for mxnet
        self.ctx = mx.gpu() if mx.context.num_gpus() > 0 else mx.cpu()
        # Verify if the nets are available
        with resources.files('pysipfenn.modelsSIPFENN') as p:
            all_files = os.listdir(p)
        for net in mxnet_networks:
            assert all_files.__contains__(net + '.json')
            assert all_files.__contains__(net + '.params')
        # Load the models
        loadedModels = {}
        with resources.files('pysipfenn.modelsSIPFENN') as modelPath:
            for net in mxnet_networks:
                loadedModels.update({net: gluon.nn.SymbolBlock.imports(
                    f'{modelPath}/{net}.json',    # architecture file
                    ['Input'],
                    f'{modelPath}/{net}.params',  # parameters file
                    ctx=self.ctx)})
        dataOuts = []
        print('Making predictions...')
        # Run for each network
        for net in loadedModels:
            dataIn = nd.array(dataInList)
            dataInCTX = dataIn.as_in_context(self.ctx)
            model = loadedModels[net]
            tempOut = model(dataInCTX)
            dataOuts.append(list(tempOut.asnumpy()))
            print(f'Obtained predictions from:  {net}')

        # Transpose the predictions
        dataOuts = np.array(dataOuts).T.tolist()[0]

        self.predictions = dataOuts
        return dataOuts

    def makePredictions(self, models, toRun, dataInList):
        dataOuts = []
        print('Making predictions...')
        # Run for each network
        dataIn = torch.from_numpy(np.array(dataInList)).float()
        for net in toRun:
            t0 = perf_counter()
            model = models[net]
            model.eval()
            if hasattr(model, 'Dropout_0'):
                tempOut = model(dataIn, None)
            else:
                tempOut = model(dataIn)
            t1 = perf_counter()
            dataOuts.append(tempOut.cpu().detach().numpy())
            print(f'Prediction rate: {round(len(tempOut)/(t1-t0), 1)} pred/s')
            print(f'Obtained {len(tempOut)} predictions from:  {net}')

        # Transpose and round the predictions
        dataOuts = np.array(dataOuts).T.tolist()[0]
        self.predictions = dataOuts
        return dataOuts

    def findCompatibleModels(self, descriptor):
        compatibleList = []
        for net in self.models:
            if descriptor in self.models[net]['descriptor']:
                compatibleList.append(net)
        return compatibleList

    def runModels(self, descriptor: str, structList: list, mode='serial', max_workers=4):

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

    def runModels_dilute(self, descriptor: str, structList: list, baseStruct='pure', mode='serial', max_workers=4):

        self.toRun = list(set(self.findCompatibleModels(descriptor)).intersection(set(self.network_list_available)))
        if len(self.toRun) == 0:
            print('The list of models to run is empty. This may be caused by selecting a descriptor not '
                  'defined/available, or if the selected descriptor does not correspond to any available network. '
                  'Check spelling and invoke the downloadModels() function if you are using base models.')
            raise TypeError
        else:
            print(f'Running {self.toRun} models')

        print('Calculating descriptors...')
        if descriptor == 'KS2022_dilute':
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

    def get_resultDicts(self):
        return [dict(zip(self.toRun, pred)) for pred in self.predictions]

    def get_resultDictsWithNames(self):
        assert self.inputFiles is not []
        return [
            dict(zip(['name']+self.toRun, [name]+pred))
            for name, pred in
            zip(self.inputFiles, self.predictions)]

    def runFromDirectory(self, directory: str, descriptor: str, mode='serial', max_workers=4):
        print('Importing structures...')
        self.inputFiles = os.listdir(directory)
        self.inputFiles = natsort.natsorted(self.inputFiles)
        structList = [Structure.from_file(f'{directory}/{eif}') for eif in tqdm(self.inputFiles)]
        self.runModels(descriptor=descriptor, structList=structList, mode=mode, max_workers=max_workers)
        print('Done!')

    def runFromDirectory_dilute(self, directory: str, descriptor: str, baseStruct='pure', mode='serial', max_workers=4):
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

    def writeResultsToCSV(self, file: str):
        assert self.toRun is not []
        with open(file, 'w+', encoding="utf-8") as f:
            f.write('Name,'+','.join(self.toRun)+'\n')
            if len(self.inputFiles) == len(self.predictions):
                for name, pred in zip(self.inputFiles, self.predictions):
                    assert len(pred) == len(self.toRun)
                    f.write(f'{name},{",".join(str(v) for v in pred)}\n')
            else:
                i = 1
                for pred in self.predictions:
                    f.write(f'{i},{",".join(str(v) for v in pred)}\n')
                    i += 1

    def writeDescriptorsToCSV(self, descriptor: str, file: str):
        # Load descriptor labels
        with open(f'descriptorDefinitions/labels_{descriptor}.csv', 'r') as f:
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


def ward2ks2022(ward2017: np.ndarray):
    assert isinstance(ward2017, np.ndarray)
    ward2017split = np.split(ward2017, [12, 15, 121, 126, 258, 264, 268, 269, 271])
    ks2022 = np.concatenate((
        ward2017split[0],
        ward2017split[2],
        ward2017split[4],
        ward2017split[6],
        ward2017split[8]
        ), axis=-1, dtype=np.float32)

    return ks2022
