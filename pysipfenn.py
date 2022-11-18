# General Imports
import os
import time
import sys
import wget
import wx
import wx.adv
import re
import numpy as np
from pymatgen.core import Structure
from datetime import datetime
import requests
import json
from concurrent import futures
from pymongo import MongoClient
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

import mxnet as mx
from mxnet import nd
from mxnet import gluon

from typing import List

# Descriptor Generators
Ward2017 = __import__('descriptorDefinitions.Ward2017', fromlist=[''])
KS2022 = __import__('descriptorDefinitions.KS2022', fromlist=[''])
KS2022_dilute = __import__('descriptorDefinitions.KS2022_dilute', fromlist=[''])
# - add new ones here if extending the code

class Calculator:
    def __init__(self):

        self.thread_pool_executor = futures.ThreadPoolExecutor(max_workers=4)
        self.process_pool_executor = futures.ProcessPoolExecutor(max_workers=12)
        self.descriptor_thread_executor = futures.ThreadPoolExecutor(max_workers=13)

        # Create a context for mxnet
        self.ctx = mx.gpu() if mx.context.num_gpus() > 0 else mx.cpu()

        # dictionary with all model information
        self.models = json.load(open('models.json'))
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
        print(f'*********  PySIPFENN Successfully Initialized  **********')




    def updateModelAvailability(self):
        all_files = os.listdir('modelsSIPFENN')
        detectedNets = []
        for net, netName in zip(self.network_list, self.network_list_names):
            if all_files.__contains__(net + '.params') and all_files.__contains__(net + '.json'):
                detectedNets.append(net)
                print('\u2714 ' + netName)
            else:
                print('\u292B ' + netName)
        self.network_list_available = detectedNets

    def downloadModels(self, network='all'):
        # Fetch all
        if network=='all':
            print('Fetching all networks!')
            for net in self.network_list:
                if net not in self.network_list_available:
                    print(f'Fetching: {net}')
                    wget.download(self.models[net]['URLjson'], f'modelsSIPFENN/{net}.json')
                    print('\nArchitecture Successfully Fetched.')
                    print('Downloading the Network Parameters. This process can take a few minutes.')
                    wget.download(self.models[net]['URLparams'], f'modelsSIPFENN/{net}.params')
                    print('\nNetwork Parameters Fetched.')
                else:
                    print(f'{net} detected on disk. Ready to use.')

            if self.network_list==self.network_list_available:
                print('All networks available!')
            else:
                print('Problem occurred.')

        # Fetch single
        elif network in self.network_list:
            print(f'Fetching: {network}')
            wget.download(self.models[network]['URLjson'], f'modelsSIPFENN/{network}.json')
            print('\nArchitecture Successfully Fetched.')
            print('Downloading the Network Parameters. This process can take a few minutes.')
            wget.download(self.models[network]['URLparams'], f'modelsSIPFENN/{network}.params')
            print('\nNetwork Parameters Fetched.')
        # Not recognized
        else:
            print('Network name not recognized')
        self.updateModelAvailability()

    def calculate_Ward2017(self, structList: List[Structure], mode='serial', max_workers=10):

        if mode=='serial':
            descList = [Ward2017.generate_descriptor(s) for s in tqdm(structList)]
            print('Done!')
            return descList
        elif mode=='parallel':
            descList = process_map(Ward2017.generate_descriptor, structList, max_workers=max_workers)
            print('Done!')
            return descList

    def calculate_KS2022(self, structList: List[Structure], mode='serial', max_workers=10):

        if mode=='serial':
            descList = [Ward2017.generate_descriptor(s) for s in tqdm(structList)]
            print('Done!')
            return descList
        elif mode=='parallel':
            descList = process_map(Ward2017.generate_descriptor, structList, max_workers=max_workers)
            print('Done!')
            return descList


    def calculate_KS2022_dilute(self, structList: List[Structure], baseStruct='pure', mode='serial', max_workers=10):
        if baseStruct=='pure' or isinstance(baseStruct, Structure):
            if mode=='serial':
                descList = [KS2022_dilute.generate_descriptor(s, baseStruct=baseStruct) for s in tqdm(structList)]
                print('Done!')
                return descList
            elif mode=='parallel':
                descList = process_map(Ward2017.generate_descriptor(baseStruct=baseStruct), structList, max_workers=max_workers)
                print('Done!')
                return descList

        elif isinstance(baseStruct, List) and len(baseStruct)==len(structList):
            if mode=='serial':
                descList = [KS2022_dilute.generate_descriptor(s, bs) for s, bs in tqdm(zip(structList, baseStruct))]
                print('Done!')
                return descList
            elif mode=='parallel':
                descList = process_map(Ward2017.generate_descriptor, structList, baseStruct, max_workers=max_workers)
                print('Done!')
                return descList

    # Create available models dictionary with loaded model neural networks
    def loadModels(self):
        for net in self.network_list_available:
            self.loadedModels.update({net: gluon.nn.SymbolBlock.imports(
                f'modelsSIPFENN/{net}.json',    # architecture file
                ['Input'],
                f'modelsSIPFENN/{net}.params',  # parameters file
                ctx=self.ctx)})

    def makePredictions(self, models, toRun, dataInList):
        dataOuts = []
        # Run for each network
        for net in toRun:
            dataIn = nd.array(dataInList)
            input = dataIn.as_in_context(self.ctx)
            model = models[net]
            tempOut = model(input)
            dataOuts.append(list(tempOut.asnumpy()))

        # Transpose the predictions
        dataOuts = np.array(dataOuts).T.tolist()[0]
        return dataOuts

    def findCompatibleModels(self, descriptor):
        compatibleList = []
        for net in self.models:
            if descriptor in self.models[net]['descriptor']:
                compatibleList.append(net)
        return compatibleList

    def runModels(self, descriptor: str, structList: list, mode='serial', max_workers=4):

        self.toRun = list(set(self.findCompatibleModels(descriptor)).union(set(self.network_list_available)))
        if len(self.toRun)==0:
            print('The list of models to run is empty. This may be caused by selecting a descriptor not defined/available, '
                  'or if the selected descriptor does not correspond to any available network. Check spelling and invoke'
                  'the downloadModels() function if you are using base models.')
            raise AssertionError

        if descriptor=='Ward2017':
            self.descriptorData = self.calculate_Ward2017(structList, mode=mode, max_workers=max_workers)
        elif descriptor=='KS2022':
            self.descriptorData = self.calculate_KS2022(structList, mode=mode, max_workers=max_workers)
        else:
            print('Descriptor handing not implemented. Check spelling.')
            raise AssertionError

        self.predictions = self.makePredictions(models=self.loadedModels, toRun=self.toRun, dataInList=self.descriptorData)

        return self.predictions


    def runModels_dilute(self, descriptor: str, structList: list, baseStruct = 'pure', mode='serial', max_workers=4):

        self.toRun = list(set(self.findCompatibleModels(descriptor)).union(set(self.network_list_available)))
        if len(self.toRun)==0:
            print('The list of models to run is empty. This may be caused by selecting a descriptor not defined/available, '
                  'or if the selected descriptor does not correspond to any available network. Check spelling and invoke'
                  'the downloadModels() function if you are using base models.')
            raise TypeError

        if descriptor=='KS2022_dilute':
            self.descriptorData = self.calculate_KS2022_dilute(structList, baseStruct=baseStruct, mode=mode, max_workers=max_workers)
        else:
            print('Descriptor handing not implemented. Check spelling.')
            raise AssertionError

        self.predictions = self.makePredictions(models=self.loadedModels, toRun=self.toRun, dataInList=self.descriptorData)

        return self.predictions

    def get_resultDicts(self):
        return [dict(zip(self.toRun, pred)) for pred in self.predictions]

    
