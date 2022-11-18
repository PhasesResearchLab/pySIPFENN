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

thread_pool_executor = futures.ThreadPoolExecutor(max_workers=4)
process_pool_executor = futures.ProcessPoolExecutor(max_workers=12)
descriptor_thread_executor = futures.ThreadPoolExecutor(max_workers=13)

models = json.load(open('models.json'))

network_list = list(models.keys())
network_list_names = [models[net]['name'] for net in network_list]

def updateModelAvailability():
    all_files = os.listdir('modelsSIPFENN')
    detectedNets = []
    for net, netName in zip(network_list, network_list_names):
        if all_files.__contains__(net + '.params') and all_files.__contains__(net + '.json'):
            detectedNets.append(net)
            print('\u2714 ' + netName)
        else:
            print('\u292B ' + netName)
    return detectedNets

network_list_available = updateModelAvailability()

def downloadModels(network='all'):
    # Fetch all
    if network=='all':
        print('Fetching all networks!')
        for net in network_list:
            if net not in network_list_available:
                print(f'Fetching: {net}')
                wget.download(models[net]['URLjson'], f'modelsSIPFENN/{net}.json')
                print('\nArchitecture Successfully Fetched.')
                print('Downloading the Network Parameters. This process can take a few minutes.')
                wget.download(models[net]['URLparams'], f'modelsSIPFENN/{net}.params')
                print('\nNetwork Parameters Fetched.')
            else:
                print(f'{net} detected on disk. Ready to use.')

        if network_list==network_list_available:
            print('All networks available!')
        else:
            print('Problem occurred.')

    # Fetch single
    elif network in network_list:
        print(f'Fetching: {network}')
        wget.download(models[network]['URLjson'], f'modelsSIPFENN/{network}.json')
        print('\nArchitecture Successfully Fetched.')
        print('Downloading the Network Parameters. This process can take a few minutes.')
        wget.download(models[network]['URLparams'], f'modelsSIPFENN/{network}.params')
        print('\nNetwork Parameters Fetched.')
    # Not recognized
    else:
        print('Network name not recognized')

#downloadModels()
network_list_available = updateModelAvailability()

def calculate_Ward2017(structList: List[Structure], mode='serial', max_workers=10):

    if mode=='serial':
        descList = [Ward2017.generate_descriptor(s) for s in tqdm(structList)]
        print('Done!')
        return descList
    elif mode=='parallel':
        descList = process_map(Ward2017.generate_descriptor, structList, max_workers=max_workers)
        print('Done!')
        return descList

def calculate_KS2022(structList: List[Structure], mode='serial', max_workers=10):

    if mode=='serial':
        descList = [Ward2017.generate_descriptor(s) for s in tqdm(structList)]
        print('Done!')
        return descList
    elif mode=='parallel':
        descList = process_map(Ward2017.generate_descriptor, structList, max_workers=max_workers)
        print('Done!')
        return descList


def calculate_KS2022_dilute(structList: List[Structure], baseStruct='pure', mode='serial', max_workers=10):
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

# Create a context for mxnet
ctx = mx.gpu() if mx.context.num_gpus() > 0 else mx.cpu()

# Create available models dictionary with loaded model neural networks
loadedModels = {}
def loadModels():
    for net in network_list_available:
        loadedModels.update({net: gluon.nn.SymbolBlock.imports(
            f'modelsSIPFENN/{net}.json',    # architecture file
            ['Input'],
            f'modelsSIPFENN/{net}.params',  # parameters file
            ctx=ctx)})

loadModels()

def makePredictions(models, toRun, dataIn):
    dataOuts = []
    for net in toRun:
        input = dataIn.as_in_context(ctx)
        model = models[net]
        tempOut = model(input)
        dataOuts.append(list(tempOut.asnumpy()))
    return dataOuts

def findCompatibleModels(descriptor):
    compatibleList = []
    for net in models:
        if descriptor in models[net]['descriptor']:
            compatibleList.append(net)
    return compatibleList


def runModels(descriptor: str, structList: list, mode='serial', max_workers=4):

    toRun = set(findCompatibleModels(descriptor)).union(set(network_list_available))
    if len(toRun)==0:
        print('The list of models to run is empty. This may be caused by selecting a descriptor not defined/available, '
              'or if the selected descriptor does not correspond to any available network. Check spelling and invoke'
              'the downloadModels() function if you are using base models.')
        raise AssertionError

    if descriptor=='Ward2017':
        dataIn = calculate_Ward2017(structList, mode=mode, max_workers=max_workers)
    elif descriptor=='KS2022':
        dataIn = calculate_KS2022(structList, mode=mode, max_workers=max_workers)
    else:
        print('Descriptor handing not implemented. Check spelling.')
        raise AssertionError

    return makePredictions(models=loadedModels, toRun=toRun, dataIn=dataIn)


def runModels_dilute(descriptor: str, structList: list, baseStruct = 'pure', mode='serial', max_workers=4):

    toRun = set(findCompatibleModels(descriptor)).union(set(network_list_available))
    if len(toRun)==0:
        print('The list of models to run is empty. This may be caused by selecting a descriptor not defined/available, '
              'or if the selected descriptor does not correspond to any available network. Check spelling and invoke'
              'the downloadModels() function if you are using base models.')
        raise TypeError

    if descriptor=='KS2022_dilute':
        dataIn = calculate_KS2022_dilute(structList, baseStruct=baseStruct, mode=mode, max_workers=max_workers)
    else:
        print('Descriptor handing not implemented. Check spelling.')
        raise AssertionError

    return makePredictions(models=loadedModels, toRun=toRun, dataIn=dataIn)

