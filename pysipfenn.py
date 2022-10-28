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

# Descriptor Generators
Ward2017 = __import__('descriptorDefinitions.Ward2017', fromlist=[''])

thread_pool_executor = futures.ThreadPoolExecutor(max_workers=4)
process_pool_executor = futures.ProcessPoolExecutor(max_workers=12)
descriptor_thread_executor = futures.ThreadPoolExecutor(max_workers=13)

models = json.load(open('models.json'))

network_list = models.keys()
network_list_names = [models[net]['name'] for net in network_list]
network_list_available = []

def updateModelAvailability():
    all_files = os.listdir('modelsSIPFENN')
    network_list_available = []
    for net, netName in zip(network_list, network_list_names):
        if all_files.__contains__(net + '.params') and all_files.__contains__(net + '.json'):
            network_list_available.append(net)
            print('\u2714 ' + netName)
        else:
            print('\u292B ' + netName)

updateModelAvailability()

def downloadModels(network='all'):
    # Fetch all
    if network=='all':
        print('Fetching all networks!')
        for net in network_list:
            print(f'Fetching: {net}')
            wget.download(models[net]['URLjson'], f'modelsSIPFENN/{net}.json')
            print('Architecture Successfully Fetched.')
            print('Downloading the Network Parameters. This process can take a few minutes.')
            wget.download(models[net]['URLparams'], f'modelsSIPFENN/{net}.params')
            print('Network Parameters Fetched.')
        updateModelAvailability()

        if network_list==network_list_available:
            print('All networks available!')
        else:
            print('Problem occurred.')
    # Fetch single
    elif network in network_list:
        print(f'Fetching: {network}')
        wget.download(models[network]['URLjson'], f'modelsSIPFENN/{network}.json')
        print('Architecture Successfully Fetched.')
        print('Downloading the Network Parameters. This process can take a few minutes.')
        wget.download(models[network]['URLparams'], f'modelsSIPFENN/{network}.params')
        print('Network Parameters Fetched.')
        updateModelAvailability()
    # Not recognized
    else:
        print('Network name not recognized')

downloadModels()
