import unittest
from pymatgen.core import Structure
from importlib import resources
import shutil
import pysipfenn
import pytest
import os

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true" and os.getenv("MODELS_FETCHED") != "true"


class TestCustomModel(unittest.TestCase):
    '''_Requires the models to be downloaded first._ Test loading a custom model by copying the Krajewski2020_NN24
    model to the current directory and loading it from there instead of the default location.
    '''

    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test depends on the ONNX network files")
    def setUp(self) -> None:
        '''Copies the model to CWD.'''
        with open(resources.files('pysipfenn').joinpath('modelsSIPFENN/SIPFENN_Krajewski2020_NN24.onnx'),
                  'rb') as modelForTest:
            with open('MyFunNet.onnx', 'wb') as modelForTestCopy:
                shutil.copyfileobj(modelForTest, modelForTestCopy)
        self.assertTrue(os.path.isfile('MyFunNet.onnx'))
        print('Copied model to current directory')
        self.c = pysipfenn.Calculator()
        print(self.c.network_list_available)
        print('Setup complete')

    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test depends on the ONNX network files")
    def testCalculation(self):
        '''Loads the model as custom, runs the calculation and compares the results to the original model results
        field by field.
        '''
        self.c.loadModelCustom(networkName='MyFunNet',
                               modelName='MyFunNetName',
                               descriptor='Ward2017',
                               modelDirectory='.')
        print(self.c.network_list_available)
        testFilesDir = resources.files('pysipfenn').joinpath('tests/testCaseFiles/exampleInputFiles/')
        self.c.runFromDirectory(directory=testFilesDir,
                                descriptor='Ward2017',
                                mode='serial')
        for p in self.c.get_resultDictsWithNames()[:3]:
            self.assertIn('MyFunNet', p.keys())
            self.assertAlmostEqual(p['MyFunNet'], p['SIPFENN_Krajewski2020_NN24'], places=9)

    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test depends on the ONNX network files")
    def tearDown(self) -> None:
        '''Deletes the copied model.'''
        self.c = None
        print('\nTearing down')
        os.remove('MyFunNet.onnx')
        print('Removed MyFunNet')
