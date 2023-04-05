import unittest
import pytest
import os

import pysipfenn
from importlib import resources

from pymatgen.core import Structure

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true" and os.getenv("MODELS_FETCHED") != "true"

class TestCore(unittest.TestCase):
    def setUp(self):
        '''Initialise the Calculator object for testing. It will be used in all tests and is not modified in any way
        by them.'''
        self.c = pysipfenn.Calculator()
        self.assertIsNotNone(self.c)

    def testInit(self):
        '''Test that the Calculator object is initialised correctly.'''
        self.assertEqual(self.c.predictions, [])
        self.assertEqual(self.c.toRun, [])
        self.assertEqual(self.c.descriptorData, [])
        self.assertEqual(self.c.inputFiles, [])

    def detectModels(self):
        '''Test that the updateModelAvailability() method works without errors and returns a list of available models.
        '''
        self.c.updateModelAvailability()
        self.assertIsInstance(self.c.network_list_available, list)

    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test depends on the ONNX network files")
    def testDownloadAndLoadModels(self):
        '''Tests that the downloadModels() method works without errors in a case whwere the models are not already
        downloaded and loads them correctly using the loadModels() method. Then also load a model explicitly using
        loadModel() and check that it is in the loadedModels list. Also check that arror is raised correctly if
        a non-available model is requested to be loaded.'''

        self.c.downloadModels(network='all')
        self.c.loadModels(network='SIPFENN_Krajewski2020_NN24')

        self.assertEqual(set(self.c.network_list_available), set(self.c.loadedModels.keys()))
        self.assertIn('SIPFENN_Krajewski2020_NN24', self.c.loadedModels)

        with self.assertRaises(ValueError):
            self.c.loadModels(network='jx9348ghfmx8345wgyf')

    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test depends on the ONNX network files")
    def testFromPOSCAR_Ward2017(self):
        self.c.updateModelAvailability()
        toRun = list(set(self.c.findCompatibleModels('Ward2017')).intersection(set(self.c.network_list_available)))
        if toRun!=[]:
            with resources.files('pysipfenn').joinpath('tests/testCaseFiles/exampleInputFiles') as testFileDir:
                print(testFileDir)
                self.c.runFromDirectory(testFileDir, 'Ward2017')
        else:
            print('Did not detect any Ward2017 models to run')

    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test depends on the ONNX network files")
    def testFromPOSCAR_KS2022(self):
        self.c.updateModelAvailability()
        toRun = list(set(self.c.findCompatibleModels('KS2022')).intersection(set(self.c.network_list_available)))
        if toRun!=[]:
            with resources.files('pysipfenn').joinpath('tests/testCaseFiles/exampleInputFiles') as testFileDir:
                print(testFileDir)
                self.c.runFromDirectory(testFileDir, 'KS2022')
        else:
            print('Did not detect any KS2022 models to run')

    def test_descriptorCalculate_Ward2017_serial(self):
        '''Test succesful execution of the descriptorCalculate() method with Ward2017 in series. A separate test for
        calculation accuracy is done in test_Ward2017.py'''
        with resources.files('pysipfenn').joinpath('tests/testCaseFiles/exampleInputFiles') as exampleInputsDir:
            exampleInputFiles = os.listdir(exampleInputsDir)[:6]
            testStructures = [Structure.from_file(f'{exampleInputsDir}/{eif}') for eif in exampleInputFiles]
            descList = self.c.calculate_Ward2017(structList=testStructures, mode='serial')
            self.assertEqual(len(descList), len(testStructures))

    def test_descriptorCalculate_Ward2017_parallel(self):
        '''Test succesful execution of the descriptorCalculate() method with Ward2017 in parallel. A separate test for
        calculation accuracy is done in test_Ward2017.py'''
        with resources.files('pysipfenn').joinpath('tests/testCaseFiles/exampleInputFiles') as exampleInputsDir:
            exampleInputFiles = os.listdir(exampleInputsDir)[:6]
            testStructures = [Structure.from_file(f'{exampleInputsDir}/{eif}') for eif in exampleInputFiles]
            descList = self.c.calculate_Ward2017(structList=testStructures, mode='parallel', max_workers=2)
            self.assertEqual(len(descList), len(testStructures))

    def test_descriptorCalculate_KS2022_serial(self):
        '''Test succesful execution of the descriptorCalculate() method with KS2022 in series. A separate test for
        calculation accuracy is done in test_KS2022.py'''
        with resources.files('pysipfenn').joinpath('tests/testCaseFiles/exampleInputFiles') as exampleInputsDir:
            exampleInputFiles = os.listdir(exampleInputsDir)
            testStructures = [Structure.from_file(f'{exampleInputsDir}/{eif}') for eif in exampleInputFiles]
            descList = self.c.calculate_KS2022(structList=testStructures, mode='serial')
            self.assertEqual(len(descList), len(testStructures))

    def test_descriptorCalculate_KS2022_parallel(self):
        '''Test succesful execution of the descriptorCalculate() method with KS2022 in parallel. A separate test for
        calculation accuracy is done in test_KS2022.py'''
        with resources.files('pysipfenn').joinpath('tests/testCaseFiles/exampleInputFiles') as exampleInputsDir:
            exampleInputFiles = os.listdir(exampleInputsDir)
            testStructures = [Structure.from_file(f'{exampleInputsDir}/{eif}') for eif in exampleInputFiles]
            descList = self.c.calculate_KS2022(structList=testStructures, mode='parallel', max_workers=4)
            self.assertEqual(len(descList), len(testStructures))


if __name__ == '__main__':
    unittest.main()
