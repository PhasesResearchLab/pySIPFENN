import unittest
import pytest
import os

import pysipfenn
from importlib import resources

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true" and os.getenv("MODELS_FETCHED") != "true"

class TestCore(unittest.TestCase):
    def setUp(self):
        self.c = pysipfenn.Calculator()
        self.assertIsNotNone(self.c)

    def testInit(self):
        self.assertEqual(self.c.predictions, [])
        self.assertEqual(self.c.toRun, [])
        self.assertEqual(self.c.descriptorData, [])
        self.assertEqual(self.c.inputFiles, [])

    def detectModels(self):
        self.c.updateModelAvailability()
        self.assertIsInstance(self.c.network_list_available, list)

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

if __name__ == '__main__':
    unittest.main()
