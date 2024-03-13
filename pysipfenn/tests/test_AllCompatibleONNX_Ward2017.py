import unittest
import pytest
import os
from importlib import resources

import pysipfenn

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true" and os.getenv("MODELS_FETCHED") != "true"

class TestAllCompatibleONNX_Ward2017(unittest.TestCase):
    '''_Requires the models to be downloaded first._ It then tests the **runtime** of the pySIPFENN on all POSCAR
    files in the exampleInputFiles directory and persistence of the results in a CSV file.
    '''
    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test depends on the ONNX network files")
    def test_runtime(self):
        '''Runs the test.'''
        c = pysipfenn.Calculator()
        with resources.files('pysipfenn').joinpath('tests/testCaseFiles/exampleInputFiles/') as exampleInputsDir:
            c.runFromDirectory(directory=exampleInputsDir, descriptor='Ward2017')
        print(c.get_resultDicts())
        c.writeResultsToCSV('Ward2017-ONNX_testResults.csv')
        