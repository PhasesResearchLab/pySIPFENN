import unittest
import pytest
import os
from importlib import resources

import pysipfenn

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true" and os.getenv("MODELS_FETCHED") != "true"

class TestAllCompatibleONNX_Ward2017(unittest.TestCase):
    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test depends on the ONNX network files")
    def test_runtime(self):
        c = pysipfenn.Calculator()
        with resources.files('pysipfenn').joinpath('tests/testCaseFiles/exampleInputFiles/') as exampleInputsDir:
            c.runFromDirectory(directory=exampleInputsDir, descriptor='Ward2017')
        print(c.get_resultDicts())
        c.writeResultsToCSV('Ward2017-ONNX_testResults.csv')

if __name__ == '__main__':
    unittest.main()