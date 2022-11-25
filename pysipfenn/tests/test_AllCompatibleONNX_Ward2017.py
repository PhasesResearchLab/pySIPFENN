import unittest
from importlib import resources

import pysipfenn

class TestAllCompatibleONNX_Ward2017(unittest.TestCase):
    def test_runtime(self):
        c = pysipfenn.Calculator()
        with resources.files('pysipfenn').joinpath('tests/testCaseFiles/exampleInputFiles/') as exampleInputsDir:
            c.runFromDirectory(directory=exampleInputsDir, descriptor='Ward2017')
        print(c.get_resultDicts())
        c.writeResultsToCSV('Ward2017-ONNX_testResults.csv')

if __name__ == '__main__':
    unittest.main()