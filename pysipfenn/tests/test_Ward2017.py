import unittest
import csv
import os
from pymatgen.core import Structure
from tqdm import tqdm
import numpy as np
from natsort import natsorted
from importlib import resources

from pysipfenn.descriptorDefinitions import Ward2017
class TestWard2017(unittest.TestCase):

    def setUp(self):
        with resources.files('pysipfenn'). \
                joinpath('tests/testCaseFiles/exampleInputFilesDescriptorTable.csv').open('r', newline='') as f:
            reader = csv.reader(f)
            self.referenceDescriptorTable = list(reader)

        self.labels = self.referenceDescriptorTable[0]
        self.testReferenceData = np.float_(self.referenceDescriptorTable[1:]).tolist()
        self.skipLabels = ['mean_WCMagnitude_Shell3']

        with resources.files('pysipfenn').joinpath('tests/testCaseFiles/exampleInputFiles/') as exampleInputsDir:
            self.exampleInputFiles = natsorted(os.listdir(exampleInputsDir))
            self.testStructures = [Structure.from_file(f'{exampleInputsDir}/{eif}') for eif in self.exampleInputFiles]

        self.functionOutput = [Ward2017.generate_descriptor(s).tolist() for s in tqdm(self.testStructures[:5])]
        with resources.files('pysipfenn'). \
                joinpath('tests/Ward2017_TestResult.csv').open('w+', newline='') as f:
            f.writelines([f'{v}\n' for v in self.functionOutput[0]])

    def test_resutls(self):
        for fo, trd, name in zip(self.functionOutput, self.testReferenceData, self.exampleInputFiles):
            for p_fo, p_trd, l in zip(fo, trd, self.labels):
                if l not in self.skipLabels:
                    with self.subTest(msg=f'Testing {l} calculated for {name}'):
                        self.assertAlmostEqual(p_fo, p_trd, places=6)


class TestWard2017Profiling(unittest.TestCase):
    '''Test the Ward2017 descriptor generation by profiling the execution time of the descriptor generation function
    for two example structures in serial and parallel (8 workers) mode.'''
    def test_serial(self):
        Ward2017.profile(test='JVASP-10001', nRuns=4)
        Ward2017.profile(test='diluteNiAlloy', nRuns=4)

    def test_parallel(self):
        Ward2017.profileParallel(test='JVASP-10001', nRuns=24)
        Ward2017.profileParallel(test='diluteNiAlloy', nRuns=24)


if __name__ == '__main__':
    unittest.main()
