import unittest
import csv
import os
from pymatgen.core import Structure
from tqdm import tqdm
import numpy as np
from natsort import natsorted
from importlib import resources

from pysipfenn.descriptorDefinitions import KS2022

class TestKS2022(unittest.TestCase):
    def setUp(self):
        with resources.files('pysipfenn'). \
                joinpath('tests/testCaseFiles/exampleInputFilesDescriptorTable.csv').open('r', newline='') as f:
            reader = csv.reader(f)
            referenceDescriptorTable = list(reader)

        self.labels = referenceDescriptorTable[0]
        self.testReferenceData = np.float_(referenceDescriptorTable[1:]).tolist()
        emptyLabels = [
            'mean_WCMagnitude_Shell1', 'mean_WCMagnitude_Shell2', 'mean_WCMagnitude_Shell3',
            'mean_NeighDiff_shell1_SpaceGroupNumber', 'var_NeighDiff_shell1_SpaceGroupNumber',
            'min_NeighDiff_shell1_SpaceGroupNumber', 'max_NeighDiff_shell1_SpaceGroupNumber',
            'range_NeighDiff_shell1_SpaceGroupNumber', 'mean_SpaceGroupNumber', 'maxdiff_SpaceGroupNumber',
            'dev_SpaceGroupNumber', 'max_SpaceGroupNumber', 'min_SpaceGroupNumber', 'most_SpaceGroupNumber',
            'CanFormIonic']
        emptyLabels.reverse()
        self.emptyLabelsIndx = [self.labels.index(l) for l in emptyLabels]

        with resources.files('pysipfenn').joinpath('tests/testCaseFiles/exampleInputFiles/') as exampleInputsDir:
            self.exampleInputFiles = natsorted(os.listdir(exampleInputsDir))
            testStructures = [Structure.from_file(f'{exampleInputsDir}/{eif}') for eif in self.exampleInputFiles]

        self.functionOutput = [KS2022.generate_descriptor(s).tolist() for s in tqdm(testStructures[:25])]
        with resources.files('pysipfenn').joinpath('tests/KS2022_TestResult.csv').open('w+', newline='') as f:
            f.writelines([f'{v}\n' for v in self.functionOutput[0]])
    def test_resutls(self):
        for fo, trd, name in zip(self.functionOutput, self.testReferenceData, self.exampleInputFiles):
            for eli in self.emptyLabelsIndx:
                trd.pop(eli)
            for p_fo, p_trd, l in zip(fo, trd, self.labels):
                if p_trd!=0 and p_fo!=0:
                    p_fo_relative = p_fo/p_trd
                    with self.subTest(msg=f'{name:<16} diff in {l}'):
                        self.assertAlmostEqual(p_fo_relative, 1, places=2)
                else:
                    with self.subTest(msg=f'{name:<16} diff in {l}'):
                        self.assertAlmostEqual(p_fo, p_trd, places=6)

class TestKS2022Profiling(unittest.TestCase):
    '''Test the KS2022 descriptor generation by profiling the execution time of the descriptor generation function
        for two example structures in serial and parallel (8 workers) mode.'''
    def test_serial(self):
        KS2022.profile(test='JVASP-10001', nRuns=4)
        KS2022.profile(test='diluteNiAlloy', nRuns=4)

    def test_parallel(self):
        KS2022.profileParallel(test='JVASP-10001', nRuns=24)
        KS2022.profileParallel(test='diluteNiAlloy', nRuns=24)


if __name__ == '__main__':
    unittest.main()
