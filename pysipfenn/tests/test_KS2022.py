import unittest
import csv
import os
from pymatgen.core import Structure
from tqdm import tqdm
import numpy as np
from natsort import natsorted
from importlib import resources

from pysipfenn.descriptorDefinitions import KS2022

with resources.files('pysipfenn').\
        joinpath('tests/testCaseFiles/exampleInputFilesDescriptorTable.csv').open('r', newline='') as f:
    reader = csv.reader(f)
    referenceDescriptorTable = list(reader)

labels = referenceDescriptorTable[0]
testReferenceData = np.float_(referenceDescriptorTable[1:]).tolist()
emptyLabels = [
    'mean_WCMagnitude_Shell1', 'mean_WCMagnitude_Shell2', 'mean_WCMagnitude_Shell3',
    'mean_NeighDiff_shell1_SpaceGroupNumber', 'var_NeighDiff_shell1_SpaceGroupNumber',
    'min_NeighDiff_shell1_SpaceGroupNumber', 'max_NeighDiff_shell1_SpaceGroupNumber',
    'range_NeighDiff_shell1_SpaceGroupNumber', 'mean_SpaceGroupNumber', 'maxdiff_SpaceGroupNumber',
    'dev_SpaceGroupNumber', 'max_SpaceGroupNumber', 'min_SpaceGroupNumber', 'most_SpaceGroupNumber',
    'CanFormIonic']
emptyLabels.reverse()
emptyLabelsIndx = [labels.index(l) for l in emptyLabels]

with resources.files('pysipfenn').joinpath('tests/testCaseFiles/exampleInputFiles/') as exampleInputsDir:
    exampleInputFiles = natsorted(os.listdir(exampleInputsDir))
    testStructures = [Structure.from_file(f'{exampleInputsDir}/{eif}') for eif in exampleInputFiles]

functionOutput = [KS2022.generate_descriptor(s).tolist() for s in tqdm(testStructures[:25])]
with resources.files('pysipfenn').joinpath('tests/KS2022_TestResult.csv').open('w+', newline='') as f:
    f.writelines([f'{v}\n' for v in functionOutput[0]])

class TestKS2022(unittest.TestCase):
    def test_resutls(self):
        for fo, trd, name in zip(functionOutput, testReferenceData, exampleInputFiles):
            for eli in emptyLabelsIndx:
                trd.pop(eli)
            for p_fo, p_trd, l in zip(fo, trd, labels):
                if p_trd!=0 and p_fo!=0:
                    p_fo_relative = p_fo/p_trd
                    with self.subTest(msg=f'{name:<16} diff in {l}'):
                        self.assertAlmostEqual(p_fo_relative, 1, places=2)
                else:
                    with self.subTest(msg=f'{name:<16} diff in {l}'):
                        self.assertAlmostEqual(p_fo, p_trd, places=6)


if __name__ == '__main__':
    unittest.main()
