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
    '''Tests the correctness of the KS2022 descriptor generation function by comparing the results to the reference data
    for the first 25 structures in the exampleInputFiles directory, stored in the exampleInputFilesDescriptorTable.csv.
    That file that is also used to test the correctness of the Ward2017, which is a superset of the KS2022.
    '''
    def setUp(self):
        '''Reads the reference data from the exampleInputFilesDescriptorTable.csv file and the labels from the first
        row of that file. Then it reads the first 25 structures from the exampleInputFiles directory and generates the
        descriptors for them. The results are stored in the functionOutput list. It defines the emptyLabelsIndx list
        that contains the indices of the labels that are not used in the KS2022 (vs Ward2017) descriptor generation. It
        also persists the test results in the KS2022_TestResult.csv file.
        '''
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
        '''Compares the results of the KS2022 descriptor generation function to the reference data on a field-by-field
        basis by calculating the relative difference between the two and requiring it to be less than 1% for all fields
        except 0-valued fields, where the absolute difference is required to be less than 1e-6.
        '''
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

    def test_cite(self):
        """Tests citation return."""
        citation = KS2022.cite()
        self.assertIn("Krajewski", citation[0])

class TestKS2022Profiling(unittest.TestCase):
    '''Test the KS2022 descriptor generation by profiling the execution time of the descriptor generation function
    for two example structures (JVASP-10001 and diluteNiAlloy).
    '''
    def test_serial(self):
        '''Test the serial execution of the descriptor generation function 4 times for each of the two examples.'''
        KS2022.profile(test='JVASP-10001', nRuns=4)
        KS2022.profile(test='diluteNiAlloy', nRuns=4)

    def test_parallel(self):
        '''Test the parallel execution of the descriptor generation function 24 times for each of the two examples
        but in parallel with up to 8 workers to speed up the execution.
        '''
        KS2022.profileParallel(test='JVASP-10001', nRuns=24)
        KS2022.profileParallel(test='diluteNiAlloy', nRuns=24)
