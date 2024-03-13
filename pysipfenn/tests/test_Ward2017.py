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
    '''Tests the correctness of the KS2022 descriptor generation function by comparing the results to the reference data
    for the first 5 structures in the exampleInputFiles directory, stored in the exampleInputFilesDescriptorTable.csv.
    '''
    def setUp(self):
        '''Reads the reference data from the exampleInputFilesDescriptorTable.csv file and the labels from the first
        row of that file. Then it reads the first 5 structures from the exampleInputFiles directory and generates the
        descriptors for them. The results are stored in the functionOutput list. It also persists the test results in
        the Ward2017_TestResult.csv file.
        '''
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
        '''Compares the results of the Ward2017 descriptor generation function to the reference data on a field-by-field
        basis by requiring the absolute difference to be less than 1e-6.'''
        for fo, trd, name in zip(self.functionOutput, self.testReferenceData, self.exampleInputFiles):
            for p_fo, p_trd, l in zip(fo, trd, self.labels):
                if l not in self.skipLabels:
                    with self.subTest(msg=f'Testing {l} calculated for {name}'):
                        self.assertAlmostEqual(p_fo, p_trd, places=6)
    
    def test_cite(self):
        """Tests citation return."""
        citation = Ward2017.cite()
        self.assertIn("Krajewski", citation[0])
        self.assertIn("Ward", citation[1])


class TestWard2017Profiling(unittest.TestCase):
    '''Test the Ward2017 descriptor generation by profiling the execution time of the descriptor generation function
    for two example structures (JVASP-10001 and diluteNiAlloy).
    '''
    def test_serial(self):
        '''Test the serial execution of the descriptor generation function 4 times for each of the two examples.'''
        Ward2017.profile(test='JVASP-10001', nRuns=4)
        Ward2017.profile(test='diluteNiAlloy', nRuns=4)

    def test_parallel(self):
        '''Test the parallel execution of the descriptor generation function 24 times for each of the two examples
        but in parallel with up to 8 workers to speed up the execution.
        '''
        Ward2017.profileParallel(test='JVASP-10001', nRuns=24)
        Ward2017.profileParallel(test='diluteNiAlloy', nRuns=24)
