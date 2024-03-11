import unittest
import csv
import numpy as np
from importlib import resources
from tqdm.contrib.concurrent import process_map

from pysipfenn.descriptorDefinitions import KS2022_randomSolutions


class TestKSRandomSolution2022(unittest.TestCase):
    def setUp(self):
        '''Load the feature ranges and means from the test file as calculated for the testing/profiling case included
        in the KS2022_randomSolution script run for 50 iterations over BCC Cr12.8 Fe12.8 Co12.8 Ni12.8 Cu12.8 Al35.9
        alloy.
        '''
        testFilePath = 'tests/testCaseFiles/TestFile_DescriptorData_KS2022_randomSolution_valueRangesMeans.csv'
        with resources.files('pysipfenn').joinpath(testFilePath).open('r', newline='') as f:
            reader = csv.reader(f)
            self.descriptorRangeList, self.descriptorMeanList = np.float_(list(reader)).T

        with resources.files('pysipfenn').joinpath('descriptorDefinitions/labels_KS2022.csv').open('r') as f:
            reader = csv.reader(f)
            self.labels = [row[0] for row in reader]

    def test_results(self):
        '''Test the descriptor generation function by comparing the results feature by feature to the reference data
        statistics, testing whether the result is within the observed range of values from the observed mean value
        allowing for additional 2% deviation from the mean value to handle numerical precision in cases where the
        feature converges to near exactly the mean value with near-zero range (e.g. coordination number in BCC in case
        of ideal lattice positions).
        '''
        testValues, meta = KS2022_randomSolutions.profile(
            test='BCC',
            returnDescriptorAndMeta=True,
            plotParameters=True)

        for testValue, descriptorRange, descriptorMean, label in zip(
                testValues,
                self.descriptorRangeList,
                self.descriptorMeanList,
                self.labels):
            with self.subTest(msg=f'{label} in BCC alloy'):
                self.assertGreaterEqual(testValue, (0.95*descriptorMean)-descriptorRange-1e-4)
                self.assertLessEqual(testValue, (1.05*descriptorMean)+descriptorRange+1e-4)

        for field in ['diffHistory', 'propHistory', 'finalAtomsN', 'finalCompositionDistance', 'finalComposition']:
            with self.subTest(msg=f'{field} present in meta'):
                self.assertIn(field, meta)

        with self.subTest(msg="Verify runtime of onlyStructural function to select slices of the descriptor"):
            structuralValues = KS2022_randomSolutions.onlyStructural(testValues)
            self.assertEqual(len(structuralValues), 103)

    def test_errors(self):
        '''Check if correct errors are raised when: (1) the test structure is not implemented.'''
        with self.assertRaises(NotImplementedError):
            KS2022_randomSolutions.profile(test='CrazyStructure')

    def test_cite(self):
        """Tests citation return."""
        citation = KS2022_randomSolutions.cite()
        self.assertIn("Krajewski", citation[0])


class TestKS2022RandomSolutionProfiling(unittest.TestCase):
    '''Test the KS2022 RS descriptor generation by profiling the execution time for example composition for each
    common alloy structure: BCC, FCC, HCP. The profiling is serial inside each task, but done in parallel across
    structures.
    '''
    def test_serialInParallel(self):
        '''Tests profiling a set of structures with parallel task execution.'''
        process_map(KS2022_randomSolutions.profile,
                    ['BCC', 'FCC', 'HCP'],
                    max_workers=3)

    def test_singleInParallel(self):
        '''Tests parallel execution profiling works.'''
        KS2022_randomSolutions.profile(test='BCC', nIterations=2)
