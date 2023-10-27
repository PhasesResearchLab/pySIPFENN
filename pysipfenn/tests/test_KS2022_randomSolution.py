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
        testValues = KS2022_randomSolutions.profile(test='BCC', returnDescriptor=True)
        for testValue, descriptorRange, descriptorMean, label in zip(
                testValues,
                self.descriptorRangeList,
                self.descriptorMeanList,
                self.labels):
            with self.subTest(msg=f'{label} in BCC alloy'):
                self.assertGreaterEqual(testValue, (0.98*descriptorMean)-descriptorRange)
                self.assertLessEqual(testValue, (1.02*descriptorMean)+descriptorRange)

    def test_errors(self):
        '''Check if correct errors are raised when: (1) the test structure is not implemented.'''
        with self.assertRaises(NotImplementedError):
            KS2022_randomSolutions.profile(test='CrazyStructure', returnDescriptor=True)


class TestKS2022RandomSolutionProfiling(unittest.TestCase):
    '''Test the KS2022 RS descriptor generation by profiling the execution time for example composition for each
    common alloy structure: BCC, FCC, HCP. The profiling is serial inside each task, but done in parallel across
    structures.
    '''
    def test_serialInParallel(self):
        process_map(KS2022_randomSolutions.profile,
                    ['BCC', 'FCC', 'HCP'],
                    max_workers=3)


if __name__ == '__main__':
    unittest.main()
