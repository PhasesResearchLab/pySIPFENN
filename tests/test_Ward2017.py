import unittest
import csv
import os
from pymatgen.core import Structure
from tqdm import tqdm
import numpy as np
from natsort import natsorted

Ward2017 = __import__('descriptorDefinitions.Ward2017', fromlist=[''])

with open('testCaseFiles/exampleInputFilesDescriptorTable.csv', newline='') as f:
    reader = csv.reader(f)
    referenceDescriptorTable = list(reader)

labels = referenceDescriptorTable[0]
testReferenceData = np.float_(referenceDescriptorTable[1:]).tolist()
skipLabels = ['mean_WCMagnitude_Shell3']

exampleInputFiles = natsorted(os.listdir('testCaseFiles/exampleInputFiles/'))
testStructures = [Structure.from_file(f'testCaseFiles/exampleInputFiles/{eif}') for eif in exampleInputFiles]

functionOutput = [Ward2017.generate_descriptor(s) for s in tqdm(testStructures[:5])]
with open('Ward2017_TestResult.csv', 'w+') as f:
    f.writelines([f'{v}\n' for v in functionOutput[0]])

class TestWard2017(unittest.TestCase):
    def test_resutls(self):
        for fo, trd, name in zip(functionOutput, testReferenceData, exampleInputFiles):
            for p_fo, p_trd, l in zip(fo, trd, labels):
                if l not in skipLabels:
                    with self.subTest(msg=f'Testing {l} calculated for {name}'):
                        self.assertAlmostEqual(p_fo, p_trd, places=6)

if __name__ == '__main__':
    unittest.main()
