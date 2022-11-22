import unittest
import pysipfenn
from importlib import resources

class TestCore(unittest.TestCase):
    def setUp(self):
        self.c = pysipfenn.Calculator()
        self.assertIsNotNone(self.c)

    def testInit(self):
        self.assertEqual(self.c.predictions, [])
        self.assertEqual(self.c.network_list_available, [])

    def detectModels(self):
        self.c.updateModelAvailability()
        self.assertIsInstance(self.c.network_list_available, list)

    def test_from_poscar(self):
        if self.c.network_list_available!=[]:
            with resources.files('pysipfenn').joinpath('tests/testCaseFiles/exampleInputFiles') as testFileDir:
                print(testFileDir)
                self.c.runFromDirectory(testFileDir, 'Ward2017')
        else:
            print('Did not detect any models to run')

if __name__ == '__main__':
    unittest.main()
