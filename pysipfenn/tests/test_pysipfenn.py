import unittest
import pytest
import os

import pysipfenn
from importlib import resources
from natsort import natsorted
from numpy import zeros

from pymatgen.core import Structure, Composition

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true" and os.getenv("MODELS_FETCHED") != "true"


class TestCore(unittest.TestCase):
    '''Test the core functionality of the Calculator object and other high-level API functions. It does not test the
    correctness of the descriptor generation functions or models, as these are delegated to other tests.
    '''
    def setUp(self):
        '''Initialise the Calculator object for testing. It will be used in all tests and is not modified in any way
        by them.
        '''
        self.c = pysipfenn.Calculator()
        self.assertIsNotNone(self.c)

    def testInit(self):
        '''Test that the Calculator object is initialized correctly.'''
        self.assertEqual(self.c.predictions, [])
        self.assertEqual(self.c.toRun, [])
        self.assertEqual(self.c.descriptorData, [])
        self.assertEqual(self.c.inputFiles, [])

    def testDestroy(self):
        """ Test that the Calculator can deallocate itself (incl. loaded models and its data)."""
        self.assertIsNotNone(self.c)
        self.c.toRun = ['model1', 'model2']
        self.c.descriptorData = [zeros([271])]*10000
        self.c.destroy()

    def detectModels(self):
        '''Test that the updateModelAvailability() method works without errors and returns a list of available models.
        '''
        self.c.updateModelAvailability()
        self.assertIsInstance(self.c.network_list_available, list)

    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test depends on the ONNX network files")
    def testDownloadAndLoadModels(self):
        '''Tests that the downloadModels() method works without errors in a case whwere the models are not already
        downloaded and loads them correctly using the loadModels() method. Then also load a model explicitly using
        loadModel() and check that it is in the loadedModels list. Also check that error is raised correctly if
        a non-available model is requested to be loaded.
        '''

        self.c.downloadModels(network='all')
        self.c.loadModels(network='SIPFENN_Krajewski2020_NN24')

        self.assertEqual(set(self.c.network_list_available), set(self.c.loadedModels.keys()))
        self.assertIn('SIPFENN_Krajewski2020_NN24', self.c.loadedModels)

        with self.assertRaises(ValueError):
            self.c.loadModels(network='jx9348ghfmx8345wgyf')

    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test depends on the ONNX network files")
    def testFromPOSCAR_Ward2017(self):
        '''Update the list of available models and identifies which models are compatible with the Ward2017 descriptor.
        Then it runs featurization from the exampleInputFiles directory.
        '''
        self.c.updateModelAvailability()
        toRun = list(set(self.c.findCompatibleModels('Ward2017')).intersection(set(self.c.network_list_available)))
        if toRun:
            with resources.files('pysipfenn').joinpath('tests/testCaseFiles/exampleInputFiles') as testFileDir:
                print(testFileDir)
                self.c.runFromDirectory(testFileDir, 'Ward2017')
        else:
            raise ValueError('Did not detect any Ward2017 models to run')

    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test depends on the ONNX network files")
    def testFromPOSCAR_KS2022(self):
        '''Update the list of available models and identifies which models are compatible with the KS2022 descriptor.
        Then it runs featurization from the exampleInputFiles directory. It also tests the printout of the Calculator
        object after the prediction run.
        '''
        self.c.updateModelAvailability()
        toRun = list(set(self.c.findCompatibleModels('KS2022')).intersection(set(self.c.network_list_available)))
        if toRun:
            with resources.files('pysipfenn').joinpath('tests/testCaseFiles/exampleInputFiles') as testFileDir:
                print(testFileDir)
                self.c.runFromDirectory(testFileDir, 'KS2022')
        else:
            raise ValueError('Did not detect any KS2022 models to run')

        with self.subTest(msg='Test Calculator printout after predictions'):
            printOut = str(self.c)
            self.assertIn('pySIPFENN Calculator Object', printOut)
            self.assertIn('Models are located in', printOut)
            self.assertIn('Loaded Networks', printOut)
            self.assertIn('Last files selected as input', printOut)
            self.assertIn('Last Prediction Run Using', printOut)
            self.assertIn('Last prediction run on', printOut)



    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test depends on the ONNX network files")
    def testFromStructure_KS2022_dilute(self):
        '''Update the list of available models and identifies which models are compatible with the KS2022_dilute
        featurization (KS2022 descriptor). Then it runs featurization from the exampleInputFiles directory. It also
        then checks that the 'pure' convenience magic works correctly by comparing the results to the original pure
        structure results.
        '''
        self.c.updateModelAvailability()
        toRun = list(set(self.c.findCompatibleModels('KS2022')).intersection(set(self.c.network_list_available)))
        if toRun:
            matStr = '{"@module": "pymatgen.core.structure", "@class": "Structure", "charge": 0, "lattice": {' \
                     '"matrix": [[2.318956, 0.000185, -0.819712], [-1.159251, 2.008215, -0.819524], [2.5e-05, ' \
                     '0.000273, 2.459206]], "pbc": [true, true, true], "a": 2.4595700289085083, ' \
                     '"b": 2.4593515311565364, "c": 2.4592060152801354, "alpha": 109.45958252256221, ' \
                     '"beta": 109.46706290007663, "gamma": 109.46912204302215, "volume": 11.453776235839058}, ' \
                     '"sites": [{"species": [{"element": "Fe", "occu": 1}], "abc": [0.0, 0.0, 0.0], "xyz": [0.0, 0.0, ' \
                     '0.0], "label": "Fe", "properties": {"magmom": 2.211}}], "@version": null}'
            struct = Structure.from_str(matStr, fmt='json')
            struct.make_supercell([2, 2, 2])
            baseStruct = struct.copy()
            struct.replace(0, 'Al')

            preds1 = self.c.runModels_dilute(descriptor='KS2022',
                                             structList=[struct],
                                             baseStruct='pure',
                                             mode='serial')

            preds2 = self.c.runModels_dilute(descriptor='KS2022',
                                             structList=[struct],
                                             baseStruct=[baseStruct],
                                             mode='serial')

            for val1, val2 in zip(preds1[0], preds2[0]):
                self.assertEqual(val1, val2)

        else:
            raise ValueError('Did not detect any KS2022 models to run')

    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test depends on the ONNX network files")
    def testFromPrototypes_KS2022_randomSolution(self):
        """Quick runtime test of the top level API for random solution structures. It does not test the accuracy, as
        that is delegated elsewhere."""

        self.c.updateModelAvailability()
        toRun = list(set(self.c.findCompatibleModels('KS2022')).intersection(set(self.c.network_list_available)))
        if toRun:
            preds = self.c.runModels_randomSolutions(
                descriptor='KS2022',
                baseStructList='FCC',
                compList='AuCu',
                compositionConvergenceCriterion=0.05,
                featureConvergenceCriterion=0.02,
                minimumSitesPerExpansion=8,
                mode='serial')
        else:
            raise ValueError('Did not detect any KS2022 models to run')

    def test_descriptorCalculate_Ward2017_serial(self):
        '''Test succesful execution of the descriptorCalculate() method with Ward2017 in series. A separate test for
        calculation accuracy is done in test_Ward2017.py.
        '''
        with resources.files('pysipfenn').joinpath('tests/testCaseFiles/exampleInputFiles') as exampleInputsDir:
            exampleInputFiles = os.listdir(exampleInputsDir)[:6]
            testStructures = [Structure.from_file(f'{exampleInputsDir}/{eif}') for eif in exampleInputFiles]
            descList = self.c.calculate_Ward2017(structList=testStructures, mode='serial')
            self.assertEqual(len(descList), len(testStructures))

    def test_descriptorCalculate_Ward2017_parallel(self):
        '''Test succesful execution of the descriptorCalculate() method with Ward2017 in parallel. A separate test for
        calculation accuracy is done in test_Ward2017.py.
        '''
        with resources.files('pysipfenn').joinpath('tests/testCaseFiles/exampleInputFiles') as exampleInputsDir:
            exampleInputFiles = os.listdir(exampleInputsDir)[:6]
            testStructures = [Structure.from_file(f'{exampleInputsDir}/{eif}') for eif in exampleInputFiles]
            descList = self.c.calculate_Ward2017(structList=testStructures, mode='parallel', max_workers=2)
            self.assertEqual(len(descList), len(testStructures))

    def test_descriptorCalculate_KS2022_serial(self):
        '''Test succesful execution of the descriptorCalculate() method with KS2022 in series. A separate test for
        calculation accuracy is done in test_KS2022.py.
        '''
        with resources.files('pysipfenn').joinpath('tests/testCaseFiles/exampleInputFiles') as exampleInputsDir:
            exampleInputFiles = os.listdir(exampleInputsDir)
            testStructures = [Structure.from_file(f'{exampleInputsDir}/{eif}') for eif in exampleInputFiles]
            descList = self.c.calculate_KS2022(structList=testStructures, mode='serial')
            self.assertEqual(len(descList), len(testStructures))

    def test_descriptorCalculate_KS2022_parallel(self):
        '''Test succesful execution of the descriptorCalculate() method with KS2022 in parallel. A separate test for
        calculation accuracy is done in test_KS2022.py.
        '''
        with resources.files('pysipfenn').joinpath('tests/testCaseFiles/exampleInputFiles') as exampleInputsDir:
            exampleInputFiles = os.listdir(exampleInputsDir)
            testStructures = [Structure.from_file(f'{exampleInputsDir}/{eif}') for eif in exampleInputFiles]
            descList = self.c.calculate_KS2022(structList=testStructures, mode='parallel', max_workers=4)
            self.assertEqual(len(descList), len(testStructures))

    def test_descriptorCalculate_KS2022_dilute_serial(self):
        """Test succesful execution of the descriptorCalculate() method with KS2022_dilute in series based on an Al
        prototype loaded from the default prototype library. A separate test for calculation accuracy is done in
        test_KS2022.py"""
        diluteStruct = self.c.prototypeLibrary['FCC']['structure'].copy()
        diluteStruct.make_supercell([2, 2, 2])
        diluteStruct.replace(0, 'Fe')
        testStructures = [diluteStruct.copy()]*2
        descList = self.c.calculate_KS2022_dilute(structList=testStructures, mode='serial')
        self.assertEqual(len(descList), len(testStructures), "Not all structures were processed.")
        for desc in descList:
            self.assertListEqual(
                desc.tolist(),
                descList[0].tolist(),
                "All descriptors should be equal for the same structure are the same."
            )


    def test_descriptorCalculate_KS2022_dilute_parallel(self):
        """Test succesful execution of the descriptorCalculate() method with KS2022_dilute in parallel based on an Al
        prototype loaded from the default prototype library. A separate test for calculation accuracy is done in
        test_KS2022.py"""
        with self.subTest(msg="Constructing dilute structures"):
            diluteStruct = self.c.prototypeLibrary['FCC']['structure'].copy()
            diluteStruct.make_supercell([2, 2, 2])
            testStructures = []
            for i in range(8):
                tempStruct = diluteStruct.copy()
                tempStruct.replace(i, 'Fe')
                testStructures.append(tempStruct)

        with self.subTest(msg="Running parallel calculation with default 'pure' base structure"):
            descList = self.c.calculate_KS2022_dilute(structList=testStructures, mode='parallel', max_workers=4)
            self.assertEqual(len(descList), len(testStructures), "Not all structures were processed.")

        with self.subTest(msg="All descriptors should be equal for the same structure as sites are equivalent"):
            temp0 = descList[0].tolist()
            for desc in descList:
                temp1 = desc.tolist()
                for t0, t1 in zip(temp0, temp1):
                    self.assertAlmostEqual(t0, t1, places=6)

        with self.subTest(msg="Running parallel calculation with defined base structures"):
            baseStructs = [diluteStruct.copy()]*8
            descList = self.c.calculate_KS2022_dilute(
                structList=testStructures,
                baseStruct=baseStructs,
                mode='parallel',
                max_workers=4)
            self.assertEqual(len(descList), len(testStructures), "Not all structures were processed.")

        with self.subTest(msg="All descriptors should be equal for the same structure as sites are equivalent"):
            for desc in descList:
                temp1 = desc.tolist()
                for t0, t1 in zip(temp0, temp1):
                    self.assertAlmostEqual(t0, t1, places=6)

    def test_RunModels_Errors(self):
        '''Test that the runModels() and runModels_dilute() methods raise errors correctly when it is called with no
        models to run or with a descriptor handling that has not been implemented.
        '''
        with self.subTest(mgs='No models to run'):
            with self.assertRaises(AssertionError):
                self.c.network_list_available = []
                self.c.runModels(descriptor='KS2022', structList=[])

        with self.subTest(mgs='No models to run dilute'):
            with self.assertRaises(AssertionError):
                self.c.network_list_available = []
                self.c.runModels_dilute(descriptor='KS2022', structList=[])

        with self.subTest(mgs='No models to run random solid solution'):
            with self.assertRaises(AssertionError):
                self.c.network_list_available = []
                self.c.runModels_randomSolutions(descriptor='KS2022', baseStructList=[], compList=[])

        with self.subTest(mgs='Descriptor not implemented'):
            with self.assertRaises(AssertionError):
                self.c.runModels(descriptor='jx9348ghfmx8345wgyf', structList=[])

        with self.subTest(mgs='Dilute descriptor not implemented'):
            with self.assertRaises(AssertionError):
                self.c.runModels_dilute(descriptor='jx9348ghfmx8345wgyf', structList=[])

        with self.subTest(mgs='Random solution descriptor not implemented'):
            with self.assertRaises(AssertionError):
                self.c.runModels_randomSolutions(descriptor='jx9348ghfmx8345wgyf', baseStructList=[], compList=[])

    def test_WriteDescriptorDataToCSV(self):
        '''Test that the writeDescriptorsToCSV() method writes the correct data to a CSV file and that the file is
        consistent with the reference output. It does that with both anonymous structures it enumerates and labeled
        structures based on the c.inputFileNames list.
        '''
        with resources.files('pysipfenn').joinpath('tests/testCaseFiles/exampleInputFiles') as exampleInputsDir:
            exampleInputFiles = natsorted(os.listdir(exampleInputsDir))[:4]
            testStructures = [Structure.from_file(f'{exampleInputsDir}/{eif}') for eif in exampleInputFiles]
            self.c.calculate_KS2022(structList=testStructures, mode='serial')

        self.c.writeDescriptorsToCSV(descriptor='KS2022',
                                     file='TestFile_DescriptorData_4_KS2022_labeled_enumerated.csv')

        with open('TestFile_DescriptorData_4_KS2022_labeled_enumerated.csv', 'r', newline='') as f1:
            with resources.files('pysipfenn').joinpath(
                    'tests/testCaseFiles/TestFile_DescriptorData_4_KS2022_labeled_enumerated.csv').open('r',
                                                                                                        newline='') as f2:
                for line1, line2 in zip(f1, f2):
                    self.assertEqual(line1, line2)

        self.c.inputFiles = ['myStructure1.POSCAR', 'myStructure2.POSCAR', 'myStructure3.POSCAR', 'myStructure4.POSCAR']

        self.c.writeDescriptorsToCSV(descriptor='KS2022',
                                     file='TestFile_DescriptorData_4_KS2022_labeled_named.csv')

        with open('TestFile_DescriptorData_4_KS2022_labeled_named.csv', 'r', newline='') as f1:
            with resources.files('pysipfenn').joinpath(
                    'tests/testCaseFiles/TestFile_DescriptorData_4_KS2022_labeled_named.csv').open('r',
                                                                                                   newline=''
                                                                                                   ) as f2:
                for line1, line2 in zip(f1, f2):
                    self.assertEqual(line1, line2)

    def test_CalculatorPrint(self):
        '''Test that the Calculator.__str__() method returns the correctly formatted string after being initialized
        but before predictions.
        '''
        printOut = str(self.c)
        self.assertIn('pySIPFENN Calculator Object', printOut)
        self.assertIn('Models are located', printOut)
        self.assertIn('Loaded Networks', printOut)

    def test_util_Ward2017toKS2022(self):
        """Tests that Ward2017 conversion to its KS2022 subset works as intended."""
        struct = self.c.prototypeLibrary['FCC']['structure']
        self.assertIsInstance(struct, Structure)
        desc1 = self.c.calculate_Ward2017([struct])[0]
        desc2 = list(self.c.calculate_KS2022([struct])[0])
        desc2from1 = list(pysipfenn.ward2ks2022(desc1))
        for d2, d21 in zip(desc2, desc2from1):
            self.assertAlmostEqual(d2, d21, places=6, msg="Direct and converted KS2022toWard2017 should be the same.")


class TestCoreRSS(unittest.TestCase):
    """Test the high-level API functionality of the Calculator object in regard to random solution structures (RSS). It
    does not test the accuracy, just all runtime modes and known physicality of the results (e.g., FCC should have
    coordination number of `12`).

    Note:
        The execution of the descriptorCalculate() method with KS2022_randomSolution is done under coarse settings
        (for speed reasons) and should not be used for any accuracy tests. A separate testing for calculation accuracy
        against consistency and reference values is done in `test_KS2022_randomSolutions.py`.
    """
    def setUp(self):
        self.c = pysipfenn.Calculator()
        self.assertIsNotNone(self.c)

    def test_descriptorCalculate_KS2022_randomSolution_serial_pair(self):
        """Test successful execution of a composition-structure pair in series"""

        with self.subTest(msg="Running single composition-structure pair"):
            d1 = self.c.calculate_KS2022_randomSolutions(
                'BCC',
                'FeNi',
                minimumSitesPerExpansion=16,
                featureConvergenceCriterion=0.02,
                compositionConvergenceCriterion=0.05,
                mode='serial')
            self.assertEqual(len(d1), 1, "Only one composition-structure pair should be processed.")
            self.assertEqual(len(d1[0]), 256, "All 256 KS2022 features should be obtained.")

    def test_descriptorCalculate_KS2022_randomSolution_serial_multiple(self):
        """Test successful execution (in series) of multiple compositions occupying the same FCC lattice."""
        with self.subTest(msg="Running multiple compositions occupying the same FCC lattice"):
            d2 = self.c.calculate_KS2022_randomSolutions(
                'FCC',
                ['FeNi', 'CrNi'],
                minimumSitesPerExpansion=16,
                featureConvergenceCriterion=0.02,
                compositionConvergenceCriterion=0.05,
                mode='serial')
            self.assertEqual(len(d2), 2, "Two composition-structure pairs should be processed.")
            self.assertEqual(len(d2[0]), 256, "All 256 KS2022 features should be obtained.")
            self.assertEqual(len(d2[1]), 256, "All 256 KS2022 features should be obtained.")
            self.assertAlmostEqual(
                float(d2[0][0]),
                float(d2[1][0])
                , places=6, msg="Coordination number (KS2022[0]) should be the same (12) for both compositions.")
            self.assertNotAlmostEqual(
                float(d2[0][13]),
                float(d2[1][13])
                , places=6, msg="mean_NeighDiff_shell1_Number (KS2022[13]) should be different (1.0vs2.0)."
            )

    def test_descriptorCalculate_KS2022_randomSolution_parallel_pair(self):
        """Test successful execution of a composition-structure pair in parallel mode. Just for the input passing
        validation."""

        with self.subTest(msg="Running single composition-structure pair"):
            d1 = self.c.calculate_KS2022_randomSolutions(
                'BCC',
                'FeNi',
                mode='parallel',
                max_workers=1)
            self.assertEqual(len(d1), 1, "Only one composition-structure pair should be processed.")
            self.assertEqual(len(d1[0]), 256, "All 256 KS2022 features should be obtained.")

    def test_descriptorCalculate_KS2022_randomSolution_parallel_multiple(self):
        """Test successful execution of manu composition-structure pairs given in ordered lists of input."""
        myBCC = self.c.prototypeLibrary['BCC']['structure']

        with self.subTest(msg="Running multiple compositions occupying multiple prototypes"):
            d2 = self.c.calculate_KS2022_randomSolutions(
                ['FCC', myBCC, 'BCC', 'HCP'],
                ['WMo', Composition('WMo'), 'FeNi', 'CrNi'],
                mode='parallel',
                max_workers=4)
            self.assertEqual(len(d2), 4, "Four composition-structure pairs should be processed.")
            for i in range(4):
                self.assertEqual(len(d2[i]), 256, "All 256 KS2022 features should be obtained.")
            self.assertNotAlmostEqual(
                float(d2[0][0]),
                float(d2[1][0]),
                places=6, msg="Coordination number (KS2022[0]) should be different for BCC and FCC.")
            self.assertAlmostEqual(
                float(d2[1][0]),
                float(d2[2][0]),
                places=6, msg="Coordination number (KS2022[0]) should be the same for both BCCs.")

        with self.subTest(msg='Verify that the metadata was correctly recorded.'):
            assert len(self.c.metas['RSS']) == 4, "There should be 4 metadata records."
            for meta in self.c.metas['RSS']:
                self.assertIn('diffHistory', meta)
                self.assertIn('propHistory', meta)
                self.assertIn('finalAtomsN', meta)
                self.assertIn('finalCompositionDistance', meta)
                self.assertIn('finalComposition', meta)
