import unittest
import pytest
import os
import pysipfenn
import torch
from importlib import resources

# Skip the tests if we're in GitHub Actions and the models haven't been fetched yet
IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true" and os.getenv("MODELS_FETCHED") != "true"

@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test depends on the ONNX network files")
class TestModelAdjusters(unittest.TestCase):
    """
    Test all model adjusting features that can operate on the Calculator object. Note that this will require
    the models to be downloaded and the environment variable MODELS_FETCHED to be set to true if running in GitHub
    Actions.

    The setup will load the Krajewski2022_NN30 model and create an ``OPTIMADEAdjuster`` object for testing that is by
    default connected to the Materials Project ``OPTIMADE`` server and looks for their GGA+U formation energies. In the
    ``testFullRoutine`` test, the adjuster will be used to adjust the model to the Hf-Mo metallic system. The test will
    cover almost all adjuster functionalities in different ways to hit all anticipated code paths. It also tests the
    ``LocalAdjuster`` class for loading data from CSV and NPY files, which is a parent class of the ``OPTIMADEAdjuster``.
    """

    def setUp(self):
        """
        Initialises the Calculator and ModelAdjuster objects for testing.
        """
        self.c = pysipfenn.Calculator(autoLoad=False)
        self.assertIsNotNone(self.c)
        self.c.loadModels("SIPFENN_Krajewski2022_NN30")
        self.assertIn('SIPFENN_Krajewski2022_NN30', self.c.loadedModels)

        self.ma = pysipfenn.OPTIMADEAdjuster(self.c, "SIPFENN_Krajewski2022_NN30")

    def testInit(self):
        """
        Test that the OPTIMADEAdjuster object has been initialized correctly.
        """
        self.assertEqual(self.ma.modelName, "SIPFENN_Krajewski2022_NN30")
        self.assertIsInstance(self.ma.model, torch.nn.Module)
        self.assertIsInstance(self.ma.calculator, pysipfenn.Calculator)

        self.assertEqual(len(self.ma.comps), 0)
        self.assertEqual(len(self.ma.names), 0)
        self.assertEqual(len(self.ma.validationLabels), 0)

    def testPlotExceptions(self):
        """
        Test that the plot does not plot anything when no data is present.
        """
        self.assertRaises(AssertionError, self.ma.plotStarting)
        self.assertRaises(AssertionError, self.ma.plotAdjusted)

    def testFullRoutine(self):
        """
        Test the full routine of the adjuster based on the default values pointing to Materials Project. Get the data
        using OPTIMADE to adjust the model to Hf-Mo metallic system. Matrix search is reduced to 4 cases to speed up
        the test and it is designed to explore all code paths in the search. The test will also check the highlighting
        and plotting functionalities of the adjuster.
        """
        self.ma.fetchAndFeturize(
            'elements HAS "Hf" AND elements HAS "Mo" AND NOT elements HAS ANY "O","C","F","Cl","S"',
            parallelWorkers=4)

        self.ma.calculator.writeDescriptorsToCSV("KS2022", "AdjusterTestDescriptors.csv")
        self.ma.calculator.writeDescriptorsToNPY("KS2022", "AdjusterTestDescriptors.npy")

        # Check highlighting and no-last-validation plotting
        self.ma.highlightPoints([32, 23, 21, 22])
        self.ma.plotStarting()

        # Hyperparameter search. The 1e-8 is on purpose, so that the model does not converge and always improves after
        # the first epoch.
        self.ma.matrixHyperParameterSearch(
            learningRates=[1e-8, 1e-3],
            optimizers= ["Adam"],
            weightDecays=[1e-4, 1e-5],
            epochs=10
        )

        self.ma.highlightPoints([0, 1, 2, 3])
        self.ma.highlightCompositions(["Hf", "Mo", "HfMo", "Hf50 Mo50", "Hf3Mo"])

        self.ma.plotStarting()
        self.ma.plotAdjusted()

        # Induce duplicates to test if they are handled
        self.ma.fetchAndFeturize(
            'elements HAS "Hf" AND elements HAS "Mo" AND NOT elements HAS ANY "O","C","F","Cl","S"',
            parallelWorkers=4)

        self.ma.adjust(
            validation=0,
            learningRate=1e-4,
            epochs=10,
            optimizer="Adamax",
            weightDecay=1e-4,
            lossFunction="MSE"
        )

        self.ma.names = []
        self.ma.plotStarting()
        self.ma.plotAdjusted()

    def testDataLoading(self):
        """
        Test the data loading functionality of the ``LocalAdjuster`` class (note, ``OPTIMADEAdjuster`` extends it). It
        will test loading from both CSV and NPY files exported from the Calculator object. Note that CSV files have
        names in the first column and headers in the first row, while NPY files are just the data arrays. It tests
        implicit loading from the ``Calculator`` object as well. Lastly, it tests the error raising for unsupported
        descriptors and data not matching the descriptor dimensions selected (an optional feature).
        """

        with resources.files('pysipfenn').joinpath('tests/testCaseFiles/') as testFileDir:

            # From CSV
            self.lma1 = pysipfenn.LocalAdjuster(
                self.c,
                model="SIPFENN_Krajewski2022_NN30",
                descriptorData=str(testFileDir.joinpath("AdjusterTestDescriptors.csv")),
                targetData=str(testFileDir.joinpath("AdjusterTestTargets.csv")),
                descriptor="KS2022"
            )
            assert self.lma1 is not None
            assert len(self.lma1.descriptorData) > 0
            assert len(self.lma1.targetData) > 0
            del self.lma1

            # From NPY
            self.lma2 = pysipfenn.LocalAdjuster(
                self.c,
                model="SIPFENN_Krajewski2022_NN30",
                descriptorData=str(testFileDir.joinpath("AdjusterTestDescriptors.npy")),
                targetData=str(testFileDir.joinpath("AdjusterTestTargets.npy")),
                descriptor="KS2022"
            )
            assert self.lma2 is not None
            assert len(self.lma2.descriptorData) > 0
            assert len(self.lma2.targetData) > 0

            self.c.descriptorData = self.lma2.descriptorData

            del self.lma2

            # Implicit, from the Calculator
            self.lma3 = pysipfenn.LocalAdjuster(
                self.c,
                targetData=str(testFileDir.joinpath("AdjusterTestTargets.csv")),
                model="SIPFENN_Krajewski2022_NN30",
                descriptor="KS2022",
            )

            # Error raising
            with self.assertRaises(AssertionError):
                self.lma4 = pysipfenn.LocalAdjuster(
                    self.c,
                    targetData=str(testFileDir.joinpath("AdjusterTestTargets.csv")),
                    model="SIPFENN_Krajewski2022_NN30",
                    descriptor="Ward2017",
                )

            with self.assertRaises(NotImplementedError):
                self.lma5 = pysipfenn.LocalAdjuster(
                    self.c,
                    targetData=str(testFileDir.joinpath("AdjusterTestTargets.csv")),
                    model="SIPFENN_Krajewski2022_NN30",
                    descriptor="SomeCrazyDescriptor",
                )

    def testEndpointOverride(self):
        """
        Test the endpoint override functionality of the ``OPTIMADEAdjuster`` class. It will test the override of the
        endpoint and the data fetching from the new endpoint.
        """
        endpoint = ["https://alexandria.icams.rub.de/pbesol"]
        targetPath = ['attributes', '_alexandria_formation_energy_per_atom']

        self.ma2 = pysipfenn.OPTIMADEAdjuster(
            self.c,
            model="SIPFENN_Krajewski2022_NN30",
            endpointOverride=endpoint,
            targetPath=targetPath)

        self.ma2.fetchAndFeturize(
            'elements HAS "Hf" AND elements HAS "Mo" AND elements HAS "Zr"',
            parallelWorkers=2)

        self.assertGreater(len(self.ma2.comps), 0, "No compositions were found, thus no data was fetched.")
        self.assertGreater(len(self.ma2.names), 0, "No names were found, thus no data was fetched.")
        self.assertGreater(
            len(self.ma2.descriptorData), 0,
            "No descriptor data was found. If the other asserts passed, this is likely a bug in the featurization "
            "or structural data has been made incompatible or otherwise corrupted.")
        
        for ref in self.ma2.references:
            self.assertGreater(len(ref), 0, "No references were found for the datapoint despite Alexendria being used and having them for this test case as of writing this (and very likely into the future).")