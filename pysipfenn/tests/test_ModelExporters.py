import unittest
import pytest
import os
import pysipfenn

# Skip the tests if we're in GitHub Actions and the models haven't been fetched yet
IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true" and os.getenv("MODELS_FETCHED") != "true"


class TestExporters(unittest.TestCase):
    '''Test all model exporting features that can operate on the Calculator object. Note that this will require
    the models to be downloaded and the environment variable MODELS_FETCHED to be set to true if running in GitHub
    Actions.'''

    def setUp(self):
        '''Initialise the Calculator object for testing.'''
        self.c = pysipfenn.Calculator()
        self.assertIsNotNone(self.c)

    def testInit(self):
        '''Test that the Calculator object is initialised correctly.'''
        self.assertEqual(self.c.predictions, [])
        self.assertEqual(self.c.toRun, [])
        self.assertEqual(self.c.descriptorData, [])
        self.assertEqual(self.c.inputFiles, [])

    def testExceptions1(self):
        '''Test that the exceptions are raised correctly by the exporters when Calculator is empty. Regardless of the
        model presence, it will skip the automatic loading of models to pretend it is a fresh install.'''
        c = pysipfenn.Calculator(autoLoad=False)

        with self.assertRaises(AssertionError,
                               msg='ONNXExporter did not raise an AssertionError on empty Calculator'):
            onnxexp = pysipfenn.ONNXExporter(c)
        with self.assertRaises(AssertionError,
                               msg='TorchExporter did not raise an AssertionError on empty Calculator'):
            torchexp = pysipfenn.TorchExporter(c)
        with self.assertRaises(AssertionError,
                               msg='CoreMLExporter did not raise an AssertionError on empty Calculator'):
            coremlexp = pysipfenn.CoreMLExporter(c)

    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test depends on the ONNX network files")
    def testExceptions2(self):
        '''Test that the exceptions are raised correctly by the exporters when the models are loaded but the
        descriptor they are trying to use is not defined in the exporter.'''
        self.assertIn('SIPFENN_Krajewski2020_NN24', self.c.models.keys(),
                      'This test requires the SIPFENN_Krajewski2020_NN24 model to be downloaded and loaded in the'
                      'Calculator object.')
        with self.assertRaises(KeyError,
                               msg='Not loaded models tried to pass silently.'):
            self.c.models['NotLoadedModel']['descriptor'] = 'NotAnImplementedDescriptor'

        self.c.models['SIPFENN_Krajewski2020_NN24']['descriptor'] = 'NotAnImplementedDescriptor'

        with self.assertRaises(NotImplementedError,
                               msg='TorchExporter did not raise an NotImplementedError on undefined descriptor'):
            torchexp = pysipfenn.TorchExporter(self.c)
            torchexp.export('SIPFENN_Krajewski2020_NN24')

        with self.assertRaises(NotImplementedError,
                               msg='CoreMLExporter did not raise an NotImplementedError on undefined descriptor'):
            coremlexp = pysipfenn.CoreMLExporter(self.c)
            coremlexp.export('SIPFENN_Krajewski2020_NN24')

        with self.assertRaises(NotImplementedError,
                               msg='ONNXExporter did not raise an NotImplementedError on undefined descriptor'):
            onnxexp = pysipfenn.ONNXExporter(self.c)
            onnxexp.export('SIPFENN_Krajewski2020_NN24')

    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test depends on the ONNX network files")
    def testModelsLoaded(self):
        '''Test that the models are loaded correctly.'''
        assert self.c.loadedModels.__len__() > 0, 'No models loaded in calculator. Nothing to export.'
        self.assertEqual(set(self.c.network_list_available), set(self.c.loadedModels.keys()))

    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test depends on the ONNX network files")
    def testONNXSimplify(self):
        '''Test that the ONNX simplification works with all models with no errors.'''
        self.onnxexp = pysipfenn.ONNXExporter(self.c)
        assert self.onnxexp.calculator == self.c
        assert self.onnxexp.simplifiedDict == {model: False for model in self.c.loadedModels.keys()}

        self.onnxexp.simplifyAll()
        assert self.onnxexp.simplifiedDict == {model: True for model in self.c.loadedModels.keys()}

    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test depends on the ONNX network files")
    def testONNXFP16(self):
        '''Test that the ONNX FP16 conversion works with all models with no errors.'''
        self.onnxexp = pysipfenn.ONNXExporter(self.c)
        assert self.onnxexp.calculator == self.c
        assert self.onnxexp.fp16Dict == {model: False for model in self.c.loadedModels.keys()}

        self.onnxexp.toFP16All()
        assert self.onnxexp.fp16Dict == {model: True for model in self.c.loadedModels.keys()}

    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test depends on the ONNX network files")
    def testONNXExport(self):
        '''Test that the ONNX export works with all models with no errors. For two of the models, the export will
        also simplify or convert to FP16 to check that it gets correctly encoded in the exported file name.'''
        self.onnxexp = pysipfenn.ONNXExporter(self.c)
        assert self.onnxexp.calculator == self.c

        self.onnxexp.simplify('SIPFENN_Krajewski2020_NN9')
        self.onnxexp.toFP16('SIPFENN_Krajewski2020_NN24')

        self.onnxexp.exportAll()

    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test depends on the ONNX network files")
    def testTorchExport(self):
        '''Test that the PyTorch export works with all models with no errors. Please note that if you are using
        custom descriptors, you will need to add them to the exporter definition in pysipfenn/core/modelExporters.py.'''
        self.torchexp = pysipfenn.TorchExporter(self.c)
        assert self.torchexp.calculator == self.c

        self.torchexp.exportAll()

    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test depends on the ONNX network files")
    def testCoreMLExport(self):
        '''Test that the CoreML export works with all models with no errors. Please note that if you are using
        custom descriptors, you will need to add them to the exporter definition in pysipfenn/core/modelExporters.py.'''
        self.coremlexp = pysipfenn.CoreMLExporter(self.c)
        assert self.coremlexp.calculator == self.c

        self.coremlexp.exportAll()
