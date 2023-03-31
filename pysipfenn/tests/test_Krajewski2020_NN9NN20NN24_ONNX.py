import unittest
import pytest
import os
from pymatgen.core import Structure
from importlib import resources

import pysipfenn

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true" and os.getenv("MODELS_FETCHED") != "true"

testFile = '0-Cr8Fe18Ni4.POSCAR'
toTest = ['SIPFENN_Krajewski2020_NN9', 'SIPFENN_Krajewski2020_NN20', 'SIPFENN_Krajewski2020_NN24']
referenceEnergies_MxNet = [0.0790368840098381, 0.0498688854277133, 0.0871851742267608]
referenceEnergies_ONNX =  [0.0790369734168053, 0.0498689748346806, 0.0871851146221161]

with resources.files('pysipfenn').joinpath('tests/testCaseFiles/exampleInputFiles/') as exampleInputsDir:
    testStructure = Structure.from_file(f'{exampleInputsDir}/{testFile}')

class TestKrajewski2020ModelsFromONNX(unittest.TestCase):
    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test depends on the ONNX network files")
    def test_resutls(self):
        c = pysipfenn.Calculator()
        c.calculate_Ward2017(structList=[testStructure])
        c.makePredictions(models=c.loadedModels, toRun=toTest, dataInList=c.descriptorData)
        for p, name, ref_onnx in zip(c.predictions[0], toTest, referenceEnergies_ONNX):
            with self.subTest(msg=f'Predicting vs ONNX with {name:<16}'):
                self.assertAlmostEqual(p, ref_onnx, places=6)

        for p, name, ref_mxnet in zip(c.predictions[0], toTest, referenceEnergies_MxNet):
            with self.subTest(msg=f'Predicting vs MxNet with {name:<16}'):
                self.assertAlmostEqual(p, ref_mxnet, places=6)

if __name__ == '__main__':
    unittest.main()