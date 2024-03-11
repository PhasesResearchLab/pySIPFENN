import unittest
from pymatgen.core import Structure
from importlib import resources
import shutil
import pysipfenn
import pytest
import os

class TestPL(unittest.TestCase):
    """Tests correct loading of the prototype library (used, e.g., for random solid solution generation)."""

    def setUp(self) -> None:
        """Load the prototype library."""
        self.c = pysipfenn.Calculator(autoLoad=False)

    def test_autoload(self):
        """Test that the default prototype library is loaded."""
        self.assertTrue(self.c.prototypeLibrary is not None)
        self.assertTrue(len(self.c.prototypeLibrary) > 0)

    def test_defaultPresent(self):
        """Test that the loaded prototype library was correctly parsed."""
        for prototype in ["FCC", "BCC", "HCP", "Diamond", "DHCP", "Sn_A5"]:
            with self.subTest(msg=prototype):
                self.assertTrue(prototype in self.c.prototypeLibrary)

    def test_correctContentFCC(self):
        """Test that the FCC prototype was correctly parsed."""
        fcc = self.c.prototypeLibrary["FCC"]
        self.assertEqual(fcc["origin"], "https://www.oqmd.org/materials/prototype/A1_Cu")
        self.assertEqual(
            fcc["POSCAR"],
            ('A1_Cu\n'
             '1.0\n'
             '  0.00000   1.80750   1.80750\n'
             '  1.80750   0.00000   1.80750\n'
             '  1.80750   1.80750   0.00000\n'
             'Cu\n'
             '1\n'
             'Direct\n'
             '  0.00000   0.00000   0.00000\n'))
        with self.subTest(msg="Is a pymatgen Structure"):
            self.assertTrue(isinstance(fcc["structure"], Structure))
        with self.subTest(msg="Is valid pymatgen Structure"):
            self.assertTrue(fcc["structure"].is_valid())
        with self.subTest(msg="Has correct formula"):
            self.assertEqual(fcc["structure"].formula, "Cu1")

    def test_customPrototypeLoad(self):
        """Test that a custom prototype can be loaded. Then test that a custom prototype can be appended to the default
        library and stay there."""

        with resources.files('pysipfenn').joinpath('tests/testCaseFiles/prototypeLibrary-custom.yaml') as f:
            self.c.parsePrototypeLibrary(customPath=f, verbose=True, printCustomLibrary=True)

        with self.subTest(msg="Custom prototype present with correct parse"):
            self.assertTrue("NicePhase" in self.c.prototypeLibrary)
            self.assertEqual(self.c.prototypeLibrary["NicePhase"]["origin"], "https://somecustomsource.org")

        with self.subTest(msg="Nice phase is a valid pymatgen Structure"):
            self.assertTrue(isinstance(self.c.prototypeLibrary["NicePhase"]["structure"], Structure))
            self.assertTrue(self.c.prototypeLibrary["NicePhase"]["structure"].is_valid())
            self.assertEqual(self.c.prototypeLibrary["NicePhase"]["structure"].formula, "U1")

        with self.subTest(msg="FCC prototype still present"):
            self.assertTrue("FCC" in self.c.prototypeLibrary)

        with self.subTest(msg="Test that it does not affect the default prototype library"):
            otherC = pysipfenn.Calculator(autoLoad=False)
            self.assertTrue("NicePhase" not in otherC.prototypeLibrary)

        # Create a backup of the default library
        self.c = pysipfenn.Calculator(autoLoad=False)
        backup = self.c.prototypeLibrary.copy()

        with resources.files('pysipfenn').joinpath('tests/testCaseFiles/prototypeLibrary-custom.yaml') as f:
            self.c.appendPrototypeLibrary(customPath=f)

        with self.subTest(msg="Custom prototype present and valid in a different Calculator instance"):
            otherC = pysipfenn.Calculator(autoLoad=False)
            self.assertTrue("NicePhase" in otherC.prototypeLibrary)
            self.assertEqual(otherC.prototypeLibrary["NicePhase"]["origin"], "https://somecustomsource.org")
            self.assertTrue(isinstance(otherC.prototypeLibrary["NicePhase"]["structure"], Structure))
            self.assertTrue(otherC.prototypeLibrary["NicePhase"]["structure"].is_valid())
            self.assertEqual(otherC.prototypeLibrary["NicePhase"]["structure"].formula, "U1")

        with self.subTest(msg="FCC/BCC/HCP prototype still present in a different Calculator instance"):
            self.assertTrue("FCC" in otherC.prototypeLibrary)
            self.assertTrue("BCC" in otherC.prototypeLibrary)
            self.assertTrue("HCP" in otherC.prototypeLibrary)

        with self.subTest(msg="Restore the original prototype library"):
            pysipfenn.overwritePrototypeLibrary(backup)
