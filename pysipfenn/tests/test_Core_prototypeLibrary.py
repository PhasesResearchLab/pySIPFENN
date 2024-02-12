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

