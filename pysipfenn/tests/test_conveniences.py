import inspect
import warnings
import subprocess
import sys

import pytest

from pysipfenn.misc.conveniences import (
    _find_pymatgen_class,
    patchCovalentRadiiForExoticElements,
    patchPymatgenForExoticElements,
)



EXPECTED_COVALENT_RADII = {
    'Bk': 1.68,
    'Cf': 1.68,
    'Es': 1.65,
    'Fm': 1.67,
    'Md': 1.73,
    'No': 1.76,
    'Lr': 1.61,
    'Rf': 1.57,
    'Db': 1.49,
    'Sg': 1.43,
    'Bh': 1.41,
    'Hs': 1.34,
    'Mt': 1.29,
    'Ds': 1.28,
    'Rg': 1.21,
    'Cn': 1.22,
    'Nh': 1.36,
    'Fl': 1.43,
    'Mc': 1.62,
    'Lv': 1.75,
    'Ts': 1.65,
    'Og': 1.57,
    'H': 0.31,
    'He': 0.28,
    'Li': 1.28,
    'Be': 0.96,
    'B': 0.84,
    'C': 0.73,
    'N': 0.71,
    'O': 0.66,
    'F': 0.57,
    'Ne': 0.58,
    'Na': 1.66,
    'Mg': 1.41,
    'Al': 1.21,
    'Si': 1.11,
    'P': 1.07,
    'S': 1.05,
    'Cl': 1.02,
    'Ar': 1.06,
    'K': 2.03,
    'Ca': 1.76,
    'Sc': 1.7,
    'Ti': 1.6,
    'V': 1.53,
    'Cr': 1.39,
    'Mn': 1.5,
    'Fe': 1.42,
    'Co': 1.38,
    'Ni': 1.24,
    'Cu': 1.32,
    'Zn': 1.22,
    'Ga': 1.22,
    'Ge': 1.2,
    'As': 1.19,
    'Se': 1.2,
    'Br': 1.2,
    'Kr': 1.16,
    'Rb': 2.2,
    'Sr': 1.95,
    'Y': 1.9,
    'Zr': 1.75,
    'Nb': 1.64,
    'Mo': 1.54,
    'Tc': 1.47,
    'Ru': 1.46,
    'Rh': 1.42,
    'Pd': 1.39,
    'Ag': 1.45,
    'Cd': 1.44,
    'In': 1.42,
    'Sn': 1.39,
    'Sb': 1.39,
    'Te': 1.38,
    'I': 1.39,
    'Xe': 1.4,
    'Cs': 2.44,
    'Ba': 2.15,
    'La': 2.07,
    'Ce': 2.04,
    'Pr': 2.03,
    'Nd': 2.01,
    'Pm': 1.99,
    'Sm': 1.98,
    'Eu': 1.98,
    'Gd': 1.96,
    'Tb': 1.94,
    'Dy': 1.92,
    'Ho': 1.92,
    'Er': 1.89,
    'Tm': 1.9,
    'Yb': 1.87,
    'Lu': 1.87,
    'Hf': 1.75,
    'Ta': 1.7,
    'W': 1.62,
    'Re': 1.51,
    'Os': 1.44,
    'Ir': 1.41,
    'Pt': 1.36,
    'Au': 1.36,
    'Hg': 1.32,
    'Tl': 1.45,
    'Pb': 1.46,
    'Bi': 1.48,
    'Po': 1.4,
    'At': 1.5,
    'Rn': 1.5,
    'Fr': 2.6,
    'Ra': 2.21,
    'Ac': 2.15,
    'Th': 2.06,
    'Pa': 2,
    'U': 1.96,
    'Np': 1.9,
    'Pu': 1.87,
    'Am': 1.8,
    'Cm': 1.69
}

def _warn_if_radii_drift(actual_radii):
    """Emit a UserWarning (not failure) if patched radii dict differs from the expected snapshot."""
    if actual_radii == EXPECTED_COVALENT_RADII:
        return
    diff = {
        k: (actual_radii.get(k), EXPECTED_COVALENT_RADII.get(k))
        for k in set(actual_radii) | set(EXPECTED_COVALENT_RADII)
        if actual_radii.get(k) != EXPECTED_COVALENT_RADII.get(k)
    }
    warnings.warn(
        f"CovalentRadius.radius after patching does not match `EXPECTED_COVALENT_RADII`. "
        f"Differences (key: actual vs expected): {diff}"
        "This may indicate that pymatgen updated their covalent radii dict and the patch is out of sync.",
        UserWarning,
        stacklevel=2,
    )

def test_find_pymatgen_class():
    cls = _find_pymatgen_class("CovalentRadius")
    assert cls is not None
    assert cls.__name__ == "CovalentRadius"
    assert cls.__module__.startswith("pymatgen")

    assert _find_pymatgen_class("DefinitelyNotAPymatgenClass_xyzzy") is None

def test_patchCovalentRadiiForExoticElements():
    patchCovalentRadiiForExoticElements()
    state = _read_pymatgen_state()

    expected_patch_keys = {
        "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg",
        "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
    }
    missing = expected_patch_keys - set(state["radii"])
    assert not missing, f"Patched dict is missing keys: {sorted(missing)}"
    _warn_if_radii_drift(state["radii"])