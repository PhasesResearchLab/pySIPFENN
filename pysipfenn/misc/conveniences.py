from importlib.resources import files
import json

def patchPymatgenForExoticElements(
        x: bool = True,
        iupacOrder: bool = True,
        radii: bool = True,
) -> None:
    """Patches pymatgen's ``core/periodic_table.json`` with (selectable) electronegativities and IUPAC ordering values
    needed to correctly handle some exotic chemical elements. The IUPAC rules are followed exactly per Table VI in the
    same reference. The electronegativity values are `not` Pauling ones but based on Oganov 2021 and are meant to be
    used primarily for providing trend information for ML model deployment (has to be included in training).

    Covalent radii are patched in memory only (for the lifetime of the current Python process), since they live as
    a hardcoded ``dict`` literal in pymatgen's source rather than as loadable JSON. Call this function near the top
    of any script that needs the extended radii.

    Args:
        x: Patch electronegativities.
        iupacOrder: Patch IUPAC ordering of elements in chemical formulas so that they can be handled at all.
        radii: Patch ``CovalentRadius.radius`` in memory with covalent radii for elements past Cm. Effect is
            session-scoped — call this function each time you start a new Python process that needs the extended set.

    Returns:
        None. The ``core/periodic_table.json`` file in local install of ``pymatgen`` is patched. Reinstall or upgrade
        of ``pymatgen`` reverses the changes.
    """

    patchIUPAC = {
        'Rf': 49.5,
        'Db': 52.5,
        'Sg': 55.5,
        'Bh': 58.5,
        'Hs': 61.5,
        'Mt': 64.5,
        'Ds': 67.5,
        'Rg': 70.5,
        'Cn': 73.5,
        'Nh': 76.5,
        'Fl': 81.5,
        'Mc': 87.5,
        'Lv': 92.5,
        'Ts': 97.5,
        'Og': -0.5
    }
    
    patchX = {
        'Ar': 3.57,
        'He': 4.42,
        'Ne': 4.44,
        'Rf': 2.27,
        'Db': 2.38,
        'Sg': 2.51,
        'Bh': 2.48,
        'Hs': 2.52,
        'Mt': 2.66,
        'Ds': 2.73,
        'Rg': 2.83,
        'Cn': 3.03,
        'Nh': 2.49,
        'Fl': 2.57,
        'Mc': 2.21,
        'Lv': 2.42,
        'Ts': 2.61,
        'Og': 2.59
    }

    patchRadii = {
        "Bk": 1.68,
        "Cf": 1.68,
        "Es": 1.65,
        "Fm": 1.67,
        "Md": 1.73,
        "No": 1.76,
        "Lr": 1.61,
        "Rf": 1.57,
        "Db": 1.49,
        "Sg": 1.43,
        "Bh": 1.41,
        "Hs": 1.34,
        "Mt": 1.29,
        "Ds": 1.28,
        "Rg": 1.21,
        "Cn": 1.22,
        "Nh": 1.36,
        "Fl": 1.43,
        "Mc": 1.62,
        "Lv": 1.75,
        "Ts": 1.65,
        "Og": 1.57,
    }


    with files("pymatgen").joinpath("core/periodic_table.json").open() as f:
        pt = json.load(f)

    # Patch periodic table
    with files("pymatgen").joinpath("core/periodic_table.json").open("w") as f:
        if x:
            for el in patchX:
                pt[el]["X"] = patchX[el]
        if iupacOrder:
            for el in patchIUPAC:
                pt[el]["IUPAC ordering"] = patchIUPAC[el]
        json.dump(pt, f)

    # Patch covalent radii in memory for the current Python process. Lives as a hardcoded
    # dict literal in pymatgen source, so on-disk patching isn't an option here.
    if radii:
        from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
        CovalentRadius.radius.update(patchRadii)