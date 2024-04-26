from importlib.resources import files

def patchPymatgenForExoticElements(
        x: bool = True,
        iupacOrder: bool = True
) -> None:
    """Patches pymatgen's ``core/periodic_table.json`` with (selectable) electronegativities and IUPAC ordering values
    needed to correctly handle some exotic chemical elements. The IUPAC rules are followed exactly per Table VI in the 
    same reference. The electronegativity values are `not` Pauling ones but based on Oganov 2021 and are meant to be 
    used primarily for providing trend information for ML model deployment (has to be included in training).

    Args:
        x: Patch electronegativities.
        iupacOrder: Patch IUPAC ordering of elements in chemical formulas so that they can be handled at all.

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