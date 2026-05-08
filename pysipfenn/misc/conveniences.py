import ast
import inspect
import json
from importlib.resources import files
from importlib import import_module
import pkgutil

def _find_pymatgen_class(class_name: str):
    """Locate a class anywhere in pymatgen, robust to module reorganization."""
    import pymatgen
    for _, modname, _ in pkgutil.walk_packages(pymatgen.__path__, prefix="pymatgen."):
        try:
            mod = import_module(modname)
        except Exception:
            continue
        obj = getattr(mod, class_name, None)
        if isinstance(obj, type) and obj.__module__.startswith("pymatgen"):
            return obj
    return None

def patchPymatgenForExoticElements(
        x: bool = True,
        iupacOrder: bool = True,
        radii: bool = True,
) -> None:
    """
    Patch pymatgen's installed element data for elements whose properties are
    missing or incomplete in the default pymatgen data files.

    This function directly edits files inside the installed pymatgen package:

    1. Patches pymatgen's ``core/periodic_table.json`` with (selectable) electronegativities and IUPAC ordering values
    needed to correctly handle some exotic chemical elements. The IUPAC rules are followed exactly per Table VI in the
    same reference. The electronegativity values are `not` Pauling ones but based on Oganov 2021 and are meant to be
    used primarily for providing trend information for ML model deployment (has to be included in training).

    2. CovalentRadius.radius
    Adds missing covalent radii for elements Bk through Og using `ast` to locate the dictionary definition in
    pymatgen's source code, merge in the missing values, and write the updated literal back to disk. Radii reference
    values from Pekka Pyykkö, The Journal of Physical Chemistry A 2015 119 (11), 2326-2337,
    DOI: 10.1021/jp5065819

    Args:
        x: Patch electronegativities.
        iupacOrder: Patch IUPAC ordering of elements in chemical formulas so that they can be handled at all.
        radii: Patch ``CovalentRadius.radius`` with covalent radii for elements past Cm.

    Returns:
        None. The ``core/periodic_table.json`` files and the python file containing the ``CovalentRadius`` in the
        local install of ``pymatgen`` are patched. Reinstall or upgrade ``pymatgen`` to reverse the changes.
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

    # Patch covalent radii on disk. 
    # We locate the dict with `ast` and splice a merged literal back in.
    if radii:
        CovalentRadius = _find_pymatgen_class("CovalentRadius")
        if CovalentRadius is None:
            raise RuntimeError(
                "Could not locate `CovalentRadius` class in pymatgen; "
                "pymatgen's layout may have changed and this patch needs updating."
            )
        source_file = inspect.getsourcefile(CovalentRadius)
        with open(source_file, "r") as f:
            src = f.read()

        dict_node = None
        for cls in ast.walk(ast.parse(src)):
            if not (isinstance(cls, ast.ClassDef) and cls.name == "CovalentRadius"):
                continue
            for stmt in cls.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    target, value = stmt.target.id, stmt.value
                elif (isinstance(stmt, ast.Assign)
                      and len(stmt.targets) == 1
                      and isinstance(stmt.targets[0], ast.Name)):
                    target, value = stmt.targets[0].id, stmt.value
                else:
                    continue
                if target == "radius" and isinstance(value, ast.Dict):
                    dict_node = value
                    break
            break

        if dict_node is None:
            raise RuntimeError(
                f"Could not locate `CovalentRadius.radius` dict in {source_file}; "
                "pymatgen's layout may have changed and this patch needs updating."
            )

        existing = ast.literal_eval(dict_node)
        # Skip writing if the file is already up to date with our patch values.
        if any(existing.get(el) is None for el in patchRadii):
            merged = {**patchRadii, **existing}

            # Match pymatgen's existing indentation by reading it from the source
            # rather than hardcoding spaces, so the patch survives style changes.
            src_lines = src.splitlines(keepends=True)
            first_key = dict_node.keys[0]
            entry_indent = src_lines[first_key.lineno - 1][:first_key.col_offset]
            close_indent = src_lines[dict_node.end_lineno - 1][:dict_node.end_col_offset - 1]

            new_literal = "{\n" + "".join(
                f'{entry_indent}"{el}": {v},\n' for el, v in merged.items()
            ) + close_indent + "}"

            # Convert (line, col) bounds to byte offsets and splice.
            line_starts = [0]
            for line in src.splitlines(keepends=True):
                line_starts.append(line_starts[-1] + len(line))
            start = line_starts[dict_node.lineno - 1] + dict_node.col_offset
            end = line_starts[dict_node.end_lineno - 1] + dict_node.end_col_offset

            src = src[:start] + new_literal + src[end:]
            with open(source_file, "w") as f:
                f.write(src)

