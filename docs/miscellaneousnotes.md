# Miscellaneous Notes for Users

## General Useful Tips

### Exporting Compressed Pymatgen Structures

Since [pymatgen v2023.05.31](https://github.com/materialsproject/pymatgen/releases/tag/v2023.05.31), or [PR#3003](https://github.com/materialsproject/pymatgen/pull/3003), you can compress Structure objects you export to JSON as in this simple example below. Considering the high volume of JSON's "boilerplate" in them, this should allow you to reduce file size by around half.

```python
from pymatgen.core import Lattice, Structure

FeO = Structure(
    lattice=Lattice.cubic(5),
    species=("Fe", "O"),
    coords=((0, 0, 0), (0.5, 0.5, 0.5)),
)

structure.to("FeO.json.gz")
structure.to("FeO.json.bz2")
```

## pySIPFENN Tricks


## Experimental