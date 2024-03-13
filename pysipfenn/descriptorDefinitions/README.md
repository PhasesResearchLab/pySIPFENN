# pySIPFENN Descriptor Definitions Directory
This is the default folder in which pySIPFENN **feature calculators** are defined. Each of them is an independent software requireing only `pysipfenn` to be present in the system, thus you can easily separate them from main library, customize, and use independently. Notes:
- Please refer to the documentation page for details.
- Each of them generates numpy `ndarray` output, where features are indexed in the same order as in corresponding `labels_***.csv` files present here.
- `assets` stores miscellenious files, like graphics, not needed for functional purposes.