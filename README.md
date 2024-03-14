# pySIPFENN
[![GitHub top language](https://img.shields.io/github/languages/top/PhasesResearchLab/pysipfenn)](https://github.com/PhasesResearchLab/pySIPFENN)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pysipfenn)](https://pypi.org/project/pysipfenn)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL_v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![PyPI - Version](https://img.shields.io/pypi/v/pysipfenn?label=PyPI&color=green)](https://pypi.org/project/pysipfenn)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/pysipfenn)](https://pypi.org/project/pysipfenn)

[![Core Linux (Ubuntu)](https://github.com/PhasesResearchLab/pySIPFENN/actions/workflows/coreTests_LinuxUbuntu.yaml/badge.svg)](https://github.com/PhasesResearchLab/pySIPFENN/actions/workflows/coreTests_LinuxUbuntu.yaml)
[![Core Mac M1](https://github.com/PhasesResearchLab/pySIPFENN/actions/workflows/coreTests_MacM1.yaml/badge.svg)](https://github.com/PhasesResearchLab/pySIPFENN/actions/workflows/coreTests_MacM1.yaml)
[![Core Mac Intel](https://github.com/PhasesResearchLab/pySIPFENN/actions/workflows/coreTests_MacIntel.yaml/badge.svg)](https://github.com/PhasesResearchLab/pySIPFENN/actions/workflows/coreTests_MacIntel.yaml)
[![Core Windows](https://github.com/PhasesResearchLab/pySIPFENN/actions/workflows/coreTests_Windows.yaml/badge.svg)](https://github.com/PhasesResearchLab/pySIPFENN/actions/workflows/coreTests_Windows.yaml)
[![Full Test](https://github.com/PhasesResearchLab/pySIPFENN/actions/workflows/fullTest.yaml/badge.svg)](https://github.com/PhasesResearchLab/pySIPFENN/actions/workflows/fullTest.yaml)
[![codecov](https://codecov.io/gh/PhasesResearchLab/pySIPFENN/branch/main/graph/badge.svg?token=S2J0KR0WKQ)](https://codecov.io/gh/PhasesResearchLab/pySIPFENN)

[![stable](https://img.shields.io/badge/Read%20The%20Docs-Stable-green)](https://pysipfenn.readthedocs.io/en/stable/) 
[![latest](https://img.shields.io/badge/Read%20The%20Docs-Latest-green)](https://pysipfenn.readthedocs.io/en/latest/)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.commatsci.2022.111254-blue)](https://doi.org/10.1016/j.commatsci.2022.111254)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7373089.svg)](https://doi.org/10.5281/zenodo.7373089)

## Summary

This repository contains 
**py**(**S**tructure-**I**nformed **P**rediction of 
**F**ormation **E**nergy using **N**eural **N**etworks) software 
package allowing efficient predictions of the energetics of 
atomic configurations. The underlying methodology and implementation
is given in

- Adam M. Krajewski, Jonathan W. Siegel, Jinchao Xu, Zi-Kui Liu, _Extensible Structure-Informed Prediction of Formation Energy with improved accuracy and usability employing neural networks_, Computational Materials Science, Volume 208, 2022, 111254, DOI:[10.1016/j.commatsci.2022.111254](https://doi.org/10.1016/j.commatsci.2022.111254)

While functionalities are similar to the software released along the 
paper, this package contains improved methods for featurizing atomic 
configurations. Notably, all of them are now written completely in 
Python, removing reliance on Java and making extensions of the software
much easier thanks to improved readability. A fuller description of capabilities is 
given in documentation at https://pysipfenn.org and at PSU Phases 
Research Lab webpage under https://phaseslab.com/sipfenn.

### Major News:

- **(v0.15.0)** A new descriptor (feature vector) calculator [**`descriptorDefinitions.KS2022_randomSolutions`**](https://github.com/PhasesResearchLab/pySIPFENN/blob/main/pysipfenn/descriptorDefinitions/KS2022_randomSolutions.py) has been implemented. It is used for structure informed featurization of compositions randomly occupying a lattice, spiritually similar to SQS generation, but also taking into account (1) chemical differences between elements and (2) structural effects. A full description will be given in the upcoming manuscript.

- **(v0.14.0)** Users can now take advantage of a **Prototype Library** to obtain common structures from any `Calculator` instance `c` with a simple `c.prototypeLibrary['BCC']['structure']`. It can be easily [updated](https://pysipfenn.readthedocs.io/en/latest/source/pysipfenn.core.html#pysipfenn.Calculator.parsePrototypeLibrary) or [appended](https://pysipfenn.readthedocs.io/en/latest/source/pysipfenn.core.html#pysipfenn.Calculator.appendPrototypeLibrary) with high-level API or by manually modifyig its YAML [here](https://github.com/PhasesResearchLab/pySIPFENN/blob/main/pysipfenn/misc/prototypeLibrary.yaml).

- **(v0.13.0)** Model exports (and more!) to PyTorch, CoreML, and ONNX are now effortless thanks to [**`core.modelExporters`**](https://github.com/PhasesResearchLab/pySIPFENN/blob/main/pysipfenn/core/modelExporters.py) module. Please note you need to install pySIPFENN with `dev` option (e.g., `pip install "pysipfenn[dev]"`) to use it. See [docs here](https://pysipfenn.readthedocs.io/en/stable/source/pysipfenn.core.html#module-pysipfenn.core.modelExporters).

- **(v0.12.2)** Swith to LGPLv3 allowing for integration with proprietary software developed by CALPHAD community, while supporting the development of new pySIPFENN features for all. Many thanks to our colleagues from 
[GTT-Technologies](https://gtt-technologies.de) and other participants of [CALPHAD 2023](https://calphad.org/calphad-2023) for fruitful discussions.

- **(March 2023 Workshop)** We would like to thank all of our amazing attendees for making our workshop, co-organized with the
[Materials Genome Foundation](https://materialsgenomefoundation.org), such a success! Over 100 of you simultaneously followed
all exercises and, at the peak, we loaded over 1,200GB of models into the HPC's RAM. 

### Applications

pySIPFENN is a very flexible tool that can, in principle, be used for
the prediction of any property of interest that depends on an atomic
configuration with very few modifications. The models shipped by
default are trained to predict formation energy because that is what our
research group is interested in; however, if one wanted to predict
Poisson’s ratio and trained a model based on the same features, adding
it would take minutes. Simply add the model in open ONNX format and link
it using the *models.json* file, as described in the documentation.

### Real-World Examples

In our line of work, pySIPFENN and the formation energies it predicts are usually used 
as a computational engine that generates proto-data for creation of thermodynamic
databases (TDBs) using ESPEI (https://espei.org). The TDBs are then used through
pycalphad (https://pycalphad.org) to predict phase diagrams and other thermodynamic
properties. 

Another of its uses in our research is guiding the Density Functional Theory (DFT)
calculations as a low-cost screening tool. Their efficient conjunction then drives the
experiments leading to discovery of new materials, as presented in these two papers:

- Sanghyeok Im, Shun-Li Shang, Nathan D. Smith, Adam M. Krajewski, Timothy 
Lichtenstein, Hui Sun, Brandon J. Bocklund, Zi-Kui Liu, Hojong Kim, Thermodynamic 
properties of the Nd-Bi system via emf measurements, DFT calculations, machine 
learning, and CALPHAD modeling, Acta Materialia, Volume 223,
2022, 117448, https://doi.org/10.1016/j.actamat.2021.117448.

- Shun-Li Shang, Hui Sun, Bo Pan, Yi Wang, Adam M. Krajewski, 
Mihaela Banu, Jingjing Li & Zi-Kui Liu, Forming mechanism of equilibrium and 
non-equilibrium metallurgical phases in dissimilar aluminum/steel (Al–Fe) joints. 
Nature Scientific Reports 11, 24251 (2021). 
https://doi.org/10.1038/s41598-021-03578-0


## Installation

Installing pySIPFENN is simple and easy by utilizing **PyPI** package repository, **conda-forge**  package repository, or by cloning from **GitHub** directly.
While not required, it is recommended to first set up a virtual environment using venv or Conda. This ensures that (a) one of the required 
versions of Python (3.9+) is used and (b) there are no dependency conflicts. If you have Conda installed on your system (see [`miniconda` install instructions](https://docs.conda.io/en/latest/miniconda.html)), you can create a new environment with a simple:

    conda create -n pysipfenn python=3.10 jupyter numpy 
    conda activate pysipfenn

If you are managing a large set of dependencies in your project, you may consider using `mamba` in place of `conda`. It is a less mature, but much faster drop-in replacement compatible with existing environments. See [`micromamba` install instructions](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html).

### Standard

If your main goal is to run pySIPFENN models, provided by us or any other vendor, you need only a subset of the capabilities of our code, so
you can follow with the following install. Simply install pySIPFENN:

- from **PyPI** with `pip`:
    ```shell
    pip install pysipfenn
    ```

- from **conda-forge** with `conda`:
    ```shell
    conda install -c conda-forge pysipfenn
    ```

- from **conda-forge** with `micromamba`:
    ```shell
    micromamba install -c conda-forge pysipfenn
    ```

- **from source**, by cloning. To get a stable version, you can specify a version tag after the URL with
`--branch <tag_name> --single-branch`, or omit it to get the development version (which may have bugs!):
    ```shell
    git clone https://github.com/PhasesResearchLab/pySIPFENN.git
    ```

  then move to `pySIPFENN` directory and install in editable (`-e`) mode.
    ```shell
    cd pySIPFENN
    pip install -e .
    ``` 

### Developer Install

If you want to utilize pySIPFENN beyond its core functionalities, for instance, to train new models on custom datasets or to export models in different 
formats or precisions, you need to install several other dependencies. This can be done by following the **from source** instructions above but appending
the last instruction with `dev` _extras_ marker.

```shell
pip install -e ".[dev]"
```

> Note: `pip install "pysipfenn[dev]"` will also work, but will be less conveninet for model modifications (which you likely want to do), as all persisted
> files will be located outside your working directory. You can quickly find where, by calling `import pysipfenn; c = pysipfenn.Calculator(); print(c)` and
> `Calculator` will tell you (amongst other things) where they are.

## Contributing

### What to Contribute

If you wish to contribute to the development of pySIPFENN you are more than welcome to do so by forking the repository and creating a pull request. As of Spring
2024, we are actively developing the code and we should get back to you within a few days. We are also open to collaborations and partnerships, so if you have
an idea for a new feature or a new model, please do not hesitate to contact us through the GitHub issues or by [email](mailto:ak@psu.edu).

In particular, we are seeking contributions in the following areas:

- **New Models**: We are always looking for new models to add to the repository. We have several (yet) unpublished ones for several different properties, so there is a good chance it will work for your case as well. We are happy to provide basic support for training, including using the default model for **transfer learning on small datasets**.

- **New Featurizers / Descriptor Sets**: We are always looking for new ways to featurize atomic configurations. 
    - We are **particularly interested** in including more domain-specific knowledge for different niches of materials science. Our KS2022 does a good job for most materials, but we look to expand it. 
    - We are **not looking for** featurizers that (a) cannot embed a structure into the feature space (e.g., most of the graph representations, which became popular in the last two years) or (b) do not encode physics into the feature space (e.g., raw atomic coordinates or 3D voxel representations).
    - Note: Autoencoders which utilize graph or 3D voxel representations to encode latent space position to predict property/properties fall into the first category and **are very welcome**.

- **Quality of Life Improvements**: We are always looking for ways to make the software easier to use and more efficient for users. If you have an idea for a new data parsing method, or a new way to visualize the results, we would love to hear about it.

### Rules for Contributing

We are currently very flexible with the rules for contributing, despite being quite opinionated :) 

Some general guidelines are:
- The `core` module is the only one that should be used by our typical end user. All **top-level APIs should be defined in the `pysipfenn.py`** through the `Calculator` class. APIs operating _on_ the `Calculator` class, to export or retrain models, should be defined outside it, but within `pysipfenn.core` module.

- All **featurizers / descriptor calculators _must_ be self-contained in a single submodule** (file or directory) of `pysipfenn.descriptorDefinitions` (i.e., not spread around the codebase) and depend only on standard Python library and current pySIPFENN dependencies, including `numpy`, `torch`, `pymatgen`, `onnx`, `tqdm`. If you need to add a new dependency, please discuss it with us first.

- All models **_must_ be ONNX models**, which can be obtained from almost any machine learning framework. We are happy to help with this process.

- All new classes, attributes, and methods **_must_ be type-annotated**. We are happy to help with this process.

- All new classes, attributes, and methods **_must_ have a well-styled docstring**. We are happy to help with this process.

- All functions, classes, and methods **_should_ have explicit inputs**, rather than passing a dictionary of parameters (*kwargs). This does require a bit more typing, but it makes the code much easier to use for the end user, who can see in the IDE exactly what parameters are available and what they do.

- All functions, classes, and methods **_should_ explain _why_ they are doing something, not just _what_** they are doing. This is critical for end-users who did not write the code and are trying to understand it. In particular, the default values of parameters should be explained in the docstring.

- All new features _must_ be tested with the `pytest` framework. **Coverage _should_ be 100%** for new code or close to it for good reasons. We are happy to help with this process.




## Cite

If you use `pySIPFENN` software, please consider citing:

- Adam M. Krajewski, Jonathan W. Siegel, Jinchao Xu, Zi-Kui Liu, _Extensible Structure-Informed Prediction of Formation Energy with improved accuracy and usability employing neural networks_, Computational Materials Science, Volume 208, 2022, 111254, DOI:[10.1016/j.commatsci.2022.111254](https://doi.org/10.1016/j.commatsci.2022.111254)

If you are using predictions from pySIPFENN models accessed through `OPTIMADE` from `MPDD`, please additionally cite:

- Matthew L. Evans, Johan Bergsma, ..., Adam M. Krajewski, ..., Zi-Kui Liu, ..., et al., _Developments and applications of the OPTIMADE API for materials discovery, design, and data exchange_, 2024, [arXiv:2402.00572](https://arxiv.org/abs/2402.00572)

