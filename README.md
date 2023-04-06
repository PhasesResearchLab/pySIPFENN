## pySIPFENN
![GitHub top language](https://img.shields.io/github/languages/top/PhasesResearchLab/pysipfenn)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pysipfenn)
![PyPI](https://img.shields.io/pypi/v/pysipfenn)
![PyPI - Downloads](https://img.shields.io/pypi/dm/pysipfenn)

[![stable](https://img.shields.io/badge/Read%20The%20Docs-Stable-green)](https://pysipfenn.readthedocs.io/en/stable/) 
[![latest](https://img.shields.io/badge/Read%20The%20Docs-Latest-green)](https://pysipfenn.readthedocs.io/en/latest/)
![GitHub license](https://img.shields.io/github/license/PhasesResearchLab/pySIPFENN)

[![Partial Test](https://github.com/PhasesResearchLab/pySIPFENN/actions/workflows/partialTest.yml/badge.svg)](https://github.com/PhasesResearchLab/pySIPFENN/actions/workflows/partialTest.yml)
[![Full Test](https://github.com/PhasesResearchLab/pySIPFENN/actions/workflows/fullTest.yml/badge.svg)](https://github.com/PhasesResearchLab/pySIPFENN/actions/workflows/fullTest.yml)
[![codecov](https://codecov.io/gh/PhasesResearchLab/pySIPFENN/branch/main/graph/badge.svg?token=S2J0KR0WKQ)](https://codecov.io/gh/PhasesResearchLab/pySIPFENN)

![GitHub last commit (by committer)](https://img.shields.io/github/last-commit/PhasesResearchLab/pysipfenn?label=Last%20Commit)
![GitHub Release Date - Published_At](https://img.shields.io/github/release-date/PhasesResearchLab/pysipfenn?label=Last%20Release)
![GitHub commits since tagged version](https://img.shields.io/github/commits-since/PhasesResearchLab/pysipfenn/v0.12.0?color=g)
![GitHub issues](https://img.shields.io/github/issues/PhasesResearchLab/pysipfenn)

[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.commatsci.2022.111254-blue)](https://doi.org/10.1016/j.commatsci.2022.111254)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7373089.svg)](https://doi.org/10.5281/zenodo.7373089)

### News

- **(v.12.0)** Official Python 3.11 support. 
- **(v.12.0)** Automated matrix-testing on Linux / Mac / Windows with Python 3.9 / 3.10 / 3.11 through GitHub Actions CLI
and test coverage report through Codecov. Tests are also generally improved and more extensive.
- **(v0.11.0)** Some common questions are now addressed in the [documentation FAQ section](https://pysipfenn.readthedocs.io/en/stable/faq.html).
- **(v0.11.0)** The model downloads from Zenodo are now multithreaded and are 15 times faster.
- **(March 2023 Workshop)** We would like to thank all of our amazing attendees for making our workshop, co-organized with the
[Materials Genome Foundation](https://materialsgenomefoundation.org), such a success! Over 100 of you simultaneously followed
all exercises and, at the peak, we loaded over 1,200GB of models into the HPC's RAM. At this point, we would also like to 
acknowledge the generous support from [IBM](https://www.ibm.com) who funded the workshop. Please stay tuned for next workshops
planned online and in-person at conferences. They will be announced both here and at the [Materials Genome Foundation](https://materialsgenomefoundation.org) website.

### Summary

This repository contains 
**py**(**S**tructure-**I**nformed **P**rediction of 
**F**ormation **E**nergy using **N**eural **N**etworks) software 
package allowing efficient predictions of the energetics of 
atomic configurations. The underlying methodology and implementation
is given in

- Adam M. Krajewski, Jonathan W. Siegel, Jinchao Xu, Zi-Kui Liu,
Extensible Structure-Informed Prediction of Formation Energy with improved accuracy and usability employing neural networks,
Computational Materials Science,
Volume 208,
2022,
111254
https://doi.org/10.1016/j.commatsci.2022.111254

While functionalities are similar to the software released along the 
paper, this package contains improved methods for featurizing atomic 
configurations. Notably, all of them are now written completely in 
Python, removing reliance on Java and making extensions of the software
much easier thanks to improved readability. A fuller description of capabilities is 
given in documentation at https://pysipfenn.org and at PSU Phases 
Research Lab webpage under https://phaseslab.com/sipfenn.

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


### Install pySIPFENN

Installing pySIPFENN is simple and easy utilizing either **PyPI** package repository or cloning from **GitHub**. 
While not required, it is recommended to first set up a virtual environment using venv or Conda. This ensures that 
one of the required versions of Python (3.9+) is used and there are no dependency conflicts. If you have Conda 
installed on your system (see instructions at https://docs.conda.io/en/latest/miniconda.html), you can create a 
new environment with:

    conda create -n pysipfenn-workshop python=3.9 jupyter
    conda activate pysipfenn-workshop

And then simply install pySIPFENN from PyPI with

    pip install pysipfenn

Alternatively, you can also install pySIPFENN in editable mode if you cloned it from GitHub like

    git clone https://github.com/PhasesResearchLab/pySIPFENN.git

Or by downloading a ZIP file. Please note, this will by default download the latest development version of the 
software, which may not be stable. For a stable version, you can specify a version tag after the URL with
`--branch <tag_name> --single-branch`.

Then, move to the pySIPFENN folder and install in editable (`-e`) mode

    cd pySIPFENN
    pip install -e .

