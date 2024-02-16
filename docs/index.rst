.. image:: _static/SIPFENN_logo_small.png
    :width: 250pt
    :alt: logo
    :align: center

=========
pySIPFENN
=========

|GitHub top language| |PyPI - Python Version| |GitHub license| |PyPI Version| |PyPI Downloads|

|GitHub last commit| |GitHub last release| |GitHub issues| |GitHub commits since previous| |GitHub commits since last| 

|Full| |Linux| |MacM1| |MacIntel| |Windows| |Coverage Status|

|Paper DOI| |Zenodo DOI|

.. |GitHub top language| image:: https://img.shields.io/github/languages/top/PhasesResearchLab/pysipfenn
    :alt: GitHub top language
    :target: https://github.com/PhasesResearchLab/pySIPFENN

.. |PyPI - Python Version| image:: https://img.shields.io/pypi/pyversions/pysipfenn
    :alt: PyPI - Python Version
    :target: https://www.python.org/downloads/release/python-3100/

.. |PyPI Version| image:: https://img.shields.io/pypi/v/pysipfenn?label=PyPI&color=green
    :target: https://pypi.org/project/pysipfenn/
    :alt: PyPI

.. |PyPI Downloads| image:: https://img.shields.io/pypi/dm/pysipfenn
    :target: https://pypi.org/project/pysipfenn/
    :alt: PyPI

.. |Full| image:: https://github.com/PhasesResearchLab/pySIPFENN/actions/workflows/fullTest.yaml/badge.svg
    :alt: Build Status
    :target: https://github.com/PhasesResearchLab/pySIPFENN/actions/workflows/fullTest.yaml
    
.. |Linux| image:: https://github.com/PhasesResearchLab/pySIPFENN/actions/workflows/coreTests_LinuxUbuntu.yaml/badge.svg
    :alt: Linux Status 
    :target: https://github.com/PhasesResearchLab/pySIPFENN/actions/workflows/coreTests_LinuxUbuntu.yaml

.. |MacM1| image:: https://github.com/PhasesResearchLab/pySIPFENN/actions/workflows/coreTests_MacM1.yaml/badge.svg
    :alt: Mac M1 Status 
    :target: https://github.com/PhasesResearchLab/pySIPFENN/actions/workflows/coreTests_MacM1.yaml

.. |MacIntel| image:: https://github.com/PhasesResearchLab/pySIPFENN/actions/workflows/coreTests_MacIntel.yaml/badge.svg
    :alt: Mac M1 Status 
    :target: https://github.com/PhasesResearchLab/pySIPFENN/actions/workflows/coreTests_MacIntel.yaml

.. |Windows| image:: https://github.com/PhasesResearchLab/pySIPFENN/actions/workflows/coreTests_Windows.yaml/badge.svg
    :alt: Windows Status 
    :target: https://github.com/PhasesResearchLab/pySIPFENN/actions/workflows/coreTests_Windows.yaml

.. |Coverage Status| image:: https://codecov.io/gh/PhasesResearchLab/pySIPFENN/branch/main/graph/badge.svg?token=S2J0KR0WKQ
    :alt: Coverage Status
    :target: https://codecov.io/gh/PhasesResearchLab/pySIPFENN

.. |GitHub license| image:: https://img.shields.io/badge/License-LGPL_v3-blue.svg
    :alt: GitHub license
    :target: https://www.gnu.org/licenses/lgpl-3.0

.. |GitHub last commit| image:: https://img.shields.io/github/last-commit/PhasesResearchLab/pySIPFENN?label=Last%20Commit
    :alt: GitHub last commit (by committer)
    :target: https://github.com/PhasesResearchLab/pySIPFENN/commits/main

.. |GitHub last release| image:: https://img.shields.io/github/release-date/PhasesResearchLab/pysipfenn?label=Last%20Release
    :alt: GitHub Release Date - Published_At
    :target: https://github.com/PhasesResearchLab/pySIPFENN/releases

.. |GitHub commits since previous| image:: https://img.shields.io/github/commits-since/PhasesResearchLab/pysipfenn/v0.13.0?color=g
    :alt: GitHub commits since previous
    :target: https://github.com/PhasesResearchLab/pySIPFENN/releases

.. |GitHub commits since last| image:: https://img.shields.io/github/commits-since/PhasesResearchLab/pysipfenn/v0.15.0?color=g
    :alt: GitHub commits since last
    :target: https://github.com/PhasesResearchLab/pySIPFENN/releases

.. |GitHub issues| image:: https://img.shields.io/github/issues/PhasesResearchLab/pySIPFENN
    :alt: GitHub issues
    :target: https://github.com/PhasesResearchLab/pySIPFENN/issues

.. |Paper DOI| image:: https://img.shields.io/badge/DOI-10.1016%2Fj.commatsci.2022.111254-blue
    :target: https://doi.org/10.1016/j.commatsci.2022.111254
    :alt: Paper DOI

.. |Zenodo DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.7373089.svg?
    :target: https://doi.org/10.5281/zenodo.7373089
    :alt: Zenodo DOI

**py** (**S** tructure - **I** nformed **P** rediction of
**F** ormation **E** nergy using **N** eural **N** etworks)
software package allows efficient predictions of the energetics of
atomic configurations. The underlying methodology and implementation
is given in

- Adam M. Krajewski, Jonathan W. Siegel, Jinchao Xu, Zi-Kui Liu, Extensible Structure-Informed Prediction of Formation Energy with improved accuracy and usability employing neural networks, Computational Materials Science, Volume 208, 2022, 111254 `(https://doi.org/10.1016/j.commatsci.2022.111254) <https://doi.org/10.1016/j.commatsci.2022.111254>`_

While functionalities are similar to the software released along the 
paper, this package contains improved methods for featurizing atomic 
configurations. Notably, all of them are now written completely in 
Python, removing reliance on Java and making extensions of the software
much easier thanks to improved readability.

News
----

- **(v0.15.0)** A new descriptor (feature vector) calculator ``descriptorDefinitions.KS2022_randomSolutions`` has been implemented. It is used 
  for structure informed featurization of compositions randomly occupying a lattice, spiritually similar to SQS generation, but also taking into 
  account (1) chemical differences between elements and (2) structural effects. A full description will be given in the upcoming manuscript.

- **(v0.14.0)** Users can now take advantage of a **Prototype Library** to obtain common structures from any ``Calculator`` instance ``c`` with a 
  simple ``c.prototypeLibrary['BCC']['structure']``. It can be easily 
  `updated <https://pysipfenn.readthedocs.io/en/latest/source/pysipfenn.core.html#pysipfenn.Calculator.parsePrototypeLibrary>`__ 
  or `appended <https://pysipfenn.readthedocs.io/en/latest/source/pysipfenn.core.html#pysipfenn.Calculator.appendPrototypeLibrary>`__ with high-level 
  API or by manually modifyig its YAML `here <https://github.com/PhasesResearchLab/pySIPFENN/blob/main/pysipfenn/misc/prototypeLibrary.yaml>`__.

- **(v0.13.0)** Model exports (and more!) to PyTorch, CoreML, and ONNX are now effortless thanks to 
  ``core.modelExporters`` module. Please note you need to install pySIPFENN with ``dev`` option (e.g., ``pip install "pysipfenn[dev]"``) to use it. 
  See `docs here <https://pysipfenn.readthedocs.io/en/stable/source/pysipfenn.core.html#module-pysipfenn.core.modelExporters>`__.

- **(v0.12.2)** Swith to LGPLv3 allowing for integration with proprietary software developed by CALPHAD community, while supporting the development 
  of new pySIPFENN features for all. Many thanks to our colleagues from `GTT-Technologies <https://gtt-technologies.de>`__ 
  and other participants of `CALPHAD 2023 <https://calphad.org/calphad-2023>`__` for fruitful discussions.

- **(March 2023 Workshop)** We would like to thank all of our amazing attendees for making our workshop, co-organized with the
  `Materials Genome Foundation <https://materialsgenomefoundation.org>`__, such a success! Over 100 of you simultaneously followed
  all exercises and, at the peak, we loaded over 1,200GB of models into the HPC's RAM. 

.. note::
   This project is under active development. We recommend using released (stable) versions.

Index
-----

.. toctree::
   install
   source/pysipfenn
   exportingmodels
   faq
   miscellaneousnotes
   examples/sipfenn_examples
   Journal Article <https://doi.org/10.1016/j.commatsci.2022.111254>
   changelog
   contributing
   genindex
   :maxdepth: 2
   :caption: Contents

Applications
------------

pySIPFENN is a very flexible tool that can, in principle, be used for
the prediction of any property of interest that depends on an atomic
configuration with very few modifications. The models shipped by
default are trained to predict formation energy because that is what our
research group is interested in; however, if one wanted to predict
Poisson’s ratio and trained a model based on the same features, adding
it would take minutes. Simply add the model in open ONNX format and link
it using the *models.json* file, as described in the documentation.

Real-World Examples
-------------------

In our line of work, pySIPFENN and the formation energies it predicts are
usually used as a computational engine that generates proto-data for
creation of thermodynamic databases (TDBs) using ESPEI
(https://espei.org). The TDBs are then used through pycalphad
(https://pycalphad.org) to predict phase diagrams and other
thermodynamic properties.

Another of its uses in our research is guiding the Density of Functional
Theory (DFT) calculations as a low-cost screening tool. Their efficient
conjunction then drives experiments leading to the discovery of new
materials, as presented in these two papers:

-  Sanghyeok Im, Shun-Li Shang, Nathan D. Smith, Adam M. Krajewski,
   Timothy Lichtenstein, Hui Sun, Brandon J. Bocklund, Zi-Kui Liu,
   Hojong Kim, Thermodynamic properties of the Nd-Bi system via emf
   measurements, DFT calculations, machine learning, and CALPHAD
   modeling, Acta Materialia, Volume 223, 2022, 117448,
   https://doi.org/10.1016/j.actamat.2021.117448.

-  Shun-Li Shang, Hui Sun, Bo Pan, Yi Wang, Adam M. Krajewski, Mihaela
   Banu, Jingjing Li & Zi-Kui Liu, Forming mechanism of equilibrium and
   non-equilibrium metallurgical phases in dissimilar aluminum/steel
   (Al–Fe) joints. Nature Scientific Reports 11, 24251 (2021).
   https://doi.org/10.1038/s41598-021-03578-0
