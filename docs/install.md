# Install pySIPFENN

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