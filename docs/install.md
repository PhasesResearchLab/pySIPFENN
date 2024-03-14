# Install pySIPFENN

Installing pySIPFENN is simple and easy by utilizing **PyPI** package repository, **conda-forge**  package repository, or by cloning from **GitHub** directly.
While not required, it is recommended to first set up a virtual environment using venv or Conda. This ensures that (a) one of the required 
versions of Python (3.9+) is used and (b) there are no dependency conflicts. If you have Conda installed on your system (see [`miniconda` install instructions](https://docs.conda.io/en/latest/miniconda.html)), you can create a new environment with a simple:

    conda create -n pysipfenn python=3.10 jupyter numpy 
    conda activate pysipfenn

If you are managing a large set of dependencies in your project, you may consider using `mamba` in place of `conda`. It is a less mature, but much faster drop-in replacement compatible with existing environments. See [`micromamba` install instructions](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html).

## Standard

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

## Developer Install

If you want to utilize pySIPFENN beyond its core functionalities, for instance, to train new models on custom datasets or to export models in different 
formats or precisions, you need to install several other dependencies. This can be done by following the **from source** instructions above but appending
the last instruction with `dev` _extras_ marker.

```shell
pip install -e ".[dev]"
```

> Note: `pip install "pysipfenn[dev]"` will also work, but will be less conveninet for model modifications (which you likely want to do), as all persisted
> files will be located outside your working directory. You can quickly find where, by calling `import pysipfenn; c = pysipfenn.Calculator(); print(c)` and
> `Calculator` will tell you (amongst other things) where they are.