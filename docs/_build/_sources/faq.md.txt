# FAQ

This page is not meant to be an extensive documentation of how to address all questions
that may arrise when using pySIPFENN as a result of things outside our control, such as
interpreter configuration, permissions, operating system issues. These can often be 
resolved by searching the internet for the error message you are receiving.

## First things to check

In case you are having issues, it is good to first check the following few things that
helped other users.

### System and Architecture 

pySIPFENN should generally run on any operating system and computer architecture. We
    have tested it on 6 common combinations, which are listed below:
   - Windows 10/11 64-bit on x86_64 (i.e. Intel/AMD)
   - Windows 11 on ARM64 (e.g. Surface Pro X, Parallels VM on M1/M2 Macs)
   - Linux on x86_64 (i.e. Intel/AMD)
   - Linux on ARM64 (e.g. Raspberry Pi 4, Jetson Nano, etc.)
   - MacOS on x86_64 (most Macs from before 2021)
   - MacOS on ARM64 (Macs with M1/M2 chip)
   
Thus, it is unlikely that your operating system (PC vs Mac) or architecture (Intel 
vs M1/M2 Mac) is the issue. However, if you are using a different platform than above, please 
let us know, and we will try to help you.

### Python Version
Make sure you are using the correct version of Python. pySIPFENN requires Python 3.9 or
higher. We recommend 3.10 for longer support. The 3.11 increases computation speed but
was not tested as thoroughly and may have some issues with dependencies. 

If you are using a different version, you can either install a new version of
Python or use a virtual environment to install pySIPFENN in. We recommend using
Conda to create a virtual environment. See the [installation instructions](install.md)
making sure to specify the version of Python.

### Up-to-date Conda / pip 

Some users had problems with (1) loading the models or (2) making predictions using them
and we traced the issue to an outdated version of Conda installed on their work station
by their IT department. Updating Conda, pip, and reinstalling pySIPFENN fixed the issue.
You can try:

    conda update -n base conda -c anaconda
    conda update pip
    pip install --upgrade --force-reinstall pysipfenn

If you downloaded the models before, they should be retained as long as you do not
change the Conda environment. If you do, you will have to download them again.

### Model Download
One of the most common concerns users have contacted us about is the inability to download
the model files in reasonable time. This is usually correlated in time with a workshop, 
after which many users try to download them from Zenodo concurrently and exhaust the
bandwidth. In such a case, we recommend trying again in a day or two.

If your download is slow or fails during normal time periods, please let us know, and we
will try to help you by providing the files directly.