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

## More Complex Issues

### Out-Of-Memory Error / Models Cannot Load

**RAM requirements to run:**

While pySIPFENN shouldn't have much trouble running on most modern desktop and protable
computers, it may have issues on older machines or on machines with limited RAM such as
VMs on cloud instances. In general, to run the models, you will need:   
- 512MB of RAM for Light Model (NN24)
- 2.5GB of RAM for KS2022 Novel Materials Model (NN30)
- **6GB of RAM to run all default models** (NN9, NN20, NN24, NN30) and follow the workshop tutorial

**RAM requirements to load:**

Now, one needs to consider how much memory is needed to get the models loaded into the
memory. This is a bit complicated, but in general, your system will need to use around twice the memory
given above. 

At the same time, if you are using any modern version of Windows or MacOS, you shouldn't have any issues with this 
limits if your system has at least 8GB of RAM (assuming no memory-intensive applications
running in the background) because they will utilize _pagefile_ (Windows) or _swap_ (MacOS) to automatically 
store the temporary data (as much as needed) that does not fit into RAM.

**Memory on Linux**

However, if you are using Linux, the _swap_ is not dynamic and if it's smaller than the
amount of memory you need, you will get an error or the process will just be 'killed' by the system. You can check
its size with

    free -h

And if the sum of RAM and swap is less than 12GB, you likely won't be able to use all of 
pySIPFENN models. Fortunately, it's an easy fix for your administrator, or if you have sudo 
access, yourself. To **temporarily increase swap** you just need to allocate some space on 
your drive (e.g. 8GB)

    sudo fallocate -l 8G /swapfile

limit access permissions for security reasons

    sudo chmod 600 /swapfile

tell Linux that it is a swap file 

    sudo mkswap /swapfile

and enable it

    sudo swapon /swapfile

Now, you should have no memory issues with running all pySIPFENN networks as of the time 
of writing (NN9, NN20, NN24, NN30). Please note that, as mentioned, this swap change is
temporary and will disappear when ou reboot. To make it permanent, you can follow one of
many guides available online.
