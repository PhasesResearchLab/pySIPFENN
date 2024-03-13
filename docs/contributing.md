# Contributing

## What to Contribute

If you wish to contribute to the development of pySIPFENN you are more than welcome to do so by forking the repository and creating a pull request. As of Spring
2024, we are actively developing the code and we should get back to you within a few days. We are also open to collaborations and partnerships, so if you have
an idea for a new feature or a new model, please do not hesitate to contact us through the GitHub issues or by `email <mailto:ak@psu.edu>`__.

In particular, we are seeking contributions in the following areas:

- **New Models**: We are always looking for new models to add to the repository. We have several (yet) unpublished ones for several different properties, so there is a good chance it will work for your case as well. We are happy to provide basic support for training, including using the default model for **transfer learning on small datasets**.

- **New Featurizers / Descriptor Sets**: We are always looking for new ways to featurize atomic configurations. 
    - We are **particularly interested** in including more domain-specific knowledge for different niches of materials science. Our KS2022 does a good job for most materials, but we look to expand it. 
    - We are **not looking for** featurizers that (a) cannot embed a structure into the feature space (e.g., most of the graph representations, which became popular in the last two years) or (b) do not encode physics into the feature space (e.g., raw atomic coordinates or 3D voxel representations).
    - Note: Autoencoders which utilize graph or 3D voxel representations to encode latent space position to predict property/properties fall into the first category and **are very welcome**.

- **Quality of Life Improvements**: We are always looking for ways to make the software easier to use and more efficient for users. If you have an idea for a new data parsing method, or a new way to visualize the results, we would love to hear about it.

## Rules for Contributing

We are currently very flexible with the rules for contributing, despite being quite opinionated :) 

Some general guidelines are:
- The ``core`` module is the only one that should be used by our typical end user. All **top-level APIs should be defined in the ``pysipfenn.py``** through the ``Calculator`` class. APIs operating _on_ the ``Calculator`` class, to export or retrain models, should be defined outside it, but within ``pysipfenn.core`` module.

- All **featurizers / descriptor calculators _must_ be self-contained in a single submodule** (file or directory) of ``pysipfenn.descriptorDefinitions`` (i.e., not spread around the codebase) and depend only on standard Python library and current pySIPFENN dependencies, including ``numpy``, ``torch``, ``pymatgen``, ``onnx``, ``tqdm``. If you need to add a new dependency, please discuss it with us first.

- All models **_must_ be ONNX models**, which can be obtained from almost any machine learning framework. We are happy to help with this process.

- All new classes, attributes, and methods **_must_ be type-annotated**. We are happy to help with this process.

- All new classes, attributes, and methods **_must_ have a well-styled docstring**. We are happy to help with this process.

- All functions, classes, and methods **_should_ have explicit inputs**, rather than passing a dictionary of parameters (*kwargs). This does require a bit more typing, but it makes the code much easier to use for the end user, who can see in the IDE exactly what parameters are available and what they do.

- All functions, classes, and methods **_should_ explain _why_ they are doing something, not just _what_** they are doing. This is critical for end-users who did not write the code and are trying to understand it. In particular, the default values of parameters should be explained in the docstring.

- All new features _must_ be tested with the ``pytest`` framework. **Coverage _should_ be 100%** for new code or close to it for good reasons. We are happy to help with this process.