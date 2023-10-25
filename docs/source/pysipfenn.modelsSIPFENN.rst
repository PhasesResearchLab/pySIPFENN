pySIPFENN Models
================

All default models for pySIPFENN are stored in the below Zenodo repository, which will be versioned with each release of
new models:

.. |Zenodo DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.7373089.svg?
    :target: https://doi.org/10.5281/zenodo.7373089
    :alt: Zenodo DOI

The model access is governed by the `models.json` file in the `pysipfenn/modelsSIPFENN` directory. For each model, the
file needs to contain:

* parent dictionary key matching the name of the model file (e.g. `SIPFENN_Krajewski2020_NN20`)
* `name` filed: string of the full name of the model displayed to the user. It can be anything.
* `descriptor` field: string of which descriptor / feature vector is taken as input to the model.
* `URL_ONNX` field: string of the URL to the ONNX model file.

additionally, the `models.json` file can contain the following optional fields:

* `URLjson` and `URLparams` fields: strings of the URL to the JSON and params files of an MxNet model. They are provided
  for legacy convenience and will be removed in some future release.

.. note:: In near future, the `models.json` file will also include model metadata such as: model version,
          training data information, model training hyperparameters, and key information about the model predictions
          including property/properties predicted, units, and range of applicability.

As of pySIPFENN v0.13.0, the `models.json` file contains the following models:

.. code-block:: JSON

    {
      "SIPFENN_Krajewski2020_NN9": {
        "name": "SIPFENN_Krajewski2020 Standard Materials Model",
        "URLjson": "https://zenodo.org/record/4279640/files/SIPFENN_Krajewski2020_NN9.json",
        "URLparams": "https://zenodo.org/record/4279640/files/SIPFENN_Krajewski2020_NN9.params",
        "URL_ONNX": "https://zenodo.org/record/7373089/files/SIPFENN_Krajewski2020_NN9.onnx",
        "descriptor": "Ward2017"
      },
      "SIPFENN_Krajewski2020_NN20": {
        "name": "SIPFENN_Krajewski2020 Novel Materials Model",
        "URLjson": "https://zenodo.org/record/4279640/files/SIPFENN_Krajewski2020_NN20.json",
        "URLparams": "https://zenodo.org/record/4279640/files/SIPFENN_Krajewski2020_NN20.params",
        "URL_ONNX": "https://zenodo.org/record/7373089/files/SIPFENN_Krajewski2020_NN20.onnx",
        "descriptor": "Ward2017"
      },
      "SIPFENN_Krajewski2020_NN24": {
        "name": "SIPFENN_Krajewski2020 Light Model",
        "URLjson": "https://zenodo.org/record/4279640/files/SIPFENN_Krajewski2020_NN24.json",
        "URLparams": "https://zenodo.org/record/4279640/files/SIPFENN_Krajewski2020_NN24.params",
        "URL_ONNX": "https://zenodo.org/record/7373089/files/SIPFENN_Krajewski2020_NN24.onnx",
        "descriptor": "Ward2017"
      },
      "SIPFENN_Krajewski2022_NN30": {
        "name": "SIPFENN_Krajewski2022 KS2022 Novel Materials Model",
        "URL_ONNX": "https://zenodo.org/record/7373089/files/SIPFENN_Krajewski2022_NN30.onnx",
        "descriptor": "KS2022"
      }
    }

Module Contents
---------------

.. automodule:: pysipfenn.modelsSIPFENN
   :members:
   :undoc-members:
   :show-inheritance:
