

class LocalAdjuster:
    """
    Adjuster class operating on local data provided to model as a pair of (a) CSV file plus directory of POSCAR files
    or (b) NumPy array of feature vectors and NumPy array of target values. It can then adjust the model with some predefined
    hyperparameters or run a fairly typical grid search, which can be interpreted manually or uploaded to the ClearML
    platform.
    """

class OPTIMADEAdjuster(LocalAdjuster):
    """
    Adjuster class operating on data provided by the OPTIMADE API. Primarily geared towards tuning or retraining of the
    models based on other atomistic databases, or their subsets, accessed through OPTIMADE, to adjust the model to a
    different domain, which in the context of DFT datasets could mean adjusting the model to predict properties with DFT
    settings used by that database or focusing its attention to specific chemistry like, for instance, all compounds of
    Sn and all perovskites. It accepts OPTIMADE query as an input and then operates based on the ``LocalAdjuster`` class.
    """
