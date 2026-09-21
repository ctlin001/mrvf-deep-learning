"""MRvF: deep-learning vascular fingerprinting from GESFIDE MRI signals.

Layout
------
config              typed settings (parameter ranges, echo geometry, training hyper-parameters, data paths)
io                  read dictionaries / echo times from .mat files
preprocessing       L2 normalisation, parameter scaling, range/finite filtering
features            triple-regime R2* features (Part A / B / C of the GESFIDE signal)
models              TripleRegimeModel (FiLM-conditioned) and the DL-4p baseline Conv1DModel
data                build train / validation / test sets from the simulated dictionaries
training            loss and training loop
dictionary_matching inner-product dictionary matching baseline
evaluation          inference, metrics, RMSE vs SNR, feature ablation, significance tests
plotting            figures (imported lazily; needs matplotlib)
pipeline            run_training() and run_evaluation(): what the CLI and notebooks call
"""
__version__ = "0.1.0"
