import numpy as np
from pymc3 import ADVI

from multiprocessing import cpu_count


# Divide by two since multithreading doesn't count
DEFAULT_CPU = max(min(cpu_count() // 2, 2), 1)


__all__ = ['default_mcmc_options', 'validate_mcmc_options']


def default_mcmc_options():
    """ Dictionary of options used in Girth MCMC.

    Args:
        n_processors: number of processors to use during runs (Default: 2)
        n_tune: number of "burn-in" samples to run (Default: 2500)
        n_samples: number of estimation samples (Default: 10000)
        variational_inference: use variational estimation (Default: False)
        variational_model: String of varational model to use 
                           ['advi', 'svgd', 'fullrank_advi'] (Default: 'advi')
        variational_samples: number of samples to use in VI (Default: 15000)

    Returns:
        options_dict: dictionary of options

    Notes:
        The n_tune and n_samples represent total samples, this will
        be divided by n_processors for multiprocessing

        More info about Variational Models at:
        https://docs.pymc.io/api/inference.html#variational-inference
    """
    return {"n_processors": DEFAULT_CPU,
            "n_tune": 2500, "n_samples": 10000,
            "variational_inference": False, 
            "variational_model": 'advi',
            "variational_samples": 15000,
            "initial_guess": True}


def validate_mcmc_options(options_dict=None):
    """ Validates an options dictionary.

    Args:
        options_dict: Dictionary with updates to default_values

    Returns:
        options_dict: Updated dictionary

    """
    validate = {'n_processors':
                    lambda x: isinstance(x, int) and x > 0,
                'n_tune':
                    lambda x: isinstance(x, int) and x > 100,
                'n_samples':
                    lambda x: isinstance(x, int) and x > 100,
                'variational_inference':
                    lambda x: isinstance(x, bool),
                'variational_model':
                    lambda x: x in ['advi', 'svgd', 'fullrank_advi'],
                'variational_samples':
                    lambda x: isinstance(x, int) and x > 100,
                "initial_guess":
                    lambda x: isinstance(x, bool)
                }
    
    # A complete options dictionary
    full_options = default_mcmc_options()
    
    if options_dict:
        if not isinstance(options_dict, dict):
            raise AssertionError("Options must be a dictionary got: "
                                f"{type(options_dict)}.")

        for key, value in options_dict.items():
            if not validate[key](value):
                raise AssertionError("Unexpected key-value pair: "
                                     f"{key}: {value}. Please see "
                                     "documentation for expected inputs.")

        full_options.update(options_dict)

    return full_options