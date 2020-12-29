import numpy as np

from multiprocessing import cpu_count


# Divide by two since multithreading doesn't count
DEFAULT_CPU = max(min(cpu_count() // 2, 2), 1)


def default_mcmc_options():
    """ Dictionary of options used in Girth MCMC.

    Args:
        n_processors: number of processors to use during runs (Default: 2)
        n_tune: number of "burn-in" samples to run (Default: 2500)
        n_samples: number of estimation samples (Default: 10000)

    Returns:
        options_dict: dictionary of options

    Notes:
        The n_tune and n_samples represent total samples, this will
        be divided by n_processors for multiprocessing
    """
    return {"n_processors": DEFAULT_CPU,
            "n_tune": 2500, "n_samples": 10000}


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
                    lambda x: isinstance(x, int) and x > 0,
                'n_samples':
                    lambda x: isinstance(x, int) and x > 7
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