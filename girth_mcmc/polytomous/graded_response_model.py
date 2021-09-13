import pymc3 as pm
from numpy import linspace, zeros, unique

from girth_mcmc.utils import Rayleigh


__all__ = ["graded_response_model", "graded_response_parameters"]


def graded_response_model(dataset, n_categories):
    """Defines the mcmc model for the graded response model.
    
    Args:
        dataset: [n_items, n_participants] 2d array of measured responses
        n_categories: number of polytomous values (i.e. Number of Likert Levels)

    Returns:
        model: PyMC3 model to run
    """
    n_items, n_people = dataset.shape
    n_levels = n_categories - 1

    # Need small deviation in offset to
    # fit into pymc framework
    mu_value = linspace(-0.1, 0.1, n_levels)

    # Run through 0, K - 1
    observed = dataset - dataset.min()

    graded_mcmc_model = pm.Model()
    
    with graded_mcmc_model:
        # Ability Parameters
        ability = pm.Normal("Ability", mu=0, sigma=1, shape=n_people)
        
        # Discrimination multilevel prior
        rayleigh_scale = pm.Lognormal("Rayleigh_Scale", mu=0, sigma=1/4, shape=1)
        discrimination = pm.Bound(Rayleigh, lower=0.25)(name='Discrimination', 
                                  beta=rayleigh_scale, offset=0.25, shape=n_items)
        
        # Threshold multilevel prior
        sigma_difficulty = pm.HalfNormal('Difficulty_SD', sigma=1, shape=1)
        for ndx in range(n_items):
            thresholds = pm.Normal(f"Thresholds{ndx}", mu=mu_value, sigma=sigma_difficulty, 
                                   shape=n_levels, transform=pm.distributions.transforms.ordered)

            # Compute the log likelihood
            kernel = discrimination[ndx] * ability
            probabilities = pm.OrderedLogistic(f'Log_Likelihood{ndx}', cutpoints=thresholds, 
                                               eta=kernel, observed=observed[ndx])

    return graded_mcmc_model


def graded_response_parameters(trace):
    """Returns the parameters from an MCMC run.

    Args:
        trace: result from the mcmc run

    Return:
        return_dictionary: dictionary of found parameters
    """
    discrimination = trace['Discrimination'].mean(0)
    n_items = discrimination.shape[0]
    n_levels = max(map(lambda ndx: trace[f'Thresholds{ndx}'].shape[1], 
                       range(n_items)))
    thresholds = zeros((n_items, n_levels))
    
    for ndx in range(n_items):
        thresholds[ndx] = trace[f'Thresholds{ndx}'].mean(0) / discrimination[ndx]
    
    return {'Discrimination': discrimination,
            'Difficulty': thresholds, 
            'Ability': trace['Ability'].mean(0),
            'Difficulty Sigma': trace['Difficulty_SD'].mean(0),
            'Rayleigh Scale': trace['Rayleigh_Scale'].mean(0)} 