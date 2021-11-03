import pymc3 as pm
from numpy import linspace, zeros, unique

import theano
from theano import tensor as tt

from girth.multidimensional import initial_guess_md
from girth_mcmc.utils import get_discrimination_indices


__all__= ["multidimensional_graded_model", "multidimensional_graded_parameters"]


def multidimensional_graded_model(dataset, n_categories, n_factors):
    """Defines the mcmc model for the multidimensional graded response model.
    
    Args:
        dataset: [n_items, n_participants] 2d array of measured responses
        n_categories: (int) number of polytomous values (i.e. Number of Likert Levels)
        n_factors: (int) number of factors to extract

    Returns:
        model: PyMC3 model to run
    """
    if n_factors < 2:
        raise AssertionError(f"Multidimensional GRM model requires "
                             f"two or more factors specified!")

    n_items, n_people = dataset.shape
    n_levels = n_categories - 1

    # Need small deviation in offset to
    # fit into pymc framework
    mu_value = linspace(-0.1, 0.1, n_levels)

    # Run through 0, K - 1
    observed = dataset - dataset.min()

    diagonal_indices, lower_indices = get_discrimination_indices(n_items, n_factors)
    lower_length = lower_indices[0].shape[0]

    graded_mcmc_model = pm.Model()
    
    with graded_mcmc_model:
        # Ability Parameters
        ability = pm.Normal("Ability", mu=0, sigma=1, shape=(n_factors, n_people))
        
        # Multidimensional Discrimination
        discrimination = tt.zeros((n_items, n_factors), dtype=theano.config.floatX)
        diagonal_discrimination = pm.Lognormal('Diagonal Discrimination', mu=0, 
                                               sigma=0.25, shape=n_factors)
        lower_discrimination = pm.Normal('Lower Discrimination', sigma=1, 
                                          shape=lower_length)
        discrimination = tt.set_subtensor(discrimination[diagonal_indices], 
                                          diagonal_discrimination)

        discrimination = tt.set_subtensor(discrimination[lower_indices], 
                                          lower_discrimination)
        
        # Threshold multilevel prior
        sigma_difficulty = pm.HalfNormal('Difficulty_SD', sigma=1, shape=1)
        for ndx in range(n_items):
            thresholds = pm.Normal(f"Thresholds{ndx}", mu=mu_value, sigma=sigma_difficulty, 
                                   shape=n_levels, transform=pm.distributions.transforms.ordered)

            # Compute the log likelihood
            kernel = pm.math.dot(discrimination[ndx], ability)
            probabilities = pm.OrderedLogistic(f'Log_Likelihood{ndx}', cutpoints=thresholds, 
                                               eta=kernel, observed=observed[ndx])

    return graded_mcmc_model


def multidimensional_graded_parameters(trace):
    """Returns the parameters from an MCMC run.

    Args:
        trace: result from the mcmc run

    Return:
        return_dictionary: dictionary of found parameters
    """
    n_factors = trace['Diagonal Discrimination'].shape[1]
    n_constraints = n_factors * (n_factors + 1) / 2 

    n_items = int((trace['Lower Discrimination'].shape[1] + n_constraints) / n_factors)
    n_levels = max(map(lambda ndx: trace[f'Thresholds{ndx}'].shape[1], 
                       range(n_items)))

    diagonal_indices, lower_indices = get_discrimination_indices(n_items, n_factors)

    discrimination = zeros((n_items, n_factors))
    discrimination[lower_indices] = trace['Lower Discrimination'].mean(0)
    discrimination[diagonal_indices] = trace['Diagonal Discrimination'].mean(0)

    thresholds = zeros((n_items, n_levels))    
    for ndx in range(n_items):
        thresholds[ndx] = trace[f'Thresholds{ndx}'].mean(0)
    
    return {'Discrimination': discrimination,
            'Difficulty': thresholds * -1, 
            'Ability': trace['Ability'].mean(0).T,
            'Difficulty Sigma': trace['Difficulty_SD'].mean(0)} 