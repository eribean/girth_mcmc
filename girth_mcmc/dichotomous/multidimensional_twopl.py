import numpy as np
import pymc3 as pm
import theano

from theano import tensor as tt

from girth.multidimensional import initial_guess_md


__all__= ["multidimensional_twopl_model", "multidimensional_twopl_parameters",
          "multidimensional_twopl_initial_guess"]


def _get_discrimination_indices(n_items, n_factors):
    """Local function to get parameters for discrimination estimation."""

    lower_indices = np.tril_indices(n_items, k=-1, m=n_factors)
    diagonal_indices = np.diag_indices(n_factors)
    lower_length = lower_indices[0].shape[0]

    # Set constraints to be the final items
    lower_indices = (n_items - 1 - lower_indices[0], lower_indices[1])
    diagonal_indices = (n_items - 1 - diagonal_indices[0], diagonal_indices[1])

    return diagonal_indices, lower_indices

def multidimensional_twopl_model(dataset, n_factors):
    """Defines the mcmc model for multidimensional 2PL logistic estimation.
    
    Args:
        dataset: [n_items, n_participants] 2d array of measured responses
        n_factors: (int) number of factors to extract

    Returns:
        model: PyMC3 model to run
    """
    if n_factors < 2:
        raise AssertionError(f"Multidimensional 2PL model requires "
                             f"two or more factors specified!")
    n_items, n_people = dataset.shape
    observed = dataset.astype('int')

    diagonal_indices, lower_indices = _get_discrimination_indices(n_items, n_factors)
    lower_length = lower_indices[0].shape[0]

    twopl_pymc_model = pm.Model()

    with twopl_pymc_model:
        # Ability Parameters (Standardized Normal)
        ability = pm.Normal("Ability", mu=0, sigma=1, shape=(n_factors, n_people))

        # Difficuly multilevel prior
        sigma_difficulty = pm.HalfNormal('Difficulty_SD', sigma=1, shape=1)
        difficulty = pm.Normal("Difficulty", mu=0, 
                            sigma=sigma_difficulty, shape=n_items)
        
        # The main diagonal must be non-negative
        discrimination = tt.zeros((n_items, n_factors), dtype=theano.config.floatX)
        diagonal_discrimination = pm.Lognormal('Diagonal Discrimination', mu=0, 
                                               sigma=0.25, shape=n_factors)
        lower_discrimination = pm.Normal('Lower Discrimination', sigma=1, 
                                          shape=lower_length)
        discrimination = tt.set_subtensor(discrimination[diagonal_indices], 
                                          diagonal_discrimination)

        discrimination = tt.set_subtensor(discrimination[lower_indices], 
                                          lower_discrimination)
        
        # Compute the probabilities
        kernel = pm.math.dot(discrimination, ability)
        kernel += difficulty[:, None]

        probabilities = pm.Deterministic("PL_Kernel", pm.math.invlogit(kernel))
        
        # Compute the log likelihood
        log_likelihood = pm.Bernoulli("Log_Likelihood", p=probabilities, observed=observed)

    return twopl_pymc_model


def multidimensional_twopl_parameters(trace):
    """Returns the parameters from an MCMC run.

    Args:
        trace: result from the mcmc run

    Return:
        return_dictionary: dictionary of found parameters
    """
    difficulty = trace['Difficulty'].mean(0)
    n_items = difficulty.shape[0]
    
    diagonal_entries = trace['Diagonal Discrimination'].mean(0)
    n_factors = diagonal_entries.shape[0]
    
    diagonal_indices, lower_indices = _get_discrimination_indices(n_items, n_factors)
    
    discrimination = np.zeros((n_items, n_factors))
    discrimination[lower_indices] = trace['Lower Discrimination'].mean(0)
    discrimination[diagonal_indices] = trace['Diagonal Discrimination'].mean(0)

    return {'Discrimination': discrimination,
            'Difficulty': difficulty,
            'Ability': trace['Ability'].mean(0).T,
            'Difficulty Sigma': trace['Difficulty_SD'].mean()}    


def multidimensional_twopl_initial_guess(dataset, n_factors):
    """Initializes initial guess for multidimensional twopl model.

    Args:
        dataset: [n_items, n_participants] 2d array of measured responses
        n_factors: (int) number of factors to extract

    Returns:
        estimated_discrimination: estimated discrimination parameters
    """
    n_items = dataset.shape[0]
    estimated_discrimination = initial_guess_md(dataset, n_factors)

    # Reformat into parameters for estimation
    diagonal_indices, lower_indices = _get_discrimination_indices(n_items, n_factors)

    return {'Diagonal Discrimination': estimated_discrimination[diagonal_indices],
            'Lower Discrimination': estimated_discrimination[lower_indices]}
    