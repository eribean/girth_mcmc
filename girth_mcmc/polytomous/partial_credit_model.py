import pymc3 as pm
from numpy import linspace, zeros, unique

from girth_mcmc.distributions import PartialCredit, Rayleigh


__all__ = ["partial_credit_model"]


def partial_credit_model(dataset, n_categories):
    """Defines the mcmc model for the partial credit model.
    
    Args:
        dataset: [n_items, n_participants] 2d array of measured responses
        n_categories: number of polytomous values (i.e. Number of Likert Levels)

    Returns:
        model: PyMC3 model to run
    """
    n_items, n_people = dataset.shape
    n_levels = n_categories - 1

    # Need small dither in offset to
    # fit into pymc framework
    mu_value = linspace(-0.05, 0.05, n_levels)

    # Run through 0, K - 1
    observed = dataset - dataset.min()

    partial_mcmc_model = pm.Model()
    
    with partial_mcmc_model:
        # Ability Parameters
        ability = pm.Normal("Ability", mu=0, sigma=1, shape=n_people)
        
        # Discrimination multilevel prior
        rayleigh_scale = pm.Lognormal("Rayleigh_Scale", mu=0, sigma=1/4, shape=1)
        discrimination = pm.Bound(Rayleigh, lower=0.25)(name='Discrimination', 
                                  beta=rayleigh_scale, offset=0.25, shape=n_items)
        
        # Threshold multilevel prior
        sigma_difficulty = pm.HalfNormal('Difficulty_SD', sigma=1, shape=1)

        # Possible Unorderd Categories
        for ndx in range(n_items):
            thresholds = pm.Normal(f"Thresholds{ndx}", mu=mu_value, 
                                   sigma=sigma_difficulty, shape=n_levels)

            # Compute the log likelihood
            kernel = discrimination[ndx] * ability
            probabilities = PartialCredit(f'Log_Likelihood{ndx}', cutpoints=thresholds, 
                                          eta=kernel, observed=observed[ndx])

    return partial_mcmc_model