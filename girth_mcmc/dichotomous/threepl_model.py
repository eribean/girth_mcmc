import pymc3 as pm

from girth_mcmc.utils import Rayleigh


__all__ = ["threepl_model", "threepl_parameters"]


def threepl_model(dataset):
    """Defines the mcmc model for three parameter logistic estimation.
    
    Args:
        dataset: [n_items, n_participants] 2d array of measured responses

    Returns:
        model: PyMC3 model to run
    """
    n_items, n_people = dataset.shape
    observed = dataset.astype('int')

    threepl_pymc_model = pm.Model()
    with threepl_pymc_model:
        # Ability Parameters (Standardized Normal)
        ability = pm.Normal("Ability", mu=0, sigma=1, shape=n_people)

        # Difficuly multilevel prior
        sigma_difficulty = pm.HalfNormal('Difficulty_SD', sigma=1, shape=1)
        difficulty = pm.Normal("Difficulty", mu=0, 
                               sigma=sigma_difficulty, shape=n_items)

        # Discrimination multilevel prior
        rayleigh_scale = pm.Lognormal("Rayleigh_Scale", mu=0, sigma=1/4, shape=1)
        discrimination = pm.Bound(Rayleigh, lower=0.25)(name='Discrimination', 
                                  beta=rayleigh_scale, offset=0.25, shape=n_items)

        # guessing prior
        exponential_lambda = pm.TruncatedNormal('Exponential_Scale',
                                                mu=15, sigma=2, shape=1,
                                                lower=10, upper=20)
        guessing = pm.Exponential('Guessing', lam=exponential_lambda, 
                                  shape=n_items)

        # Compute the probabilities
        kernel = ability[None, :] - difficulty[:, None]
        kernel *= discrimination[:, None]
        probabilities = pm.Deterministic("PL_Kernel", guessing[:, None] + 
                                         (1 - guessing[:, None]) * 
                                         pm.math.invlogit(kernel))

        # Get the log likelihood
        log_likelihood = pm.Bernoulli("Log_Likelihood", p=probabilities, observed=observed)

    return threepl_pymc_model
   

def threepl_parameters(trace):
    """Returns the parameters from an MCMC run.

    Args:
        trace: result from the mcmc run

    Return:
        return_dictionary: dictionary of found parameters
    """
    return {'Discrimination': trace['Discrimination'].mean(0),
            'Difficulty': trace['Difficulty'].mean(0),
            'Guessing': trace['Guessing'].mean(0),
            'Ability': trace['Ability'].mean(0),
            'Difficulty Sigma': trace['Difficulty_SD'].mean(),
            'Rayleigh Scale': trace['Rayleigh_Scale'].mean(),
            'Guessing Lambda': trace['Exponential_Scale'].mean()
            }