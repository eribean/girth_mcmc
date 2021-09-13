import pymc3 as pm

__all__ = ['rasch_model', 'rasch_parameters']


def rasch_model(dataset):
    """Defines the mcmc model for Rasch estimation.
    
    Args:
        dataset: [n_items, n_participants] 2d array of measured responses

    Returns:
        model: PyMC3 model to run
    """
    n_items, n_people = dataset.shape
    observed = dataset.astype('int')

    rasch_pymc_model = pm.Model()
    with rasch_pymc_model:
        # Ability Parameters (Standardized Normal)
        ability = pm.Normal("Ability", mu=0, sigma=1, shape=n_people)

        # Difficuly multilevel prior
        sigma_difficulty = pm.HalfNormal('Difficulty_SD', sigma=1, shape=1)
        difficulty = pm.Normal("Difficulty", mu=0, 
                               sigma=sigma_difficulty, shape=n_items)

        # Compute the probabilities
        kernel = ability[None, :] - difficulty[:, None]
        probabilities = pm.Deterministic("PL_Kernel", pm.math.invlogit(kernel))

        # Get the log likelihood
        log_likelihood = pm.Bernoulli("Log_Likelihood", p=probabilities, observed=observed)

    return rasch_pymc_model


def rasch_parameters(trace):
    """Returns the parameters from an MCMC run.

    Args:
        trace: result from the mcmc run

    Return:
        return_dictionary: dictionary of found parameters
    """
    return {'Difficulty': trace['Difficulty'].mean(0),
            'Ability': trace['Ability'].mean(0),
            'Difficulty_sigma': trace['Difficulty_SD'].mean()
            }
   