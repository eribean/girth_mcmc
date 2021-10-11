import numpy as np

import pymc3 as pm

from girth_mcmc.utils import validate_mcmc_options
from girth_mcmc.dichotomous import (rasch_model, rasch_parameters,
                                    onepl_model, onepl_parameters,
                                    twopl_model, twopl_parameters,
                                    multidimensional_twopl_model, multidimensional_twopl_parameters,
                                    multidimensional_twopl_initial_guess, threepl_model,
                                    threepl_parameters)
from girth_mcmc.polytomous import  graded_response_model, graded_response_parameters


class GirthMCMC(object):
    """GIRTH MCMC class to run estimation models using PyMC3.

    Parameters:
        model: (string) ['Rasch', '1PL', '2PL', '3PL', 'GRM', '2PL_md'] which model to run
        model_args: (tuple) tuple of arguments to pass to model
        options: (dict) mcmc options dictionary
    
    Options:
        * n_processors: (int) number of processors
        * n_tune: number of "burn-in" samples to run 
        * n_samples: number of estimation samples
        * initial_guess: (boolean) use initial estimate in multidimensional
                         methods

    Notes:
        'GRM' requires setting the number of levels
        '2PL_md' requires setting the number of factors
    """

    def __init__(self, model, model_args=None, options=None):
        """Constructor method to run markov models."""
        self.options = validate_mcmc_options(options)
        self.model = model.lower()
        self.model_args = model_args

        # Trace Model, Parameters Extraction, Initial guess
        model_parameters = {
            # Unidimensional Models
            'rasch': (rasch_model, rasch_parameters, None),
            '1pl': (onepl_model, onepl_parameters, None),
            '2pl': (twopl_model, twopl_parameters, None),
            '3pl': (threepl_model, threepl_parameters, None),
            'grm': (graded_response_model, graded_response_parameters, None),

            # Multidimensional Models
            '2pl_md': (multidimensional_twopl_model, 
                       multidimensional_twopl_parameters,
                       multidimensional_twopl_initial_guess)
        }[model.lower()]

        self.pm_model = model_parameters[0]
        self.return_method = model_parameters[1]

        if self.options['initial_guess'] and model_parameters[2] is not None:
            self.initial_guess = model_parameters[2]

        else:
            self.initial_guess = lambda x, *args: None

        self.trace = None

    def build_model(self, dataset):
        """Builds the model to run.

            Args:
                dataset: [n_items, n_participants] 2d array of measured responses
            
            Returns:
                pymc_model: model ready to run
                initial_guess: dictionary of start values for sampler
        """
        if self.model_args:
            local_model = self.pm_model(dataset, *self.model_args)
            initial_guess = self.initial_guess(dataset, *self.model_args)

        else:
            local_model = self.pm_model(dataset)
            initial_guess = self.initial_guess(dataset)

        return local_model, initial_guess

    def __call__(self, dataset, **kwargs):
        """Begins the MCMC sampling process.
        
        Args:
            dataset: [n_items, n_participants] 2d array of measured responses
            kwargs: any named arguments passed to the trace, 
                    for variational methods, use 'inf_kwargs' to pass
                    arguments to fit function i.e. inf_kwargs={'jitter': 1}
        
        Returns:
            results_dictionary: dictionary of mean a posterori item values
        """
        # Run the sampling
        built_model, initial_guess = self.build_model(dataset)

        # Run the Model
        if self.options['variational_inference']:
            with built_model:
                result = pm.fit(method=self.options['variational_model'],
                                start=initial_guess,
                                n=self.options['variational_samples'], **kwargs)
            
            trace = result.sample(self.options['n_samples'])



        else: #MCMC Sampler
            n_tune = self.options['n_tune'] // self.options['n_processors']
            n_samples = self.options['n_samples'] // self.options['n_processors']

            with built_model:
                trace = pm.sample(n_samples, tune=n_tune,
                                chains=self.options['n_processors'], 
                                cores=self.options['n_processors'],
                                start=initial_guess,
                                return_inferencedata=False, **kwargs)
        
        # store the trace
        self.trace = trace

        # Return the values
        return self.return_method(trace)