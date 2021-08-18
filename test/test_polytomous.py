import unittest

import numpy as np

from girth import create_synthetic_irt_polytomous
from girth_mcmc import GirthMCMC


class TestPolytomous(unittest.TestCase):
    """Tests the mcmc for polytomous data."""

    # Only smoke tests for now

    def test_graded_response(self):
        """Testing the grm."""
        np.random.seed(46899)
        n_categories = 3

        difficulty = np.random.randn(5, n_categories-1)
        difficulty = np.sort(difficulty, 1)        
        discrimination = 0.96 * np.sqrt(-2 * np.log(np.random.rand(5)))
        theta = np.random.randn(150)

        syn_data = create_synthetic_irt_polytomous(difficulty, discrimination, 
                                                   theta, model='grm')

        girth_model = GirthMCMC(model='GRM', model_args=(n_categories,),
                                options={'n_tune': 1000, 'n_samples': 1000})
        result = girth_model(syn_data, progressbar=False)


class TestPolytomousVariational(unittest.TestCase):
    """Tests variational inference for polytomous data."""

    # Only smoke tests for now

    def test_graded_response(self):
        """Testing the grm."""
        np.random.seed(67841)
        n_categories = 3

        difficulty = np.random.randn(5, n_categories-1)
        difficulty = np.sort(difficulty, 1)        
        discrimination = 0.96 * np.sqrt(-2 * np.log(np.random.rand(5)))
        theta = np.random.randn(150)

        syn_data = create_synthetic_irt_polytomous(difficulty, discrimination, 
                                                   theta, model='grm')

        girth_model = GirthMCMC(model='GRM', model_args=(n_categories,),
                                options={'variational_inference': True,
                                         'variational_samples': 1000,
                                         'n_samples': 1000})
        result = girth_model(syn_data, progressbar=False)


if __name__ == '__main__':
    unittest.main()
