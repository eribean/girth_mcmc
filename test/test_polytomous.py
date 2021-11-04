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

    def test_partial_credit(self):
        """Testing Partial Credit Model."""
        rng = np.random.default_rng(84445166253145643984335315216)

        n_categories = 3
        difficulty = np.random.randn(5, n_categories-1)
        discrimination = 0.96 * np.sqrt(-2 * np.log(np.random.rand(5)))
        theta = np.random.randn(150)

        syn_data = create_synthetic_irt_polytomous(difficulty, discrimination, 
                                                   theta, model='pcm')

        girth_model = GirthMCMC(model='PCM', model_args=(n_categories,),
                                options={'n_tune': 1000, 'n_samples': 1000})
        result = girth_model(syn_data, progressbar=False)

    def test_multidimensional_grm(self):
        """Testing Multidimensional GRM."""
        rng = np.random.default_rng(29452344633211231635433213)

        n_categories = 3
        n_factors = 2

        discrimnation = rng.uniform(-2, 2, (20, n_factors))
        difficulty = np.sort(rng.standard_normal((20, n_categories - 1))*.5, axis=1)*-1        
        thetas = rng.standard_normal((n_factors, 250))

        syn_data = create_synthetic_irt_polytomous(difficulty, discrimnation, 
                                                   thetas, model='grm_md', seed=rng)

        girth_model = GirthMCMC(model='GRM_MD', model_args=(n_categories, n_factors),
                                options={'n_tune': 1000, 'n_samples': 1000})
        result = girth_model(syn_data, progressbar=False)

        with self.assertRaises(AssertionError):
            girth_model = GirthMCMC(model='GRM_MD', model_args=(n_categories, 1),
                                    options={'n_tune': 1000, 'n_samples': 1000})
            result = girth_model(syn_data, progressbar=False)

    def test_multidimensional_pcm(self):
        """Testing Multidimensional PCM."""
        rng = np.random.default_rng(29452344633211231635433213)

        n_categories = 3
        n_factors = 2

        discrimnation = rng.uniform(-2, 2, (20, n_factors))
        difficulty = np.sort(rng.standard_normal((20, n_categories - 1))*.5, axis=1)*-1        
        thetas = rng.standard_normal((n_factors, 250))

        syn_data = create_synthetic_irt_polytomous(difficulty, discrimnation, 
                                                   thetas, model='grm_md', seed=rng)

        girth_model = GirthMCMC(model='PCM_MD', model_args=(n_categories, n_factors),
                                options={'n_tune': 1000, 'n_samples': 1000})
        result = girth_model(syn_data, progressbar=False)

        with self.assertRaises(AssertionError):
            girth_model = GirthMCMC(model='PCM_MD', model_args=(n_categories, 1),
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

    def test_partial_credit(self):
        """Testing Partial Credit Model Variational Inference."""
        rng = np.random.default_rng(84445166253145643984335315216)

        n_categories = 3
        difficulty = np.random.randn(5, n_categories-1)
        discrimination = 0.96 * np.sqrt(-2 * np.log(np.random.rand(5)))
        theta = np.random.randn(150)

        syn_data = create_synthetic_irt_polytomous(difficulty, discrimination, 
                                                   theta, model='pcm')

        girth_model = GirthMCMC(model='PCM', model_args=(n_categories,),
                                options={'variational_inference': True,
                                         'variational_samples': 1000,
                                         'n_samples': 1000})

        result = girth_model(syn_data, progressbar=False)        

    def test_multidimensional_grm(self):
        """Testing Multidimensional Variational GRM."""
        rng = np.random.default_rng(8484959050677840349349)

        n_categories = 3
        n_factors = 2

        discrimnation = rng.uniform(-2, 2, (20, n_factors))
        difficulty = np.sort(rng.standard_normal((20, n_categories-1))*.5, axis=1)*-1        
        thetas = rng.standard_normal((n_factors, 250))

        syn_data = create_synthetic_irt_polytomous(difficulty, discrimnation, 
                                                   thetas, model='grm_md', seed=rng)

        girth_model = GirthMCMC(model='GRM_MD', model_args=(n_categories, n_factors),
                                options={'variational_inference': True,
                                         'variational_samples': 1000,
                                         'n_samples': 1000})
        result = girth_model(syn_data, progressbar=False)        

    def test_multidimensional_pcm(self):
        """Testing Multidimensional Variational PCM."""
        rng = np.random.default_rng(8484959050677840349349)

        n_categories = 3
        n_factors = 2

        discrimnation = rng.uniform(-2, 2, (20, n_factors))
        difficulty = np.sort(rng.standard_normal((20, n_categories-1))*.5, axis=1)*-1        
        thetas = rng.standard_normal((n_factors, 250))

        syn_data = create_synthetic_irt_polytomous(difficulty, discrimnation, 
                                                   thetas, model='grm_md', seed=rng)

        girth_model = GirthMCMC(model='PCM_MD', model_args=(n_categories, n_factors),
                                options={'variational_inference': True,
                                         'variational_samples': 1000,
                                         'n_samples': 1000})
        result = girth_model(syn_data, progressbar=False)       
if __name__ == '__main__':
    unittest.main()
