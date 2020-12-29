import unittest

import numpy as np

from girth_mcmc import (create_synthetic_irt_dichotomous, 
                        GirthMCMC)



class TestDichotomous(unittest.TestCase):
    """Tests the mcmc for dichotomous class."""

    # Only smoke tests for now

    def test_rasch(self):
        """Testing the rasch model."""
        np.random.seed(46899)
        difficulty = np.random.randn(10)
        theta = np.random.randn(100)

        syn_data = create_synthetic_irt_dichotomous(difficulty, 1, theta)

        girth_model = GirthMCMC(model='Rasch', 
                                options={'n_tune': 500, 'n_samples': 1000})
        result = girth_model(syn_data, progressbar=False)
        
    def test_onepl(self):
        """Testing the onepl model."""
        np.random.seed(86317)
        discrimination = 1.32
        difficulty = np.random.randn(10)
        theta = np.random.randn(100)

        syn_data = create_synthetic_irt_dichotomous(difficulty, discrimination, 
                                                    theta)

        girth_model = GirthMCMC(model='1PL', 
                                options={'n_tune': 500, 'n_samples': 1000})
        result = girth_model(syn_data, progressbar=False)

    def test_twopl(self):
        """Testing the twopl model."""
        np.random.seed(79987)
        discrimination = 0.89 * np.sqrt(-2 * np.log(np.random.rand(10)))
        difficulty = np.random.randn(10)
        theta = np.random.randn(100)

        syn_data = create_synthetic_irt_dichotomous(difficulty, discrimination, 
                                                    theta)

        girth_model = GirthMCMC(model='2PL', 
                                options={'n_tune': 500, 'n_samples': 1000})
        result = girth_model(syn_data, progressbar=False)

    def test_threepl(self):
        np.random.seed(8749)
        discrimination = 1.28 * np.sqrt(-2 * np.log(np.random.rand(10)))
        difficulty = np.random.randn(10)
        guessing = np.abs(np.random.rand(10)*0.05)
        theta = np.random.randn(100)

        syn_data = create_synthetic_irt_dichotomous(difficulty, discrimination, 
                                                    theta, guessing=guessing)

        girth_model = GirthMCMC(model='3PL', 
                                options={'n_tune': 500, 'n_samples': 1000})
        result = girth_model(syn_data, progressbar=False)


class TestDichotomousVariational(unittest.TestCase):
    """Tests variational inference for dichotomous class."""

    # Only smoke tests for now

    def test_rasch(self):
        """Testing the rasch model."""
        np.random.seed(46899)
        difficulty = np.random.randn(10)
        theta = np.random.randn(100)

        syn_data = create_synthetic_irt_dichotomous(difficulty, 1, theta)

        girth_model = GirthMCMC(model='Rasch', 
                                options={'variational_inference': True,
                                         'variational_samples': 1000,
                                         'n_samples': 1000})
        result = girth_model(syn_data, progressbar=False)

    def test_onepl(self):
        """Testing the onepl model."""
        np.random.seed(86317)
        discrimination = 1.32
        difficulty = np.random.randn(10)
        theta = np.random.randn(100)

        syn_data = create_synthetic_irt_dichotomous(difficulty, discrimination, 
                                                    theta)

        girth_model = GirthMCMC(model='1PL', 
                                options={'variational_inference': True,
                                         'variational_samples': 1000,
                                         'n_samples': 1000})
        result = girth_model(syn_data, progressbar=False)

    def test_twopl(self):
        """Testing the twopl model."""
        np.random.seed(79987)
        discrimination = 0.89 * np.sqrt(-2 * np.log(np.random.rand(10)))
        difficulty = np.random.randn(10)
        theta = np.random.randn(100)

        syn_data = create_synthetic_irt_dichotomous(difficulty, discrimination, 
                                                    theta)

        girth_model = GirthMCMC(model='2PL', 
                                options={'variational_inference': True,
                                         'variational_samples': 1000,
                                         'n_samples': 1000})
        result = girth_model(syn_data, progressbar=False)

    def test_threepl(self):
        np.random.seed(8749)
        discrimination = 1.28 * np.sqrt(-2 * np.log(np.random.rand(10)))
        difficulty = np.random.randn(10)
        guessing = np.abs(np.random.rand(10)*0.05)
        theta = np.random.randn(100)

        syn_data = create_synthetic_irt_dichotomous(difficulty, discrimination, 
                                                    theta, guessing=guessing)

        girth_model = GirthMCMC(model='3PL', 
                                options={'variational_inference': True,
                                         'variational_samples': 1000,
                                         'n_samples': 1000})
        result = girth_model(syn_data, progressbar=False)

if __name__ == '__main__':
    unittest.main()
