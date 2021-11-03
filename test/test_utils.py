import unittest

import numpy as np

from girth_mcmc.utils import validate_mcmc_options, default_mcmc_options
from girth_mcmc.utils import tag_missing_data_mcmc, get_discrimination_indices


class TestMCMCOptions(unittest.TestCase):
    """Test Fixture for MCMC Options."""

    def setUp(self):
        """Setup constructor."""
        self.number_of_keys = 7

    def test_default_options(self):
        """Testing default creation."""

        option_dict = default_mcmc_options()
        self.assertEqual(self.number_of_keys, len(option_dict.keys()))

        option_dict.pop('n_processors')
        self.assertDictEqual(
            option_dict,
            {
            "n_tune": 2500, "n_samples": 10000, 
            "variational_inference": False, 
            "variational_model": 'advi', 
            "variational_samples": 15000, 
            "initial_guess": True})

    def test_validate_options(self):
        """Validating MCMC Options."""

        option_dict = validate_mcmc_options({'n_processors': 4})
        self.assertEqual(self.number_of_keys, len(option_dict.keys()))
        self.assertDictEqual(
            option_dict,
            {"n_processors": 4,
            "n_tune": 2500, "n_samples": 10000, 
            "variational_inference": False, 
            "variational_model": 'advi', 
            "variational_samples": 15000, 
            "initial_guess": True})

        bad_keys = {"n_processors": "4",
            "n_tune": 54.3, "n_samples": 5235.23, 
            "variational_inference": 2, 
            "variational_model": 'advis', 
            "variational_samples": 15000.22, 
            "initial_guess": 'True'}

        for (key, value) in bad_keys.items():
            with self.assertRaises(AssertionError):
                validate_mcmc_options({key: value})

        with self.assertRaises(AssertionError):
            validate_mcmc_options([1, 2, 3])


class TestMissingValue(unittest.TestCase):
    """Test Fixture for missing data."""

    def test_missing_data(self):
        """Testing missing data."""
        rng = np.random.default_rng(3428089752309487572980345)

        random_data = rng.integers(0, 4, (500, 500))
        mask_bad = random_data < 2

        tagged_data = tag_missing_data_mcmc(random_data, [2, 3])
        
        np.testing.assert_equal(mask_bad, tagged_data.mask)


class TestDiscriminationIndices(unittest.TestCase):
    """Testing the discrimination indices."""

    def test_discrimination_indices(self):
        """Testing creating the discrimination indices."""

        n_items = [10, 20, 30]
        n_factors = [2, 3, 4, 5]

        for item in n_items:
            for factor in n_factors:
                diagonal_indices, lower_indices = get_discrimination_indices(item, 
                                                                             factor)
                lower_size = item * factor - factor * (factor + 1) / 2

                self.assertEqual(diagonal_indices[0].size, factor)
                self.assertEqual(diagonal_indices[1].size, factor)
            
                self.assertEqual(lower_indices[0].size, lower_size)
                self.assertEqual(lower_indices[1].size, lower_size)

                # Using lower, the rows are flipped, flip them back
                flipped = item - diagonal_indices[0] - 1
                np.testing.assert_equal(flipped, diagonal_indices[1])

                test = np.zeros((item, factor))
                test[lower_indices] = 1

                self.assertEqual(test.sum(), lower_size)

                for ndx1 in range(factor):
                    for ndx2 in range(ndx1, factor):
                        self.assertEqual(test[item - ndx1 -1, ndx2], 0)


if __name__ == "__main__":
    unittest.main()