import unittest

import numpy as np

from girth_mcmc.utils import validate_mcmc_options, default_mcmc_options
from girth_mcmc.utils import tag_missing_data_mcmc


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


if __name__ == "__main__":
    unittest.main()