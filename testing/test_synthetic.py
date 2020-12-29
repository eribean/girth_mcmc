import unittest

import numpy as np

from girth_mcmc import create_synthetic_irt_dichotomous
from girth_mcmc import create_synthetic_mirt_dichotomous
from girth_mcmc import create_synthetic_irt_polytomous
from girth_mcmc import create_correlated_abilities

from girth_mcmc.synthetic import (_my_digitize, _credit_func, 
                                  _graded_func, _unfold_func,
                                  _check_difficulty_parameters)


class TestSynthetic(unittest.TestCase):
    """Testing the creation of synthetic irt function."""

    def test_synthetic_irt_creation(self):
        """Testing the creation of synthetic data."""
        seed = 31

        # Regression test
        expected = np.array([[False, False, False, False,  True,  True],
                             [False, False,  True,  True,  True,  True],
                             [False, False, False,  True,  True,  True]])

        value = create_synthetic_irt_dichotomous(np.array([1.2, -0.2, 1.3]),
                                                 1.31, np.linspace(-6, 6, 6),
                                                 seed=seed)

        np.testing.assert_array_equal(expected, value)


    def test_synthetic_mirt_creation(self):
        """Testing the creation of synthetic data."""
        seed = 164
        np.random.seed(seed-1)
        # Regression test
        expected = np.array([[False, False, False, False, False, False],
                             [ True, False, False,  True, False,  True],
                             [ True,  True, False, False, False, False],
                             [ True,  True,  True,  True, False,  True],
                             [ True,  True,  True,  True,  True,  True]])

        n_factors = 3
        n_items = 5
        n_people = 6
        discrimination = np.random.randn(n_items, n_factors)
        difficulty = np.linspace(-5, 5, n_items)
        thetas = np.random.randn(n_factors, n_people)
        value = create_synthetic_mirt_dichotomous(difficulty, discrimination,
                                                  thetas, seed)

        np.testing.assert_array_equal(expected, value)


    def test_synthetic_mirt_creation_single(self):
        """Testing the creation of synthetic data, common discrimination."""
        seed = 546
        np.random.seed(seed-1)
        # Regression test
        expected = np.array([[False, False, False, False, False, False],
                             [False, False, False, False, False, False],
                             [False, False, False,  True,  True,  True],
                             [False,  True,  True,  True,  True,  True],
                             [ True,  True,  True,  True,  True,  True]])

        n_factors = 3
        n_items = 5
        n_people = 6
        discrimination = np.random.randn(1, n_factors)
        difficulty = np.linspace(-5, 5, n_items)
        thetas = np.random.randn(n_factors, n_people)
        value = create_synthetic_mirt_dichotomous(difficulty, discrimination,
                                                  thetas, seed)

        np.testing.assert_array_equal(expected, value)


    def test_correlated_abilities(self):
        """Testing the creation of correlated abilities."""
        np.random.seed(120)
        n_participants = 1000
        rho = 0.73
        correlation_matrix = np.array([[1, rho], [rho, 1]])

        output = create_correlated_abilities(correlation_matrix, n_participants)
        output_corr = np.corrcoef(output)

        np.testing.assert_almost_equal(output_corr, correlation_matrix, decimal=1)


class TestPolytomousSynthetic(unittest.TestCase):
    """Testing the creation of synthetic polytomous irt function."""

    def test_my_digititize(self):
        """Testing local digitize function."""
        test_candidate = np.zeros((3,))

        # Zero position
        test_candidate[:] = [0.25, .3, .6]
        output = _my_digitize(test_candidate)

        self.assertEqual(output, 0)

        # One position
        test_candidate[:] = [0.3, .25, .6]
        output = _my_digitize(test_candidate)

        self.assertEqual(output, 1)

        # One position boundary
        test_candidate[:] = [0.3, .25, .3]
        output = _my_digitize(test_candidate)

        self.assertEqual(output, 1)

        # Two position 
        test_candidate[:] = [0.35, .25, .3]
        output = _my_digitize(test_candidate)

        self.assertEqual(output, 2)


    def test_check_difficulty_parameters(self):
        """Testing the validity check of polytomous difficulty."""
        np.random.seed(513)
        n_size = 5
        difficulty = np.random.rand(5, n_size)

        # PCM
        result = _check_difficulty_parameters(np.sort(difficulty, axis=1), 'grm')
        self.assertEqual(result, n_size + 1)

        # GRM
        with self.assertRaises(AssertionError):
            _check_difficulty_parameters(difficulty, 'grm')

        result = _check_difficulty_parameters(np.sort(difficulty, axis=1), 'grm')
        self.assertEqual(result, n_size + 1)

        #GUM
        difficulty = np.arange(n_size*n_size).reshape(n_size, n_size)

        result = _check_difficulty_parameters(difficulty, 'gum')
        self.assertEqual(result, n_size // 2 + 1)

        with self.assertRaises(AssertionError):
            _check_difficulty_parameters(difficulty[:, :-1], 'gum')

        difficulty[:, 0] = np.random.permutation(difficulty[:, 0])
        with self.assertRaises(AssertionError):
            _check_difficulty_parameters(difficulty, 'gum')


    def test_graded_function(self):
        """Testing the graded response model computation"""
        # Create basic data
        difficulties = np.array([-2.3, .3, 1.2])
        discrimination = 0.78
        thetas = np.linspace(-3, 3, 100)

        # Initialize output variable and call data
        output = np.zeros((difficulties.size + 1, thetas.size))
        _graded_func(difficulties, discrimination, thetas, output)

        # Compare to hand computations
        first_position = 1.0 / (1.0 + np.exp(discrimination * (thetas - difficulties[0])))
        second_position = 1.0 / (1.0 + np.exp(discrimination * (thetas - difficulties[1])))
        last_position = 1.0 / (1.0 + np.exp(discrimination * (thetas - difficulties[2])))

        np.testing.assert_array_almost_equal(output[0], first_position)
        np.testing.assert_array_almost_equal(output[1], second_position - first_position)
        np.testing.assert_array_almost_equal(output[2], last_position - second_position)
        np.testing.assert_array_almost_equal(output[3], 1 - last_position)
        

    def test_credit_function(self):
        """Testing the partial credit computation"""
        # Create basic data
        difficulties = np.array([-0.67, .24, .84])
        discrimination = 1.24
        thetas = np.linspace(-3, 3, 100)

        # Initialize output variable and call data
        output = np.zeros((difficulties.size + 1, thetas.size))
        _credit_func(difficulties, discrimination, thetas, output)

        # Compare to hand computations
        first_position = 1.0 
        second_position = np.exp(discrimination * (thetas - difficulties[0]))
        third_position = np.exp(discrimination * (thetas - difficulties[1])) * second_position
        last_position = np.exp(discrimination * (thetas - difficulties[2])) * third_position

        normalizing = first_position + second_position + third_position + last_position

        np.testing.assert_array_almost_equal(output[0], first_position / normalizing)
        np.testing.assert_array_almost_equal(output[1], second_position / normalizing)
        np.testing.assert_array_almost_equal(output[2], third_position / normalizing)
        np.testing.assert_array_almost_equal(output[3], last_position / normalizing)


    def test_unfolding_function(self):
        """Testing the graded unfolding model computation."""
        np.random.seed(831)
        offset = 0.3
        difficulties = np.array([-1.3, -.7, -.3, 0, .3, .7, 1.3])
        discrimination = 0.871
        thetas = np.linspace(-3, 3, 100)
        
        # Initialize output variable and call data
        output = np.zeros((difficulties.size + 1, thetas.size))
        _unfold_func(difficulties + offset, discrimination, thetas, output)

        # Compare to hand calculations
        span = difficulties.size / 2
        beta = discrimination * (thetas - offset)
        first_position = (np.cosh((span - 0) * beta) * 
                          np.exp(-discrimination * 0))
        second_position = (np.cosh((span - 1) * beta) * 
                           np.exp(-discrimination * difficulties[:1].sum()))
        third_position = (np.cosh((span - 2) * beta) * 
                           np.exp(-discrimination * difficulties[:2].sum()))
        last_position = (np.cosh((span - 3) * beta) * 
                         np.exp(-discrimination * difficulties[:3].sum()))

        normalizing = first_position + second_position + third_position + last_position

        np.testing.assert_array_almost_equal(output[0], first_position / normalizing)
        np.testing.assert_array_almost_equal(output[1], second_position / normalizing)
        np.testing.assert_array_almost_equal(output[2], third_position / normalizing)
        np.testing.assert_array_almost_equal(output[3], last_position / normalizing)

        # Test SRC works too
        _unfold_func(difficulties + offset, discrimination, thetas, output, src=1.)
        np.testing.assert_array_almost_equal(output[0] + output[-1], first_position / normalizing)
        np.testing.assert_array_almost_equal(output[1] + output[-2], second_position / normalizing)
        np.testing.assert_array_almost_equal(output[2] + output[-3], third_position / normalizing)
        np.testing.assert_array_almost_equal(output[3] + output[-4], last_position / normalizing)


    def test_create_polytomous_data_fail(self):
        """Testing synthetic polytomous function fails with 1 level"""
        np.random.seed(808)
        difficulty = np.array([[1.0]])
        bad_graded_difficulty = np.random.rand(4, 5)
        even_gum_difficulty = np.random.rand(4, 4)

        with self.assertRaises(AssertionError):
            create_synthetic_irt_polytomous(difficulty, difficulty, difficulty)

        with self.assertRaises(KeyError):
            create_synthetic_irt_polytomous([1, 2, 3], difficulty, 
                                            difficulty, model='boom')

        with self.assertRaises(AssertionError):
            create_synthetic_irt_polytomous(bad_graded_difficulty, 1.2,
                                            np.random.randn(100))
        
        with self.assertRaises(AssertionError):
            create_synthetic_irt_polytomous(even_gum_difficulty, 1.2,
                                            np.random.randn(100),
                                            model="gum")

        # Test for skew symmetric fail
        with self.assertRaises(AssertionError):
            create_synthetic_irt_polytomous(bad_graded_difficulty, 1.2,
                                            np.random.randn(100),
                                            model="gum")


    def test_check_polytomous_discrimination(self):
        """Smoke tests if a single value for discrimination passes"""
        difficulty = np.array([[1.0, 2., 3.]])
        discrimination = 3

        # Simple Smoke tests
        create_synthetic_irt_polytomous(difficulty, discrimination, difficulty)
        create_synthetic_irt_polytomous(difficulty, np.array([discrimination]), 
                                        difficulty)


    def test_check_polytomous_regression(self):
        """Regression testing graded and credit polytomous functions"""
        seed = 876
        np.random.seed(seed)
        difficulty_gum = np.array([np.linspace(-1.2, 1.2, 5),
                                   np.linspace(-.8, .8, 5),
                                   np.linspace(-1.7, 1.7, 5),
                                   np.linspace(-1.0, 1.0, 5)])
        difficulty_pcm = np.random.randn(5, 4)
        difficulty = np.sort(difficulty_pcm, axis=1)
        discrimination = 1.23
        thetas = np.random.randn(8)

        # Regression Tests
        poly_data_graded = create_synthetic_irt_polytomous(difficulty, 
                                                           discrimination,
                                                           thetas,
                                                           model='grm',
                                                           seed=seed)

        poly_data_credit = create_synthetic_irt_polytomous(difficulty_pcm, 
                                                           discrimination,
                                                           thetas,
                                                           model='pcm',
                                                           seed=seed)

        poly_data_gum = create_synthetic_irt_polytomous(difficulty_gum, 
                                                        discrimination,
                                                        thetas,
                                                        model='gum',
                                                        seed=seed)

        expected_graded = np.array([[5, 2, 2, 5, 2, 2, 5, 5],
                                    [5, 5, 1, 5, 5, 1, 2, 1],
                                    [5, 5, 1, 3, 2, 1, 5, 1],
                                    [3, 5, 3, 5, 4, 1, 4, 1],
                                    [5, 5, 3, 4, 3, 3, 5, 2]])

        expected_partial = np.array([[5, 5, 2, 5, 2, 1, 5, 3],
                                     [5, 5, 1, 5, 4, 1, 4, 1],
                                     [5, 5, 1, 3, 1, 1, 5, 1],
                                     [5, 5, 1, 5, 4, 1, 5, 1],
                                     [5, 5, 2, 5, 4, 2, 5, 2]])

        expected_unfold = np.array([[2, 1, 2, 2, 2, 2, 2, 3],
                                    [2, 2, 1, 2, 3, 1, 1, 1],
                                    [2, 3, 2, 2, 3, 1, 2, 1],
                                    [1, 3, 3, 3, 3, 1, 1, 1]])

        np.testing.assert_array_equal(poly_data_graded, expected_graded)
        np.testing.assert_array_equal(poly_data_credit, expected_partial)
        np.testing.assert_array_equal(poly_data_gum, expected_unfold)

if __name__ == '__main__':
    unittest.main()
