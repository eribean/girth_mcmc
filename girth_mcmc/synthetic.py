import numpy as np
from scipy.special import expit


def irt_evaluation(difficulty, discrimination, thetas):
    """ Evaluation of unidimensional IRT model.

    Evaluates an IRT model and returns the exact values.  This function
    supports only unidimemsional models

    Assumes the model
        P(theta) = 1.0 / (1 + exp(discrimination * (theta - difficulty)))

    Args:
        difficulty: (1d array) item difficulty parameters
        discrimination:  (1d array | number) item discrimination parameters
        thetas: (1d array) person abilities

    Returns:
        probabilities: (2d array) evaluation of sigmoid for given inputs
    """
    # If discrimination is a scalar, make it an array
    if np.atleast_1d(discrimination).size == 1:
        discrimination = np.full_like(difficulty, discrimination,
                                      dtype='float')

    kernel = thetas - difficulty[:, None]
    kernel *= discrimination[:, None]
    return expit(kernel)


def create_correlated_abilities(correlation_matrix, n_participants):
    """ Creates correlated ability parameters based on an input correlation matrix.

    This is a helper function for use in synthesizing multi-dimensional data
    assuming multivariate normal distribution

    Args:
        correlation_matrix: (2d array) Symmetric matrix defining
                            the correlation between the abilities
        n_participants: number of participants to synthesize

    Returns:
        abilities: (2d array) correlated abilities
    """
    lower = np.linalg.cholesky(correlation_matrix)

    return lower @ np.random.randn(correlation_matrix.shape[0], n_participants)


def create_synthetic_irt_dichotomous(difficulty, discrimination, thetas,
                                     guessing=0, seed=None):
    """ Creates dichotomous unidimensional synthetic IRT data.

    Creates synthetic IRT data to test parameters estimation functions.  
    Only for use with dichotomous outputs

    Assumes the model
        P(theta) = 1.0 / (1 + exp(discrimination * (theta - difficulty)))

    Args:
        difficulty: [array] of difficulty parameters
        discrimination:  [array | number] of discrimination parameters
        thetas: [array] of person abilities
        guessing: [array | number] of guessing parameters associated with items
        seed: Optional setting to reproduce results

    Returns:
        synthetic_data: (2d array) realization of possible response given parameters

    """
    if seed:
        np.random.seed(seed)

    if np.ndim(guessing) < 1:
        guessing = np.full_like(difficulty, guessing)

    continuous_output = irt_evaluation(difficulty, discrimination, thetas)

    # Add guessing parameters
    continuous_output *= (1.0 - guessing[:, None])
    continuous_output += guessing[:, None]

    # convert to binary based on probability
    random_compare = np.random.rand(*continuous_output.shape)

    return (random_compare <= continuous_output).astype('int')


def create_synthetic_mirt_dichotomous(difficulty, discrimination, thetas,
                                      seed=None):
    """ Creates dichotomous multidimensional synthetic IRT data.

    Assumes the model
        P(theta) = 1.0 / (1 + exp(-1 * (dot(discrimination,theta) + difficulty)))

    Args:
        difficulty: [array, M] of difficulty parameters
        discrimination:  [2-D array, MxN] of discrimination parameters
        thetas: [2-D array, NxP] of person abilities
        seed: Optional setting to reproduce results

    Returns:
        synthetic_data: (2d array) realization of possible response given parameters


    Example:
        n_factors = 3
        n_items = 15
        n_people = 500
        difficulty = np.linspace(-2.5, 2.5, n_items)
        discrimination = np.random.randn(n_items, n_factors)
        thetas = np.random.randn(n_factors, n_people)

        synthetic_data = create_synthetic_mirt_dichotomous(difficulty, discrimination, thetas)
    """
    if seed:
        np.random.seed(seed)

    # If the input is just a vector of discriminations
    if (np.ndim(discrimination) == 1) or (discrimination.shape[0] == 1):
        discrimination = np.vstack((discrimination,) * difficulty.shape[0])

    # Inline computation of the logistic kernel
    kernel_terms = discrimination @ thetas
    kernel_terms += difficulty[:, None]
    continuous_output = expit(kernel_terms)

    # convert to binary based on probability
    random_compare = np.random.rand(*continuous_output.shape)

    return (random_compare <= continuous_output).astype('int')


# Private functions for polytomous outputs
def _my_digitize(the_input):
    """
    Private function to compute polytomous levels.
    The input has been concatenated to use the
    vectorize functions (value, thresholds)       
    """
    return np.searchsorted(the_input[1:], the_input[0])


def _graded_func(difficulty, discrimination, thetas, output):
    """
    Private function to compute the probabilities for
    the graded response model.  This is done in place
    and does not return anything
    """
    # This model is based on the difference of standard
    # logistic functions.

    # Do first level
    output[0] = 1.0 - irt_evaluation(np.array([difficulty[0]]),
                                     discrimination, thetas)

    for level_ndx in range(1, output.shape[0]-1):
        right = irt_evaluation(np.array([difficulty[level_ndx]]),
                               discrimination, thetas)
        left = irt_evaluation(np.array([difficulty[level_ndx-1]]),
                              discrimination, thetas)
        output[level_ndx] = left - right

    # Do last level
    output[-1] = irt_evaluation(np.array([difficulty[-1]]),
                                discrimination, thetas)


def _credit_func(difficulty, discrimination, thetas, output):
    """
    Private function to compute the probabilities for
    the partial credit model.  This is done in place
    and does not return anything
    """
    # This model is based on exponentials and normalized to
    # make sure the expected probablity is equal to one
    output *= 0.0  # clear any previous values
    output[1:, :] += thetas
    output[1:, :] -= difficulty[:, None]
    output *= discrimination
    np.cumsum(output, axis=0, out=output)
    np.exp(output, out=output)

    normalizing_term = 1.0 / np.sum(output, axis=0)
    output *= normalizing_term


def _unfold_func(difficulty, discrimination, thetas, output, src=0.):
    """
    Private function to compute the probabilities for
    the graded unfolding model.  This is done in place
    and does not return anything
    """
    # Call partial credit model on difficulties
    _credit_func(difficulty, discrimination, thetas, output)

    # Add the probabilities together, should always be even
    for ndx in range(output.shape[0] // 2):
        output[ndx] += (output[-(ndx+1)] * (1. - src))
        output[-(ndx+1)] *= src


def _check_difficulty_parameters(difficulty, model):
    """
    Given a model type, check the difficulty parameters
    for validity
    """
    max_value = difficulty.shape[1] + 1

    if model in ["grm"]:
        # Check that all the arguments are sorted
        if not np.all(difficulty[:, :-1] < difficulty[:, 1:]):
            raise AssertionError("Difficulty Parameters must be "
                                 "in ascending order")

    elif model in ['gum']:
        # Parameters must be odd
        if max_value % 2:
            raise AssertionError("There must be an odd number of "
                                 "difficulty parameters")

        # Parameters must be skew-symmetric about the center point
        middle_index = (difficulty.shape[1] - 1) // 2
        adjusted_difficulty = (difficulty -
                               difficulty[:, middle_index][:, None])

        if not np.all(np.abs(adjusted_difficulty.sum(axis=1)) < 1e-7):
            raise AssertionError("Difficulty Parameters must be "
                                 "symmetric about offset")

        max_value = middle_index + 1

    return max_value


def create_synthetic_irt_polytomous(difficulty, discrimination, thetas,
                                    model='grm', seed=None):
    """ Creates polytomous unidimensional synthetic IRT data.

    Creates polytomous output with specified number of levels from [1, levels]

    Args:
        difficulty: [2D array (items x n_levels-1)] of difficulty parameters
        discrimination:  [array | number] of discrimination parameters
        thetas: [array] of person abilities
        model: ["grm", "pcm", "gum"] string specifying which polytomous model to use
                'grm': Graded Response Model
                'pcm': Generalized Partial Credit Model
                'gum': Generalized Graded Unfolding Model
        seed: Optional setting to reproduce results

    Returns:
        synthetic_data: (2d array) realization of possible response given parameters
    """
    difficulty = np.atleast_2d(difficulty)
    n_items, n_levels = difficulty.shape

    if n_levels == 1:
        raise AssertionError("Polytomous items must have more than 1 threshold")

    if seed:
        np.random.seed(seed)

    # Check for single input of discrimination
    if np.atleast_1d(discrimination).size == 1:
        discrimination = np.full((n_items,), discrimination)

    # Get the model to use, will throw error if not supported
    probability_func = {'grm': _graded_func,
                        'pcm': _credit_func,
                        'gum': _unfold_func}[model.lower()]

    # Check difficulty parameters for validity
    clip_high = _check_difficulty_parameters(difficulty, model.lower())

    # Initialize output for memory concerns
    level_scratch = np.zeros((n_levels + 2, thetas.size))
    output = np.zeros((n_items, thetas.size), dtype='int')

    # Loop over items and compute probability estimates
    # for each of the levels and assign level based on
    # those probabilities
    for item_ndx in range(n_items):
        # Obtain the probabilities for the data (in-place)
        probability_func(difficulty[item_ndx], discrimination[item_ndx],
                         thetas, level_scratch[1:, :])

        # Get the thresholds of the levels
        np.cumsum(level_scratch[1:, :], axis=0, out=level_scratch[1:, :])
        level_scratch[0] = np.random.rand(thetas.size)

        # Discritize the outputs based on the thresholds
        output[item_ndx] = np.apply_along_axis(
            _my_digitize, axis=0, arr=level_scratch)

    # Add 1 to return [1, n_levels]
    output += 1
    np.clip(output, 1, clip_high, out=output)
    return output
