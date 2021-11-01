import numpy as np


__all__ = ['get_discrimination_indices']


def get_discrimination_indices(n_items, n_factors):
    """Local function to get parameters for discrimination estimation."""

    lower_indices = np.tril_indices(n_items, k=-1, m=n_factors)
    diagonal_indices = np.diag_indices(n_factors)
    lower_length = lower_indices[0].shape[0]

    # Set constraints to be the final items
    lower_indices = (n_items - 1 - lower_indices[0], lower_indices[1])
    diagonal_indices = (n_items - 1 - diagonal_indices[0], diagonal_indices[1])

    return diagonal_indices, lower_indices