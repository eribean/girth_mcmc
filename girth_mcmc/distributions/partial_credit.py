import numpy as np
import theano.tensor as tt

from theano.tensor.nnet import softmax
from theano.tensor.extra_ops import cumsum
from pymc3.theanof import floatX

from pymc3.distributions.discrete import Categorical


__all__ = ['PartialCredit']


class PartialCredit(Categorical):
    """Computed the probability for the partial credit model given a set of
    cutpoints and observations.
    """

    def __init__(self, eta, cutpoints, *args, **kwargs):
        eta = tt.as_tensor_variable(floatX(eta))
        cutpoints = tt.concatenate(
            [
                tt.as_tensor_variable([0.0]),
                tt.as_tensor_variable(cutpoints)
            ])
        cutpoints = tt.shape_padaxis(cutpoints, 0)
        eta = tt.shape_padaxis(eta, 1)

        p = softmax(cumsum(eta - cutpoints, axis=1))

        super().__init__(p=p, *args, **kwargs)
