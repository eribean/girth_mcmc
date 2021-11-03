import numpy as np

import pymc3 as pm
import theano.tensor as tt
from pymc3.distributions.distribution import draw_values, generate_samples
from pymc3.distributions.dist_math import bound


__all__ = ['Rayleigh']


class Rayleigh(pm.distributions.Weibull):
    """Register a rayleigh distribution in pymc."""
    
    def __init__(self, beta, offset=0, *args, **kwargs):
        """Constructor class for Rayleigh distribution.
        
        Args:
            beta: scale parameter that controls the shape of the distribution
            offset: begining of the rayleigh distribution
        """
        super().__init__(alpha=2, beta=pm.math.sqrt(2)*beta, *args, **kwargs)
        self.offset = offset
        
    def random(self, point=None, size=None):
        """
        Draw random values from Weibull distribution.
        Parameters
        ----------
        point: dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size: int, optional
            Desired size of random sample (returns one sample if not
            specified).
        Returns
        -------
        array
        """
        alpha, beta = draw_values([self.alpha, self.beta], point=point, size=size)

        def _random(a, b, size=None):
            return b * (-np.log(np.random.uniform(size=size))) ** (1 / a) + self.offset

        return generate_samples(_random, alpha, beta, dist_shape=self.shape, size=size)

    def logp(self, value):
        """
        Calculate log-probability of Weibull distribution at specified value.
        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or theano tensor
        Returns
        -------
        TensorVariable
        """
        alpha = self.alpha
        beta = self.beta
        value_ = pm.math.maximum(value - self.offset, 1e-313)
        return bound(
            tt.log(alpha)
            - tt.log(beta)
            + (alpha - 1) * tt.log(value_ / beta)
            - (value_ / beta) ** alpha,
            value_ >= 0,
            alpha > 0,
            beta > 0,
        )    