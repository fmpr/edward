#!/usr/bin/env python
"""
Bayesian linear regression using mean-field variational inference.

Probability model:
    Bayesian linear model
    Prior: Normal
    Likelihood: Normal
Variational model
    Likelihood: Mean-field Normal
"""
from __future__ import print_function

import edward as ed
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import tensorflow as tf

from edward.models import Variational, Normal, InvGamma
from edward.stats import norm, invgamma
from edward.util import get_dims


class ARModel:
    """
    Autoregressive model for time-series y.

    p(y, z) = [\prod_{t=1}^T Normal(y_t | x_t, \sigma^2) 
                Normal(x_t | \mu + \beta_1 * x_{t-1} + \beta_2 * x_{t-2} + ..., proc_noise)]
                Normal(\mu | 0, d)
                [\prod_{k=1}^K Normal(\beta_k | 0, c)],

    where z = {x, \mu, \beta, \sigma} and K is the level of the autoregressive process AR(K).

    Parameters
    ----------
    proc_noise : float, optional
        Variance of the latent AR process.
    prior_variance : float, optional
        Variance of the normal prior on weights; aka L2
        regularization parameter, ridge penalty, scale parameter.
    """
    def __init__(self, T, K):
        self.T = T
        self.K = K
        self.proc_noise = 1
        self.obs_noise = 1
        self.a = 1
        self.b = 1
        self.c = 10
        self.d = 10

    def log_prob(self, xs, zs):
        """Return a vector [log p(xs, zs[1,:]), ..., log p(xs, zs[S,:])]."""

        # extract observed variables
        y = xs['y']

        # extract latent variables
        x, coeffs = zs
        mu = coeffs[:, 0]           # (n_samples x 1)
        beta = coeffs[:, 1:]        # (n_samples x K)
        n_samples = get_dims(x)[0]

        # priors
        log_prior = norm.logpdf(mu, 0, self.d)
        log_prior += tf.reduce_sum(norm.logpdf(beta, 0, self.c), 1)

        for t in range(self.K):
            log_prior += norm.logpdf(x[:,t], mu, 100)       # fat prior on first observations
        for t in range(self.K, self.T):
            mean = mu + tf.reduce_sum(tf.mul(beta, x[:,t-self.K:t]), 1)
            log_prior += norm.logpdf(x[:,t], mean, self.proc_noise)

        # likelihood
        y = tf.tile(tf.expand_dims(y, 1), [1, n_samples])                   # repeat y to be (T x n_samples)
        log_lik = tf.reduce_sum(norm.logpdf(tf.transpose(y), x, self.obs_noise), 1)
        
        return log_prior + log_lik

    def predict(self, xs, zs):
        """Return a prediction for each data point, averaging over
        each set of latent variables z in zs."""

        x, coeffs = zs
        x_hat = tf.reduce_mean(x, 0)

        return x_hat

def build_toy_dataset(T, noise_std=0.5):
    source = 2*np.sin(np.linspace(0, 6, T))
    y = source + np.random.normal(0, noise_std, T)
    plt.plot(source, 'k-')
    plt.plot(y, 'r-')
    plt.show()
    return {'y': y}, source


# fix seed for reproducibility
ed.set_seed(42)

# generate artificial dataset
T = 40
K = 2
data, source = build_toy_dataset(T)

# build model
model = ARModel(T, K)

# specify variational distribution
variational = Variational()
variational.add(Normal(model.T))    # latent states
variational.add(Normal(1 + K))      # mean and AR coefficients

# run mean-field variational inference
inference = ed.MFVI(model, variational, data)
inference.run(n_iter=1000, n_samples=50, n_print=100)

# evaluate model
print("MAE:", ed.evaluate('mae', model, variational, data, source))
print("RMSE:", np.sqrt(ed.evaluate('mse', model, variational, data, source)))

# plot smoothed latent state estimates
x_hat = variational.layers[0].loc.eval()
plt.plot(source, 'k-')
plt.plot(data['y'], 'r-')
plt.plot(x_hat, 'b-')
plt.xlabel("time")
plt.ylabel("y value")
plt.title("smoothed time series")
plt.legend(["true source", "noisy observations", "smoothed observations"])
plt.show()