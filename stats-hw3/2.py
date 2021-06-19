# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
import pandas as pd
from scipy.special import factorial
import warnings



randvar_count = 200
m = 50
p = sps.uniform(0, 1).rvs(1)[0]
BOOTSTREP_SIZE = 200


class BootstrepedDistribution:
    def __init__(self, np_distribution_class, bootstrep_size, *args, **kwargs):
        self._np_distribution_class = np_distribution_class
        self._np_distribution = np_distribution_class(*args)
        self._distrib_params = args
        self._bootstrep_size = bootstrep_size

        self._sample = self._np_distribution.rvs(size=randvar_count, **kwargs)

    def get_param_bootstrep(self, n, estimator_args, estimator_func):  #
        bootstrap_estimators = np.zeros(self._bootstrep_size)
        bootstrap_param_samples = self._np_distribution_class(*estimator_args) \
            .rvs((self._bootstrep_size, n))

        for k in range(self._bootstrep_size):
            bootstrap_estimators[k] = estimator_func(bootstrap_param_samples[k]) / m

        return bootstrap_estimators

    def get_bootstrep_variance(self, bootstrep_func, bootstrep_estimate_func, estimator_func):
        bootstrap_variance_estimators = np.zeros(randvar_count)
        for n in range(randvar_count):
            effective_estimator = estimator_func(self._sample[:n + 1])
            bootstrap_variance_estimators[n] = self._s2(
                bootstrep_func(self, n + 1, effective_estimator, bootstrep_estimate_func))

        return bootstrap_variance_estimators

    def _s2(self, bootstrap_estimators):
        return np.mean(bootstrap_estimators ** 2) - np.mean(bootstrap_estimators) ** 2

    def get_estimators(self, sample, estimators_func):
        params = [estimators_func(sample[:n]) for n in range(1, sample.shape[0])]
        params.append(estimators_func(sample))

        return params

# %%

def gen_distributions() -> dict:
    result = {}
    result['$Bin(m, p)$'] = BootstrepedDistribution(sps.binom(m, p), lambda array: np.mean(array / m))


def make_binom_experiment():
    d = BootstrepedDistribution(sps.binom, 500, m, p)
    meanm = lambda array: np.mean(array) / m
    X = d.get_bootstrep_variance(BootstrepedDistribution.get_param_bootstrep, meanm, meanm)

    X1 = d.get_bootstrep_variance(BootstrepedDistribution.get_param_bootstrep, lambda array: array[0],
                                  lambda array: np.min(array) / m)

    start = 0
    plt.figure(figsize=(12, 6))
    plt.plot(range(start + 1, randvar_count + 1), X[start:],
             label=r"$\overline{X}$", linewidth=4)
    plt.plot(range(start + 1, randvar_count + 1), X1[start:], label=r"$X_1$")
    plt.plot(range(start + 1, randvar_count + 1), [p * (1 - p) / n / m for n in range(start + 1, randvar_count + 1)],
             label=r"$\frac{1}{I_X(p)}$", color='y', linewidth=3)
    plt.title(r"График зависимости дисперсии оценки параметра $p$ от $n$", fontsize=20)
    plt.xlabel(r"$n$", fontsize=20)
    plt.ylabel(r"$s^2(\hat{p})$", fontsize=20)
    plt.legend(loc='right', fontsize=20)
    plt.show()

def make_norm_experiment():
    sigma = 2.1
    d = BootstrepedDistribution(sps.norm, BOOTSTREP_SIZE, p, sigma)
    Xmean = d.get_bootstrep_variance(BootstrepedDistribution.get_param_bootstrep, np.mean, np.mean)

    Xmedian = d.get_bootstrep_variance(BootstrepedDistribution.get_param_bootstrep, np)

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, randvar_count + 1), Xmean,
             label=r"$\overline{X}$")
    plt.plot(range(1, randvar_count + 1), Xmedian,
             label=r"$\hat{\mu}$")
    plt.plot(range(1, randvar_count + 1), [sigma / n for n in range(1, randvar_count + 1)],
             label=r"$\frac{1}{I_X(a)}$", linewidth=4, color='y')
    plt.title(r"График зависимости дисперсии оценок параметра $a$ от $n$", fontsize=20)
    plt.xlabel(r"$n$", fontsize=20)
    plt.ylabel(r"$s^2(\hat{a})$", fontsize=20)
    plt.ylim(0, 0.05)
    plt.legend(loc='best', fontsize=20);


make_norm_experiment()