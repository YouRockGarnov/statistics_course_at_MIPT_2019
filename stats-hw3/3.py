# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
import pandas as pd
from scipy.special import factorial
import warnings
from BootstrepedDistribution import BootstrepedDistribution

class BootstrepedDistribution:
    def __init__(self, np_distribution_class, bootstrep_size, sample_size, thetas, *args, **kwargs):
        self._sps_distribution_class = np_distribution_class

        try:
            self._sps_distribution = np_distribution_class(*args)
        except Exception:
            self._sps_distribution = None

        self._thetas = thetas
        self._distrib_params = args
        self._bootstrep_size = bootstrep_size
        self._sample_size = sample_size

    def get_param_bootstrep(self, n, estimator, estimator_func):  #
        bootstrap_estimators = np.zeros(self._bootstrep_size)
        bootstrap_param_samples = self._sps_distribution_class(estimator) \
            .rvs((self._bootstrep_size, n))

        for k in range(self._bootstrep_size):
            bootstrap_estimators[k] = estimator_func(bootstrap_param_samples[k])

        return bootstrap_estimators

    def get_bootstrep_variance(self, bootstrep_func, bootstrep_estimate_func, estimator_func):
        bootstrap_variance_estimators = np.zeros(len(self._thetas))
        for i, theta in enumerate(self._thetas):
            sample = self._sps_distribution_class(theta).rvs(size=self._sample_size)
            effective_estimator = estimator_func(sample)
            bootstrap_variance_estimators[i] = self._s2(
                bootstrep_func(self, self._sample_size, effective_estimator, bootstrep_estimate_func))

        return bootstrap_variance_estimators

    def _s2(self, bootstrap_estimators):
        return np.mean(bootstrap_estimators ** 2) - np.mean(bootstrap_estimators) ** 2

    def get_estimators(self, sample, estimators_func):
        params = [estimators_func(sample[:n]) for n in range(1, sample.shape[0])]
        params.append(estimators_func(sample))

        return params

RANDVAR_COUNT = 200
m = 50
p = sps.uniform(0, 1).rvs(1)[0]
BOOTSTREP_SIZE = 200

# %%

def main():
    # distrib = BootstrepedDistribution(sps.bernoulli, BOOTSTREP_SIZE, RANDVAR_COUNT)
    thetas = np.linspace(0, 1, 101)
    rao_kramers_estimation = thetas * (1 - thetas) / RANDVAR_COUNT

    plt.plot(thetas, rao_kramers_estimation)
    plt.show()

    distrib = BootstrepedDistribution(sps.bernoulli, BOOTSTREP_SIZE, RANDVAR_COUNT, thetas)
    variance_estimation = distrib.get_bootstrep_variance(BootstrepedDistribution.get_param_bootstrep, np.mean, np.mean)

    plt.plot(thetas, variance_estimation)
    plt.show()


if __name__ == '__main__':
    main()