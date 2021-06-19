import numpy as np


class BootstrepedDistribution:
    def __init__(self, np_distribution_class, bootstrep_size, sample_size, *args, **kwargs):
        self._sps_distribution_class = np_distribution_class

        try:
            self._sps_distribution = np_distribution_class(*args)
        except Exception:
            self._sps_distribution = None

        self._distrib_params = args
        self._bootstrep_size = bootstrep_size
        self._sample_size = sample_size

        self._sample = self._sps_distribution.rvs(size=self._sample_size, **kwargs)

    def get_param_bootstrep(self, n, estimator, estimator_func):  #
        bootstrap_estimators = np.zeros(self._bootstrep_size)
        bootstrap_param_samples = self._sps_distribution_class(self._distrib_params[0], estimator) \
            .rvs((self._bootstrep_size, n))

        for k in range(self._bootstrep_size):
            bootstrap_estimators[k] = estimator_func(bootstrap_param_samples[k]) / m

        return bootstrap_estimators

    def get_bootstrep_variance(self, bootstrep_func, bootstrep_estimate_func, estimator_func):

        bootstrap_variance_estimators = np.zeros(self._sample_size)
        for n in range(self._sample_size):
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