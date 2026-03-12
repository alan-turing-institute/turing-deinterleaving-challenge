from abc import ABC, abstractmethod
from typing import Callable, Type, Dict

import numpy as np
import sklearn


class Deinterleaver(ABC):
    """
    Abstract base class for deinterleaving models.
    This class defines the interface for deinterleaving models.
    The model takes as input a batch of pulse train sand outputs
    a batch sequences of emitter IDs

    Parameters
    ----------
    default_label: int | str | None
        If set, this label used for pulses that cannot be associated to any emitter.
        They are ignored during evaluation.
    """
    def __init__(self, default_label: int | str | float = None):
        self.default_label = default_label

    @abstractmethod
    def __call__(
        self,
        data: np.ndarray,
    ): # -> Int[Array, "batch_size seq_len"]:
        raise NotImplementedError()


class IdentityModel(Deinterleaver):
    def __init__(
            self,
            clusterer: str | Callable,
            cl_params: Dict | None = None,
            default_label: str | None = None,
    ):
        super(IdentityModel, self).__init__(default_label=default_label)
        cl_params = cl_params or {}
        self.cl_params = cl_params
        if isinstance(clusterer, str):
            clusterer = self._str_to_clusterer(clusterer)
        self.clusterer = clusterer(**self.cl_params)

    @staticmethod
    def _str_to_clusterer(cluster_str: str) -> Type:
        cluster_str = cluster_str.lower()
        match cluster_str:
            case "dbscan":
                return sklearn.cluster.DBSCAN
            case "kmeans":
                return sklearn.cluster.KMeans
            case "meanshift":
                return sklearn.cluster.MeanShift
            case "spectral":
                return sklearn.cluster.SpectralClustering
            case "agglomerative":
                return sklearn.cluster.AgglomerativeClustering
            case "hdbscan":
                return sklearn.cluster.HDBSCAN
            case "optics":
                return sklearn.cluster.OPTICS
            case "gaussian":
                return sklearn.cluster.GaussianMixture
            case _:
                raise ValueError("Unknown clustering algorithm")

    def __call__(self, data: np.ndarray) -> np.ndarray:
        cl = self.clusterer.fit_predict(data)
        return cl