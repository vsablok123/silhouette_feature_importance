import numpy as np


def silhouette_feature_importance(X, labels):
    """
    The Silhouette Coefficient is calculated using the mean intra-cluster
    distance (a) and the mean nearest-cluster distance (b) for each sample.
    The Silhouette Coefficient for a sample is ``(b - a) / max(a, b)``.
    To clarrify, b is the distance between a sample and the nearest cluster
    that b is not a part of.
    The feature importance is inferred by looking at the features which contribute the most
    to the silhouette coefficient.
    Parameters
    ----------
    X : array [n_samples_a, n_features]
        Feature array.
    labels : array, shape = [n_samples]
             label values for each sample
    Returns
    -------
    silhouette : array, shape = [n_features]
        Feature importance for each feature

    """
    n = labels.shape[0]
    A = np.array([_intra_cluster_distance_slow(X, labels,i)
                  for i in range(n)])
    B = np.array([_nearest_cluster_distance_slow(X, labels, i)
                  for i in range(n)])
    print(f"A shape = {A.shape}")
    print(f"B shape = {B.shape}")
    sil_samples = abs(B - A)
    # nan values are for clusters of size 1, and should be 0
    return np.mean(np.nan_to_num(sil_samples), axis=0)

def _intra_cluster_distance_slow(X, labels, i):
    """Calculate the mean intra-cluster distance for sample i.
    Parameters
    ----------
    X : array [n_samples_a, n_features]
        Feature array.
    labels : array, shape = [n_samples]
        label values for each sample
    i : int
        Sample index being calculated. It is excluded from calculation and
        used to determine the current label
    Returns
    -------
    a : array [n_features]
        Mean intra-cluster distance for sample i
    """
    indices = np.where(labels == labels[i])[0]
    if len(indices) == 0:
        return 0.
    a = np.mean([abs(X[i] - X[j]) for j in indices if not i == j], axis=0)
    return a
    
def _nearest_cluster_distance_slow(X, labels, i):
    """Calculate the mean nearest-cluster distance for sample i.
    Parameters
    ----------
    X : array [n_samples_a, n_features]
        Feature array.
    labels : array, shape = [n_samples]
        label values for each sample
    i : int
        Sample index being calculated. It is used to determine the current
        label.
    Returns
    -------
    b : array [n_features]
        Mean nearest-cluster distance for sample i
    """
    label = labels[i]
    b = np.mean([np.mean(
                [abs(X[i] - X[j]) for j in np.where(labels == cur_label)[0]], axis=0
            ) for cur_label in set(labels) if not cur_label == label], axis=0)
    
    return b
