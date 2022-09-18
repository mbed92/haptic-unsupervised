from sklearn import cluster, mixture, metrics

from utils.metrics import purity_score, clustering_accuracy


def get_sklearn_clustering_metrics_x_labels():
    """ https://towardsdatascience.com/performance-metrics-in-machine-learning-part-3-clustering-d69550662dc6 """
    return (
        ("Silhouette", metrics.silhouette_score),
    )


def get_sklearn_clustering_metrics_true_pred():
    """ https://towardsdatascience.com/performance-metrics-in-machine-learning-part-3-clustering-d69550662dc6 """
    return (
        ("AdjustedRand", metrics.adjusted_rand_score),
        ("MutualInfo", metrics.normalized_mutual_info_score),
        ("Purity", purity_score),
        ("ClusteringAccuracy", clustering_accuracy)
    )


def get_sklearn_clustering_algorithms(params, connectivity):
    two_means = cluster.MiniBatchKMeans(n_clusters=params["n_clusters"])
    ward = cluster.AgglomerativeClustering(
        n_clusters=params["n_clusters"], linkage="ward", connectivity=connectivity
    )
    spectral = cluster.SpectralClustering(
        n_clusters=params["n_clusters"],
        eigen_solver="arpack",
        affinity="nearest_neighbors",
    )
    average_linkage = cluster.AgglomerativeClustering(
        linkage="average",
        affinity="cityblock",
        n_clusters=params["n_clusters"],
        connectivity=connectivity,
    )
    birch = cluster.Birch(n_clusters=params["n_clusters"])
    gmm = mixture.GaussianMixture(
        n_components=params["n_clusters"], covariance_type="full"
    )

    return (
        ("KMeans", two_means),
        ("SpectralClustering", spectral),
        ("Ward", ward),
        ("AgglomerativeClustering", average_linkage),
        ("BIRCH", birch),
        ("GaussianMixture", gmm),
    )
