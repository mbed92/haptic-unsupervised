from sklearn import cluster, mixture, metrics

from utils.metrics import purity_score


def get_sklearn_clustering_metrics_x_labels():
    """ https://towardsdatascience.com/performance-metrics-in-machine-learning-part-3-clustering-d69550662dc6 """
    return (
        ("silhouette_score", metrics.silhouette_score),
        ("calinski_harabasz_score", metrics.calinski_harabasz_score),
        ("davies_bouldin_score", metrics.davies_bouldin_score)
    )


def get_sklearn_clustering_metrics_true_pred():
    """ https://towardsdatascience.com/performance-metrics-in-machine-learning-part-3-clustering-d69550662dc6 """
    return (
        ("rand_score", metrics.rand_score),
        ("adjusted_rand_score", metrics.adjusted_rand_score),
        ("mutual_info_score", metrics.normalized_mutual_info_score),
        ("purity_score", purity_score)
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
        ("MiniBatch\nKMeans", two_means),
        ("Spectral\nClustering", spectral),
        ("Ward", ward),
        ("Agglomerative\nClustering", average_linkage),
        ("BIRCH", birch),
        ("Gaussian\nMixture", gmm),
    )
