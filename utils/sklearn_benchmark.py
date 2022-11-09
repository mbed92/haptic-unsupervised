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
        n_clusters=params["n_clusters"],
        connectivity=connectivity)
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
    gmm = mixture.GaussianMixture(n_components=params["n_clusters"])

    return (
        ("KMeans", two_means),
        ("SpectralClustering", spectral),
        ("Ward", ward),
        ("AgglomerativeClustering", average_linkage),
        ("BIRCH", birch),
        ("GaussianMixture", gmm),
    )


RANDOM_SEED = 0

SCIKIT_LEARN_PARAMS = {
    "quantile": 0.3,
    "eps": 0.3,
    "damping": 0.9,
    "preference": -200,
    "n_neighbors": 3,
    "n_latent_size": 10,
    "min_samples": 7,
    "xi": 0.05,
    "min_cluster_size": 0.1,
    "n_rows": 2,
    "tsne_n_components": 2,
    "svd_components": 20,
    "title_size": 18,
    "figsize": (15, 15)
}
