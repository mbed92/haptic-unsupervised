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
    # K-Means description: https://scikit-learn.org/stable/modules/clustering.html#k-means
    two_means = cluster.MiniBatchKMeans(n_clusters=params["n_clusters"],
                                        init=params["kmeans"]["init"])

    # Ward description: https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering
    ward = cluster.AgglomerativeClustering(
        linkage="ward",
        n_clusters=params["n_clusters"],
        connectivity=connectivity,
        affinity="euclidean"  # If linkage is “ward”, only “euclidean” is accepted.
    )

    # Spectral description: https://scikit-learn.org/stable/modules/clustering.html#spectral-clustering
    spectral = cluster.SpectralClustering(
        n_clusters=params["n_clusters"],
        eigen_solver=params["spectral"]["eigen_solver"],
        affinity=params["spectral"]["affinity"]
    )

    # Agglomerative description: https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering
    average_linkage = cluster.AgglomerativeClustering(
        linkage="average",
        affinity=params["average_linkage"]["affinity"],
        n_clusters=params["n_clusters"],
        connectivity=connectivity,
    )

    # Birch description: https://scikit-learn.org/stable/modules/clustering.html#birch
    birch = cluster.Birch(n_clusters=params["n_clusters"],
                          threshold=params["birch"]["threshold"],
                          branching_factor=params["birch"]["branching_factor"])

    # GaussianMixture description: https://scikit-learn.org/stable/modules/mixture.html#mixture
    gmm = mixture.GaussianMixture(n_components=params["n_clusters"],
                                  covariance_type=params["gmm"]["covariance_type"],
                                  n_init=params["gmm"]["n_init"],
                                  init_params=params["gmm"]["init_params"])

    # save SCIKIT_LEARN_PARAMS to txt file
    with open('SCIKIT_LEARN_PARAMS.txt', 'w') as f:
        print(SCIKIT_LEARN_PARAMS, file=f)

    return (
        ("KMeans", two_means),
        ("Spectral", spectral),
        ("Ward", ward),
        ("Agglomerative", average_linkage),
        ("BIRCH", birch),
        ("GaussianMixture", gmm),
    )


RANDOM_SEED = 0

SCIKIT_LEARN_PARAMS = {
    "kmeans": {
        "init": "k-means++"
    },
    "spectral": {
        "affinity": "nearest_neighbors",  # How to construct the affinity matrix. Choices: 'nearest_neighbors', 'rbf'.
        "eigen_solver": "arpack",  # Eigenvalue decomposition strategy for Spectral embedding.,
        "gamma": 1.0,  # Kernel coeff for rbf, poly, sigmoid, laplacian and chi2 kernels. Ignored by other kernels.
        "degree": 3,  # Degree of the polynomial kernel. Ignored by other kernels.
        "coef0": 1,  # Zero coefficient for polynomial and sigmoid kernels. Ignored by other kernels.
    },
    "average_linkage": {
        "affinity": "manhattan",  # Metric used to compute the linkage. Can be “euclidean”, “l1”, “l2”, “manhattan”,
    },
    "birch": {
        "threshold": 0.5,  # The radius of the subcluster obtained by merging a new sample and the closest subcluster.
        "branching_factor": 50,  # Maximum number of CF subclusters in each node.
    },
    "gmm": {
        "covariance_type": "full",  # Choices are 'full', 'tied', 'diag', 'spherical'.
        "n_init": 10,  # Number of initializations to perform.
        "init_params": "kmeans",  # The method used to initialize the weights. Choices are 'kmeans', 'random'.
    },
    "n_neighbors": 3,  # Number of neighbors for manifold learning methods.
    "n_rows": 2,  # # Number of rows in visualizations.
    "tsne_n_components": 2,  # Number of dimensions for t-SNE visualization.
    "svd_components": 20,  # Number of components for truncated SVD.
    "title_size": 18,  # Font size for plot titles.
    "figsize": (15, 15),  # Figure size for visualizations.
}
