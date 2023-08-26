import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style and context
sns.set(style="white")
sns.set_context("talk")

# Sample data (replace with your actual data)
algorithms = ['DEC', 'DEC (raw)', 'Agglomerative', 'BIRCH', 'GaussianMixture', 'KMeans', 'Spectral', 'Ward']
metrics = ['MutualInfo', 'Purity', 'ClusteringAccuracy']

# BioTac2 results
means = np.array([
    [0.921, 0.846, 0.781],
    [0.915, 0.824, 0.770],
    [0.099, 0.062, 0.056],
    [0.315, 0.163, 0.150],
    [0.257, 0.144, 0.130],
    [0.319, 0.172, 0.153],
    [0.323, 0.187, 0.156],
    [0.333, 0.180, 0.161]
])
std_devs = np.array([
    [0.002, 0.005, 0.003],
    [0.003, 0.011, 0.003],
    [0.025, 0.013, 0.012],
    [0.004, 0.005, 0.007],
    [0.086, 0.032, 0.027],
    [0.002, 0.006, 0.004],
    [0.001, 0.000, 0.001],
    [0.003, 0.003, 0.001]
])

# # Touching results
# means = np.array([
#     [0.701, 0.659, 0.596],
#     [0.685, 0.652, 0.572],
#     [0.137, 0.215, 0.212],
#     [0.335, 0.316, 0.292],
#     [0.340, 0.314, 0.278],
#     [0.314, 0.289, 0.265],
#     [0.291, 0.347, 0.306],
#     [0.313, 0.280, 0.259]
# ])
# std_devs = np.array([
#     [0.004, 0.006, 0.001],
#     [0.008, 0.002, 0.002],
#     [0.000, 0.000, 0.000],
#     [0.011, 0.013, 0.013],
#     [0.017, 0.023, 0.013],
#     [0.021, 0.012, 0.013],
#     [0.014, 0.005, 0.004],
#     [0.000, 0.000, 0.000]
# ])

x = np.arange(len(algorithms))
bar_width = 0.2

plt.figure(figsize=(20, 5))  # Adjust the figsize

for i, metric in enumerate(metrics):
    plt.bar(x + i * bar_width, means[:, i], yerr=std_devs[:, i], label=metric, width=bar_width)

plt.xlabel('Algorithms')
plt.ylabel('Scores')
plt.title('Clustering Algorithm Performance (BioTac2)')
plt.xticks(x + bar_width * (len(metrics) - 1) / 2, algorithms)
plt.yticks(np.arange(0, 1.1, 0.1))  # Set y-axis grid lines at intervals of 0.1
plt.legend()
plt.tight_layout()  # Apply tight layout
plt.grid(True)
plt.show()
