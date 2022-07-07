import glob

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# embedding_files = sorted(
#     glob.glob("/home/mbed/Projects/haptic-unsupervised/experiments/supervised/put/*/00000/default/tensors.tsv"))
# label_files = sorted(
#     glob.glob("/home/mbed/Projects/haptic-unsupervised/experiments/supervised/put/*/00000/default/metadata.tsv"))
# NUM_CLASSES = 8
# x = np.array([np.loadtxt(f) for f in embedding_files])
# y = np.array([[int(s.replace('tensor(', '').replace('.)', '')) for s in np.loadtxt(f, str)] for f in label_files])
# classes = ["Art. grass", "Rubber", "Carpet", "PVC", "Ceramic", "Foam", "Sand", "Rock"]
# physical_params = np.array([1.52, 1.36, 1.99, 0.81, 0.58, 1.59, 0.53, 0.69])

embedding_files = sorted(
    glob.glob("/home/mbed/Projects/haptic-unsupervised/experiments/supervised/touching/*/00000/default/tensors.tsv"))
label_files = sorted(
    glob.glob("/home/mbed/Projects/haptic-unsupervised/experiments/supervised/touching/*/00000/default/metadata.tsv"))
x = np.array([np.loadtxt(f) for f in embedding_files])
y = np.array(
    [np.array([int(s.replace('tensor(', '').replace('.)', '')) for s in np.loadtxt(f, str)]) for f in label_files])
NUM_CLASSES = 11
classes = ["Bubble", "Cardboard1", "Cardboard2", "Gum", "Leather", "Natural bag", "Plastic", "Sheet", "Sponge",
           "Styrofoam", "Synth. Bag"]
physical_params = np.array([1.0, 0.4331, 0.4331, 0.001, 0.1, 8.27, 0.621, 1.0, 1.0, 2.28, 0.0003])  # put


def pick_classes(x, y):
    x_by_classes, y_by_classes = list(), list()
    for class_idx in range(NUM_CLASSES):
        idx = np.argwhere(y == class_idx)
        x_by_classes.append(np.squeeze(x[idx]))
        y_by_classes.append(np.squeeze(y[idx]))
    return np.array(x_by_classes), np.array(y_by_classes)

distances = list()
for test_class_idx, (x_run, y_run) in enumerate(zip(x, y)):
    x_by_classes, y_by_classes = pick_classes(x_run, y_run)

    centroids = list()
    for cluster in x_by_classes:
        centroids.append(np.mean(np.asarray(cluster), 0))

    weights = list()
    for k in range(len(centroids)):
        weights.append(np.linalg.norm(np.asarray(centroids[k]) - np.asarray(centroids[test_class_idx])))

    w = np.array([weights[i] for i in range(len(weights)) if i != test_class_idx])
    w = np.max(w) - w
    w = w / np.max(w)
    pp = np.array([physical_params[i] for i in range(len(physical_params)) if i != test_class_idx])
    mean_phys = np.average(pp, weights=w)
    print(mean_phys)

    weights = np.asarray(weights)[np.newaxis]
    distances.append(weights)

dist_mat = np.concatenate(distances, 0)
ax = sns.heatmap(dist_mat, square=True, annot=True, cmap='Blues', fmt='.2f', linewidths=0.0, linecolor='black')
ax.set_yticklabels(classes, rotation=20, horizontalalignment='right', fontsize=5)
ax.set_xticklabels(classes, rotation=20, horizontalalignment='center', fontsize=5)
plt.show()
