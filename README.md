# Haptic recognition using unsupervised learning

This repository contains a code related to the research on Unsupervised Haptic Recognition conducted towards
the Ph.D. thesis of the author of this repository. The target is to learn to differentiate
between terrains/fabrics touched by the robot used in the experiments without any supervision.

## Method

The unsupervised learning algorithm used in the following repository is the
Deep Embedding Clustering [DEC 2016](https://dl.acm.org/doi/10.5555/3045390.3045442).

## Steps to reproduce:

In the [experiments](experiments) folder there exist several scripts related to the training procedure:

* Do ```git submodule update --init``` as we use some functionalities from [HAPTR](https://github.com/kolaszko/haptic_transformer) repo.
* [tune_autoencoder](experiments/tune.py) - a script that tunes some hyperparameters using
  Tune lib. It speeds up a little bit further research.
* [train_stacked_autoencoder](experiments/train_stacked_autoencoder.py) - training procedure of the Stacked
  AutoEncoder (SAE) of time series.
* [train_deep_embedding_clustering](experiments/train_dec.py) - a fine-tuning of the SAC in a way
  where the distances between clusters are maximized.

## Datasets:

* [Touching - Robot Soft 2019](https://drive.google.com/open?id=1NhUFJys-3D6-3BT6slBOmPYQa8bqx4cY) - force readings from
  the OptoForce sensor mounted on the tooltip of the robotic arm.

## Tune results:
* For PUT:
```
Best trial config: {'kernel': 7, 'activation': ReLU(), 'dropout': 0.1731244609139919, 'lr': 0.001993122444870846, 'weight_decay': 0.0003703603065553894}
```

* For TOUCHING:
```
Best trial config: {'kernel': 11, 'activation': GeLU(), 'dropout': 0.1300238908180922, 'lr': 0.0006863541995399743, 'weight_decay': 0.00018546443538449212}
```
