import argparse
import glob
import os

import pandas as pd

TOUCHING_CLASSES = ["Cardboard1", "Cardboard2", "Gum", "Leather", "Natural bag", "Plastic", "Sheet", "Sponge",
                    "Styrofoam"]
PUT_CLASSES = ["Art. Grass", "Rubber", "Carpet", "PVC", "Ceramic", "Foam", "Sand", "Rock"]


def df_cleaned_up(df: pd.DataFrame):
    df = df.astype(str)
    df = df.applymap(lambda x: x.replace("tensor(", ""))
    df = df.applymap(lambda x: x.replace(")", ""))
    df = df.applymap(lambda x: float(x))
    return df


def num_classes_and_clusters(df: pd.DataFrame):
    return df.iloc[:, 0].nunique(), df.iloc[:, 1].nunique()


def variety(df: pd.DataFrame):
    m = list()

    clusters = df.iloc[:, 1].unique()
    if len(clusters) > 0:
        for cluster in clusters:
            class_indexes = df.index[df.iloc[:, 1] == cluster].tolist()
            unique_classes = df.iloc[class_indexes, 0].value_counts()
            m.append(unique_classes)

    return m


def main(args):
    for run_name in args.results_runs:
        experiment_path = os.path.join(args.results_folder, run_name, args.results_experiments)
        true_classes_paths = sorted(glob.glob("{0}/**/*0/default/metadata*".format(experiment_path), recursive=True))
        clusters_paths = sorted(glob.glob("{0}/**/*1/default/metadata*".format(experiment_path), recursive=True))
        assert len(true_classes_paths) == len(clusters_paths)

        # compare true classes with corresponding cluster assignments
        for true_c_path, clusters_c_path in zip(true_classes_paths, clusters_paths):
            df_true = df_cleaned_up(pd.read_csv(true_c_path, header=None))
            df_pred = df_cleaned_up(pd.read_csv(clusters_c_path, header=None))
            assert df_true.size == df_pred.size

            # assert num classes in true set
            classes_in_ds = TOUCHING_CLASSES if "touching" in true_c_path else PUT_CLASSES
            if not len(classes_in_ds) == df_true.iloc[:, 0].nunique():
                continue

            # keep all data in one df
            df_total = pd.concat([df_true, df_pred], axis=1)
            num_cls, num_clu = num_classes_and_clusters(df_total)
            assignments = variety(df_total)

            print(true_c_path, "num_cls", num_cls, "num_clusters", num_clu)
            for i, assignment in enumerate(assignments):
                print(f"Cluster {i}/{num_clu} - CLASS / NUMBER OF OCCURRENCES:\n", assignment)
            print("=================\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-folder', type=str, default="/home/mbed/Projects/haptic-unsupervised/results")
    parser.add_argument('--results-runs', nargs='+', required=True)
    parser.add_argument('--results-experiments', type=str, required=True)
    args, _ = parser.parse_known_args()
    main(args)
