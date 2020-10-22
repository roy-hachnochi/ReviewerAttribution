import sklearn_relief as relief
import numpy as np

# ======================================================================================================================
if __name__ == '__main__':
    # features_filename = "./results/reviewer_classification/articles_features.csv"
    # labels_filename = "./results/reviewer_classification/articles_labels.csv"
    # n_features = 50
    #
    # X = np.loadtxt(features_filename, delimiter=",")
    # y = np.loadtxt(labels_filename, delimiter=",")

    X = np.array([[0, 0, 0, 3],
                  [1, 1, 0, 3],
                  [0, 1, 0, 3],
                  [1, 0, 2, 3],
                  [1, 0, 0, 3]])
    y = np.array([0, 0, 1, 1, 1])
    n_features = 2

    r = relief.ReliefF(n_features=n_features)
    X_transformed = r.fit_transform(X, y)
    print()


