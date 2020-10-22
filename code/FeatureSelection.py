import sklearn_relief as relief
import numpy as np

# ======================================================================================================================
if __name__ == '__main__':
    features_filename = "./results/reviewer_classification/articles_features.csv"
    labels_filename = ""  # TODO
    n_features = 50

    X = np.loadtxt(features_filename, delimiter=",")
    y = np.loadtxt(labels_filename, delimiter=",")

    r = relief.ReliefF(n_features=n_features)
    X_transformed = r.fit_transform(X, y)



