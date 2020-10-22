from skrebate import ReliefF
import numpy as np

# ======================================================================================================================
if __name__ == '__main__':
    features_filename = "./results/reviewer_classification/articles_features.csv"
    labels_filename = "./results/reviewer_classification/articles_labels.csv"
    n_features = 10

    X = np.loadtxt(features_filename, delimiter=",")
    y = np.loadtxt(labels_filename, delimiter=",")

    # X = np.array([[-1,2,3,4,5,3,6,2,7,3,6],
    #               [-1,5,2,6,-1,3,5,2,5,-8,3],
    #               [-1,2,-2,43,1,4,4,4,-3,-1,-1],
    #               [-10,2,2,2,2,0,0,0,-3,2,2],
    #               [1,2,0,-1,-1,-1,-1,-2,-3,-4,-5],
    #               [1,2,1,2,3,4,5,6,-3,8,9],
    #               [1,0,0,0,0,0,0,0,0,0,0]])
    # y = np.array([1,1,1,1,0,0,0])

    r = ReliefF(n_features_to_select=n_features, n_neighbors=10)
    r.fit(X, y)
    print(r.feature_importances_)
    print(r.top_features_)


