from skrebate import ReliefF
import numpy as np
import matplotlib.pyplot as plt

# ======================================================================================================================
if __name__ == '__main__':
    features_filename = "./results/reviewer_classification/toy_features.csv"
    labels_filename = "./results/reviewer_classification/toy_labels.csv"
    n_features = 10
    n_neighbors = 10
    nFeatures = [50, 70, 100, 30, 15, 7]  # number of features for each feature type
    feature_names = ['unigram hist', 'bigram hist', 'trigram hist', 'fourgram hist', 'fivegram hist', 'LM perplexity']

    X = np.loadtxt(features_filename, delimiter=",")
    labels = np.loadtxt(labels_filename, delimiter=",", dtype='str')
    class_to_labels_dict = list(set(labels))
    labels_to_class_dict = {label: i for i, label in enumerate(class_to_labels_dict)}
    y = np.array([labels_to_class_dict[label] for label in labels])

    # X = np.array([[-1,2,3,4,5,3,6,2,7,3,6],
    #               [-1,5,2,6,-1,3,5,2,5,-8,3],
    #               [-1,2,-2,43,1,4,4,4,-3,-1,-1],
    #               [-10,2,2,2,2,0,0,0,-3,2,2],
    #               [1,2,0,-1,-1,-1,-1,-2,-3,-4,-5],
    #               [1,2,1,2,3,4,5,6,-3,8,9],
    #               [1,0,0,0,0,0,0,0,0,0,0]])
    # y = np.array([1,1,1,1,0,0,0])

    r = ReliefF(n_features_to_select=n_features, n_neighbors=n_neighbors)
    r.fit(X, y)

    plt.figure()
    ind = 0
    for i in range(len(nFeatures)):
        plt.bar(range(ind, ind + nFeatures[i]), r.feature_importances_[ind:(ind + nFeatures[i])], label=feature_names[i])
        ind = ind + nFeatures[i]  # TODO: colors
    plt.grid()
    plt.title('Feature Importance')
    plt.xlabel('Feature')
    plt.ylabel('Feature importance weight')
    plt.legend(loc='best', shadow=True, fancybox=True, framealpha=0.7)
    plt.show()
    print()


