from skrebate import ReliefF
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt

# ======================================================================================================================
if __name__ == '__main__':
    features_filename = "./results/main/reviews_features.csv"
    labels_filename = "./results/main/reviews_labels.csv"
    nFeatures_factor = 0.3
    n_neighbors = 10
    nFeatures = [50, 70, 100, 30, 15, 7]  # number of features for each feature type
    feature_names = ['unigram hist', 'bigram hist', 'trigram hist', 'fourgram hist', 'fivegram hist', 'LM perplexity']

    X = np.loadtxt(features_filename, delimiter=",")
    X = preprocessing.StandardScaler().fit_transform(X)
    labels = np.loadtxt(labels_filename, delimiter=",", dtype='str')
    class_to_labels_dict = list(set(labels))
    labels_to_class_dict = {label: i for i, label in enumerate(class_to_labels_dict)}
    y = np.array([labels_to_class_dict[label] for label in labels])

    n_features = int(X.shape[1] * nFeatures_factor)
    r = ReliefF(n_features_to_select=n_features, n_neighbors=n_neighbors)
    r.fit(X, y)

    feature_ranks = np.zeros(len(r.top_features_))
    for rank, ind in enumerate(r.top_features_):
        feature_ranks[ind] = rank

    colors = ['r', 'y', 'g', 'c', 'b', 'm', 'k']

    plt.figure()

    plt.subplot(2, 1, 1)
    ind = 0
    for i in range(len(nFeatures)):
        plt.bar(range(ind, ind + nFeatures[i]), r.feature_importances_[ind:(ind + nFeatures[i])], label=feature_names[i], color=colors[i])
        ind = ind + nFeatures[i]
    plt.bar(range(ind, len(r.feature_importances_)), r.feature_importances_[ind:], label='other', color=colors[-1])
    plt.grid()
    plt.xlabel('Feature')
    plt.ylabel('Feature importance weight')
    plt.legend(loc='best', shadow=True, fancybox=True, framealpha=0.7)

    plt.subplot(2, 1, 2)
    ind = 0
    for i in range(len(nFeatures)):
        plt.bar(feature_ranks[ind:(ind + nFeatures[i])], r.feature_importances_[ind:(ind + nFeatures[i])],
                label=feature_names[i], color=colors[i])
        ind = ind + nFeatures[i]
    plt.bar(feature_ranks[ind:], r.feature_importances_[ind:], label='other', color=colors[-1])
    plt.axvline(n_features, color='r', ls='--')
    plt.grid()
    plt.xlabel('Feature')
    plt.ylabel('Feature importance weight')
    plt.legend(loc='best', shadow=True, fancybox=True, framealpha=0.7)

    plt.suptitle('Feature Importance')
    plt.tight_layout()
    plt.show()
