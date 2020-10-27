from skrebate import ReliefF
import numpy as np
import matplotlib.pyplot as plt

# ======================================================================================================================
if __name__ == '__main__':
    features_filename = "./results/reviewer_classification/toy_features.csv"
    labels_filename = "./results/reviewer_classification/toy_labels.csv"
    n_features = 10
    n_neighbors = 10
    nFeatures = [50, 70, 100, 30, 15, 10]  # number of features for each feature type
    feature_names = ['unigram hist', 'bigram hist', 'trigram hist', 'fourgram hist', 'fivegram hist', 'LM perplexity']

    X = np.loadtxt(features_filename, delimiter=",")
    labels = np.loadtxt(labels_filename, delimiter=",", dtype='str')
    class_to_labels_dict = list(set(labels))
    labels_to_class_dict = {label: i for i, label in enumerate(class_to_labels_dict)}
    y = np.array([labels_to_class_dict[label] for label in labels])

    r = ReliefF(n_features_to_select=n_features, n_neighbors=n_neighbors)
    r.fit(X, y)

    colors = ['r', 'y', 'g', 'c', 'b', 'm']
    plt.figure()
    ind = 0
    for i in range(len(nFeatures)):
        plt.bar(range(ind, ind + nFeatures[i]), r.feature_importances_[ind:(ind + nFeatures[i])], label=feature_names[i], color=colors[i])
        ind = ind + nFeatures[i]
    plt.grid()
    plt.title('Feature Importance')
    plt.xlabel('Feature')
    plt.ylabel('Feature importance weight')
    plt.legend(loc='best', shadow=True, fancybox=True, framealpha=0.7)
    plt.show()
    print()


