from sklearn import preprocessing
from skrebate import ReliefF
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from sklearn import manifold
from sklearn.decomposition import KernelPCA

# ======================================================================================================================
if __name__ == '__main__':
    n_components = 3
    n_neighbors = 10
    nFeatures_factor = 0.5
    features_filename = "results/Main/articles_features.csv"
    labels_filename = "results/Main/articles_labels.csv"
    isArticlesReviews = True
    isUseArticles = False

    if isArticlesReviews:
        X = np.loadtxt("./results/Main/all_features.csv", delimiter=",")
        labels = np.loadtxt("./results/Main/all_labels.csv", delimiter=",", dtype='str')
        if isUseArticles:
            X = X[:70, :]
            labels = labels[:70]
        else:
            X = X[70:, :]
            labels = labels[70:]
    else:
        X = np.loadtxt(features_filename, delimiter=",")
        labels = list(np.loadtxt(labels_filename, delimiter=",", dtype='str'))

    # get labels dictionary:
    class_to_labels_dict = sorted(list(set(labels)))
    labels_to_class_dict = {label: i for i, label in enumerate(class_to_labels_dict)}
    y = np.array([labels_to_class_dict[label] for label in labels])

    # scaling:
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)

    # select good features:
    n_features = int(X.shape[1] * nFeatures_factor)
    r = ReliefF(n_features_to_select=n_features, n_neighbors=n_neighbors)
    r.fit(X, y)
    X = r.transform(X)

    # perform clustering:
    print('Clustering...')
    methods = OrderedDict()
    methods['KPCA'] = KernelPCA(n_components, kernel='sigmoid', random_state=0)
    # methods['LLE'] = manifold.LocallyLinearEmbedding(n_neighbors, n_components, method='standard')
    # methods['Isomap'] = manifold.Isomap(n_neighbors, n_components)
    # methods['MDS'] = manifold.MDS(n_components, max_iter=100, n_init=1)
    # methods['SE'] = manifold.SpectralEmbedding(n_components=n_components, n_neighbors=n_neighbors)
    methods['t-SNE'] = manifold.TSNE(n_components=n_components, init='pca', random_state=0)

    # Plot results
    fig = plt.figure(figsize=(15, 8))
    fig.suptitle("Manifold Learning with {} neighbors".format(n_neighbors), fontsize=14)
    for i, (label, method) in enumerate(methods.items()):
        Y = method.fit_transform(X)
        print("{}".format(label))
        if n_components == 2:
            ax = fig.add_subplot(1, 2, i + 1)
            ax.scatter(Y[:, 0], Y[:, 1], c=y, cmap=plt.cm.Spectral)
        elif n_components == 3:
            ax = fig.add_subplot(1, 2, i + 1, projection='3d')
            ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=y, cmap=plt.cm.Spectral)
        else:
            ax = fig.add_subplot(1, 2, i + 1)
            print("Can't plot components...")
        ax.set_title("{}".format(label))
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.axis('tight')
    plt.show()




    # # km = DBSCAN(eps=0.3, min_samples=5)
    # km = KMeans(max_iter=1000, n_clusters=k)
    # km.fit(X_train_scaled)
    # clusters = km.labels_
    #
    # # dimensionality reduction:
    # random_state = 20
    # np.random.seed(random_state)
    #
    # t_sne = TSNE(n_components=2, perplexity=30)
    # X_TSNE = t_sne.fit_transform(X_train_scaled)
    # pca = PCA(n_components=2, random_state=random_state)
    # X_PCA = pca.fit_transform(X_train_scaled)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(2, 2, 1)
    # ax.scatter(X_TSNE[:, 0], X_TSNE[:, 1], c=y_train, cmap=plt.cm.rainbow)
    # ax.set_title('tSNE - Ground Truth')
    #
    # ax = fig.add_subplot(2, 2, 2)
    # ax.scatter(X_TSNE[:, 0], X_TSNE[:, 1], c=clusters, cmap=plt.cm.rainbow)
    # ax.set_title('tSNE - Clusters')
    #
    # ax = fig.add_subplot(2, 2, 3)
    # ax.scatter(X_PCA[:, 0], X_PCA[:, 1], c=y_train, cmap=plt.cm.rainbow)
    # ax.set_title('PCA - Ground Truth')
    #
    # ax = fig.add_subplot(2, 2, 4)
    # ax.scatter(X_PCA[:, 0], X_PCA[:, 1], c=clusters, cmap=plt.cm.rainbow)
    # ax.set_title('PCA - Clusters')
    #
    # plt.tight_layout()
    # plt.show()


