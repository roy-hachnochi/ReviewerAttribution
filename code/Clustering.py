from Preprocess import *
from FeatureExtractor import *
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

# ======================================================================================================================
if __name__ == '__main__':
    nTokens = [50, 70, 100, 30, 15]
    ignore = ['.', '[', ']', '/', '(', ')', ';', UNK_TOKEN]
    k = 9

    # load and preprocess dataset:
    print('Preprocessing Data...')
    dataset_train, labels_train = get_train("./datasets/dataset_bmj/train")

    # get labels dictionary:
    class_to_labels_dict = list(set(labels_train))
    labels_to_class_dict = {label: i for i, label in enumerate(class_to_labels_dict)}

    # extract features:
    print('Extracting features...')
    feature_ext = FeatureExtractor(ignore=ignore)
    feature_ext.fit(dataset_train, nTokens=nTokens)
    X_train = feature_ext.transform(dataset_train)
    y_train = np.array([labels_to_class_dict[label] for label in labels_train])
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    # perform clustering:
    print('Clustering...')
    # km = DBSCAN(eps=0.3, min_samples=5)
    km = KMeans(max_iter=1000, n_clusters=k)
    km.fit(X_train_scaled)
    clusters = km.labels_

    # dimensionality reduction:
    random_state = 20
    np.random.seed(random_state)

    t_sne = TSNE(n_components=2, perplexity=30)
    X_TSNE = t_sne.fit_transform(X_train_scaled)
    pca = PCA(n_components=2, random_state=random_state)
    X_PCA = pca.fit_transform(X_train_scaled)

    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1)
    ax.scatter(X_TSNE[:, 0], X_TSNE[:, 1], c=y_train, cmap=plt.cm.rainbow)
    ax.set_title('tSNE - Ground Truth')

    ax = fig.add_subplot(2, 2, 2)
    ax.scatter(X_TSNE[:, 0], X_TSNE[:, 1], c=clusters, cmap=plt.cm.rainbow)
    ax.set_title('tSNE - Clusters')

    ax = fig.add_subplot(2, 2, 3)
    ax.scatter(X_PCA[:, 0], X_PCA[:, 1], c=y_train, cmap=plt.cm.rainbow)
    ax.set_title('PCA - Ground Truth')

    ax = fig.add_subplot(2, 2, 4)
    ax.scatter(X_PCA[:, 0], X_PCA[:, 1], c=clusters, cmap=plt.cm.rainbow)
    ax.set_title('PCA - Clusters')

    plt.tight_layout()
    plt.show()


