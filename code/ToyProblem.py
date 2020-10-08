import os
import heapq
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

# ======================================================================================================================
def tokenize(filename):
    data = open(filename, "r", encoding='utf-8-sig').readlines()
    data = ''.join(str(line) for line in data)
    data = data.lower()
    signs = ['.', ';', ',', ':', '!', '?', '(', ')', '/', '\'', '\"', '[', ']', '{', '}']
    for sign in signs:
        data = data.replace(sign, " " + sign + " ")
    tokens = data.split()
    return tokens

# ======================================================================================================================
def preprocess(folderName, maxWords=np.inf):
    folders = []
    dataset = []
    labels = []
    dirs = os.listdir(folderName)
    for folder in dirs:
        if os.path.isdir(folderName + "/" + folder):
            folders.append(folder)
    for author in folders:
        files = os.listdir(folderName + "/" + author)
        for file in files:
            if os.path.isfile(folderName + "/" + author + "/" + file):
                tokens = tokenize(folderName + "/" + author + "/" + file)
                tokens = [tokens[i] for i in range(min(len(tokens), maxWords))]
                dataset.append(tokens)
                labels.append(author)
    return dataset, labels

# ======================================================================================================================
def test_train_split(X, y, p_train):
    n_samples = X.shape[0]

    rand_gen = np.random.RandomState(0)
    indices = np.arange(0, n_samples)
    rand_gen.shuffle(indices)

    n_samples_train = int(n_samples * p_train)
    train_indices = indices[:n_samples_train]
    test_indices = indices[n_samples_train:]

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, y_train, X_test, y_test

# ======================================================================================================================
class WordHistogram:
    def __init__(self):
        self.frequentWords = []
        self.nonFrequentWords = []
        self.N = 0

    def fit(self, dataset, N=1000, ignore=[]):
        wordHist = {}
        for corpus in dataset:
            for token in corpus:
                if token not in ignore:
                    if token in wordHist:
                        wordHist[token] += 1
                    else:
                        wordHist[token] = 1
        self.frequentWords = heapq.nlargest(N, wordHist, key=wordHist.get)
        self.nonFrequentWords = heapq.nsmallest(N, wordHist, key=wordHist.get)
        self.N = N

    def transform(self, corpus):
        wordHist = np.zeros(self.N * 2)
        for token in corpus:
            if token in self.frequentWords:
                wordHist[self.frequentWords.index(token)] += 1
            if token in self.nonFrequentWords:
                wordHist[self.N + self.nonFrequentWords.index(token)] += 1
        return wordHist / len(corpus)


def average_word_length(corpus, ignore=[]):
    words = [word for word in corpus if word not in ignore]
    return sum(len(word) for word in words) / len(words)


def num_words(corpus, ignore=[]):
    words = [word for word in corpus if word not in ignore]
    return len(words)

# ======================================================================================================================
class FeatureExtractor:
    def __init__(self):
        self.word_hist = WordHistogram()
        self.nWords = 0

    def fit(self, dataset, nWords=1000, ignore=[]):
        self.word_hist.fit(dataset, N=nWords, ignore=ignore)
        self.nWords = nWords

    def transform(self, dataset, ignore=[]):
        N = len(dataset)
        nFeatures = 2 * self.nWords + 2
        X = np.zeros((N, nFeatures))
        for ind, corpus in enumerate(dataset):
            X[ind, :-2] = self.word_hist.transform(corpus)
            X[ind, -2] = average_word_length(corpus, ignore=ignore)
            X[ind, -1] = num_words(corpus, ignore=ignore)
        X = np.array(X)
        return X

# ======================================================================================================================
def plot_features(features, nWords):
    freqWords = features[:nWords]
    nonFreqWords = features[nWords:(2 * nWords)]
    avgWordLen = features[-2]
    numWords = features[-1]
    fig, axes = plt.subplots(1, 1)
    axes.bar(np.arange(nWords), freqWords)
    axes.set_title(
        'Average word length = {:.3f}\nNumber of words = {}\nMost Frequent Words Histogram:'.format(avgWordLen,
                                                                                                    numWords))
    axes.grid('on')
    fig.suptitle('Features', fontsize=14, fontweight='bold')
    plt.tight_layout()

# ======================================================================================================================
# Note: out of date!
def plot_features_compare(features, labels, nWords):
    freqWords = features[:, :nWords]
    nonFreqWords = features[:, nWords:(2 * nWords)]
    avgWordLen = features[:, -2]
    numWords = features[:, -1]

    plt.subplot(2, 2, 1)
    plt.bar(np.arange(nWords), freqWords[0, :], alpha=0.5)
    plt.bar(np.arange(nWords), freqWords[1, :], alpha=0.5)
    plt.title('Most Frequent Words - Same Author')
    plt.grid('on')

    plt.subplot(2, 2, 2)
    plt.bar(np.arange(nWords), freqWords[0, :], alpha=0.5)
    plt.bar(np.arange(nWords), freqWords[10, :], alpha=0.5)
    plt.title('Most Frequent Words - Different Authors')
    plt.grid('on')

    plt.subplot(2, 1, 2)
    colors = cm.rainbow(np.linspace(0, 1, len(list(set(labels)))))
    plt.scatter(avgWordLen, numWords, c=colors[labels])
    plt.xlabel('Average Word Length')
    plt.ylabel('Number Of Words')
    plt.grid('on')

    plt.suptitle('Features Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()

# ======================================================================================================================
if __name__ == '__main__':
    nWords = 250
    p_train = 0.85
    maxWordsList = [200, 500, 1000, 5000, 10000, 20000, 40000, 50000, 70000, 80000, 100000, 150000, 200000, np.inf]
    plotFeatures = False
    plotConfMat = False

    accuracyList = []
    for ind, maxWords in enumerate(maxWordsList):
        # load and preprocess dataset:
        print('{}: Preprocessing Data...'.format(ind))

        dataset, labels = preprocess("../toy_data", maxWords=maxWords)
        labels_unique = list(set(labels))
        labels_to_class_dict = dict()
        class_to_labels_dict = []
        for i, label in enumerate(labels_unique):
            labels_to_class_dict[label] = i
            class_to_labels_dict.append(label)

        # extract features:
        print('{}: Extracting features...'.format(ind))
        feature_ext = FeatureExtractor()
        feature_ext.fit(dataset, nWords=nWords, ignore=['.', ','])
        print(feature_ext.word_hist.frequentWords)
        print(feature_ext.word_hist.nonFrequentWords)
        X = feature_ext.transform(dataset, ignore=['.', ','])
        y = np.array([labels_to_class_dict[label] for label in labels])

        # plot example of features:
        if plotFeatures:
            plot_features(X[3, :], nWords)
            plot_features_compare(X, y, nWords)

        # cross validation:
        print('{}: Cross Validation...'.format(ind))
        scaler = preprocessing.StandardScaler().fit(X)
        X_scaled = scaler.transform(X)
        clf = svm.SVC()
        y_pred = cross_val_predict(clf, X_scaled, y, cv=10)
        accuracy = (y_pred == y).mean()
        accuracyList.append(accuracy)
        print('{0}: Accuracy: {1}'.format(ind, accuracy))
        if plotConfMat:
            conf_mat = confusion_matrix(y, y_pred)
            df_cm = pd.DataFrame(conf_mat, index=[label for label in class_to_labels_dict],
                                 columns=[label.split()[-1] for label in class_to_labels_dict])
            plt.figure()
            sn.heatmap(df_cm, annot=True, cmap="Blues")
            plt.title('Cross Validation Results')
            plt.show()

        # train model:
        # print('Training Model...')
        # X_train, y_train, X_test, y_test = test_train_split(X, y, p_train)
        # scaler = preprocessing.StandardScaler().fit(X_train)
        # X_train_scaled = scaler.transform(X_train)
        # clf = svm.SVC()
        # clf.fit(X_train_scaled, y_train)

        # get prediction and calculate metrics:
        # print('Classifying...')
        # X_test_scaled = scaler.transform(X_test)
        # y_pred = clf.predict(X_test_scaled)
        # accuracy = (y_pred == y_test).mean()
        # for i in range(len(y_pred)):
        #     print('Predicted: {0} | Label: {1}'.format(class_to_labels_dict[y_pred[i]], class_to_labels_dict[y_test[i]]))
        # print('Accuracy: {0}'.format(accuracy))
        # disp = plot_confusion_matrix(clf, X_test_scaled, y_test, display_labels=class_to_labels_dict, cmap=plt.cm.Blues, normalize='true')
        # disp.ax_.set_title('Confusion Matrix')
        # plt.show()

    plt.figure()
    plt.plot(maxWordsList, accuracyList, 'o-')
    plt.title('Accuracy vs. Text Length')
    plt.xlabel('Max Number of Words')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()
    print()
