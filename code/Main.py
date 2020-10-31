from Preprocess import *
from FeatureExtractor import *
from sklearn import preprocessing
from sklearn import svm
from sklearn.cluster import OPTICS
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import plot_confusion_matrix, confusion_matrix
from skrebate import ReliefF
import numpy as np
import seaborn as sn
import pandas as pd
import os
import matplotlib.pyplot as plt

# ======================================================================================================================
if __name__ == '__main__':
    nTokens = [50, 70, 100, 30, 15]  # number of tokens for each n-gram histogram
    ignore = ['.', '[', ']', '/', '(', ')', ';', UNK_TOKEN]  # tokens to ignore
    pTrain = 0.7  # train-test split
    min_samples = 7  # minimum samples for clustering algorithm
    nFeatures_factor = 0.2  # factor for feature selection
    LM_folderName = "./Language_Models/toy_60/"  # "./Language_Models/articles_all/", "./Language_Models/articles_70/", "./Language_Models/reviews_70/"
    results_folderName = "./results/main/"
    features_fileName = "toy_features.csv"
    labels_fileName = "toy_labels.csv"
    loadData = True  # load features and labels instead of creating them
    saveFeatures = False  # save feature matrices and labels
    plotFeatures = False  # plot example of features
    plotConfMat = True  # plot confusion matrix
    isSplitTrainTest = True  # perform train-test split, if False - read separate train and test data
    isCrossVal = True  # perform cross validation

    os.makedirs(results_folderName, exist_ok=True)

    if loadData:
        if isSplitTrainTest:
            X = np.loadtxt(results_folderName + features_fileName, delimiter=",")
            labels = list(np.loadtxt(results_folderName + labels_fileName, delimiter=",", dtype='str'))
            X_train, labels_train, X_test, labels_test = test_train_split(X, labels, pTrain)
            X_train = np.array(X_train)
            X_test = np.array(X_test)
        else:
            X = np.loadtxt("./results/main/all_features.csv", delimiter=",")
            labels = np.loadtxt("./results/main/all_labels.csv", delimiter=",", dtype='str')
            X_train = X[:70, :]
            labels_train = labels[:70]
            X_test = X[70:, :]
            labels_test = labels[70:]
    else:
        # load and preprocess dataset:
        print('Preprocessing Data...')
        if isSplitTrainTest:
            # dataset, labels = get_train("./datasets/toy_data/train")
            dataset, labels = get_train("./datasets/dataset_bmj/train")
            # dataset, labels = get_test("./datasets/dataset_bmj/test")
            dataset_train, labels_train, dataset_test, labels_test = test_train_split(dataset, labels, pTrain)
        else:
            dataset_train, labels_train = get_train("./datasets/dataset_bmj/train")
            dataset_test, labels_test = get_test("./datasets/dataset_bmj/test")
        if sorted(list(set(labels_train))) != sorted(list(set(labels_test))):
            print("Error: labels_train and labels_test are different")
            exit()

        # extract features:
        print('Extracting features...')
        feature_ext = FeatureExtractor(ignore=ignore)
        feature_ext.fit(dataset_train, nTokens, LM_foldername=LM_folderName, labels=list(set(labels_train)))
        print(feature_ext.unigram.frequentWords)
        print(feature_ext.bigram.frequentWords)
        print(feature_ext.trigram.frequentWords)
        X_train = feature_ext.transform(dataset_train)
        X_test = feature_ext.transform(dataset_test)

    # get labels dictionary:
    class_to_labels_dict = sorted(list(set(labels_train)))
    labels_to_class_dict = {label: i for i, label in enumerate(class_to_labels_dict)}
    y_train = np.array([labels_to_class_dict[label] for label in labels_train])
    y_test = np.array([labels_to_class_dict[label] for label in labels_test])

    # plot example of features:
    if saveFeatures:
        np.savetxt(results_folderName + features_fileName, np.concatenate([X_train, X_test], axis=0), delimiter=",")
        np.savetxt(results_folderName + labels_fileName, np.concatenate([labels_train, labels_test], axis=0),
                   delimiter=",", fmt='%s')
    if plotFeatures:
        plot_features(X_train[3, :], nTokens)
        plot_features_compare(X_train, y_train, nTokens)

    # remove outliers from train data:
    X_train_inliers = []
    y_train_inliers = []
    for label in labels_to_class_dict.values():
        X = X_train[y_train == label, :]
        y = y_train[y_train == label]
        clusters = OPTICS(min_samples=min_samples).fit_predict(X)
        print(clusters)  # TODO: for debug, remove later
        X_train_inliers.append(X[clusters != -1, :])
        y_train_inliers.append(y[clusters != -1])
    X_train = np.concatenate(X_train_inliers, axis=0)
    y_train = np.concatenate(y_train_inliers)

    # scaling:
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # select good features:
    n_features = int(X_train_scaled.shape[1] * nFeatures_factor)
    n_neighbors = 10
    r = ReliefF(n_features_to_select=n_features, n_neighbors=n_neighbors)
    r.fit(X_train_scaled, y_train)
    X_train_scaled = r.transform(X_train_scaled)
    X_test_scaled = r.transform(X_test_scaled)

    # train model:
    print('Training Model...')
    clf = svm.SVC(class_weight='balanced')
    clf.fit(X_train_scaled, y_train)

    # get prediction and calculate metrics:
    print('Classifying...')
    y_pred = clf.predict(X_test_scaled)
    accuracy = (y_pred == y_test).mean()
    for i in range(len(y_pred)):
        print('Predicted: {0} | Label: {1}'.format(class_to_labels_dict[y_pred[i]], class_to_labels_dict[y_test[i]]))
    print('Accuracy: {0}'.format(accuracy))
    if plotConfMat:
        disp = plot_confusion_matrix(clf, X_test_scaled, y_test, display_labels=class_to_labels_dict, cmap=plt.cm.Blues,
                                     normalize='true')
        disp.ax_.set_title('Confusion Matrix')
        plt.xticks([], [])
        plt.show()

    # cross validation:
    if isCrossVal:
        if not isSplitTrainTest:
            print('Can\'t perform cross validation for articles-reviews setting')
        else:
            print('Cross Validation...')
            X = np.concatenate([X_train, X_test], axis=0)
            y = np.concatenate([y_train, y_test], axis=0)
            scaler = preprocessing.StandardScaler().fit(X)
            X_scaled = scaler.transform(X)
            X_scaled = r.transform(X_scaled)
            clf = svm.SVC(class_weight='balanced')
            y_pred = cross_val_predict(clf, X_scaled, y, cv=10)
            accuracy = (y_pred == y).mean()
            print('Accuracy: {0}'.format(accuracy))
            if plotConfMat:
                conf_mat = confusion_matrix(y, y_pred)
                df_cm = pd.DataFrame(conf_mat, index=[label for label in class_to_labels_dict],
                                     columns=[label.split()[-1] for label in class_to_labels_dict])
                plt.figure()
                sn.heatmap(df_cm, annot=True, cmap="Blues")
                plt.title('Cross Validation Results')
                plt.show()
