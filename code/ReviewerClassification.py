from Preprocess import *
from FeatureExtractor import *
from sklearn import preprocessing
from sklearn import svm
import numpy as np
from sklearn.metrics import plot_confusion_matrix

# ======================================================================================================================
def test_train_split(X, y, pTrain):
    nSamples = len(X)

    randGen = np.random.RandomState(0)
    indices = np.arange(0, nSamples)
    randGen.shuffle(indices)

    n_samples_train = int(nSamples * pTrain)
    train_indices = indices[:n_samples_train]
    test_indices = indices[n_samples_train:]

    X_train = [X[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    y_train = [y[i] for i in train_indices]
    y_test = [y[i] for i in test_indices]

    return X_train, y_train, X_test, y_test


def test_train_split_improved(X, y, pTrain):
    nSamples = len(X)
    randGen = np.random.RandomState(0)

    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for label in list(set(y)):
        indices = [i for i in range(nSamples) if y[i] == label]  # appearances of current label
        n = len(indices)
        n_samples_train = int(n * pTrain)
        randGen.shuffle(indices)
        train_indices = indices[:n_samples_train]
        test_indices = indices[n_samples_train:]
        X_train += [X[i] for i in train_indices]
        X_test += [X[i] for i in test_indices]
        y_train += [y[i] for i in train_indices]
        y_test += [y[i] for i in test_indices]

    return X_train, y_train, X_test, y_test

# ======================================================================================================================
if __name__ == '__main__':
    nTokens = [50, 70, 100, 30, 15]
    ignore = ['.', '[', ']', '/', '(', ')', ';', UNK_TOKEN]
    pTrain = 0.6
    LM_folderName = "./Language_Models/"
    plotFeatures = False
    plotConfMat = True
    TrainReviews = True

    # load and preprocess dataset:
    print('Preprocessing Data...')
    if TrainReviews:
        dataset, labels = get_test("./datasets/dataset_bmj/test")
        dataset_train, labels_train, dataset_test, labels_test = test_train_split_improved(dataset, labels, pTrain)
    else:
        dataset_train, labels_train = get_train("./datasets/dataset_bmj/train")
        dataset_test, labels_test = get_test("./datasets/dataset_bmj/test")
    if sorted(list(set(labels_train))) != sorted(list(set(labels_test))):
        print("Error: labels_train and labels_test are different")
        exit()

    # get labels dictionary:
    class_to_labels_dict = list(set(labels_train))
    labels_to_class_dict = {label: i for i, label in enumerate(class_to_labels_dict)}

    # extract features:
    print('Extracting features...')
    feature_ext = FeatureExtractor(ignore=ignore)
    feature_ext.load_language_models(LM_folderName, class_to_labels_dict)
    feature_ext.fit(dataset_train, nTokens=nTokens)
    print(feature_ext.unigram.frequentWords)
    print(feature_ext.bigram.frequentWords)
    print(feature_ext.trigram.frequentWords)
    X_train = feature_ext.transform(dataset_train)
    y_train = np.array([labels_to_class_dict[label] for label in labels_train])
    X_test = feature_ext.transform(dataset_test)
    y_test = np.array([labels_to_class_dict[label] for label in labels_test])

    # plot example of features:
    if plotFeatures:
        plot_features(X_train[3, :], nTokens)
        plot_features_compare(X_train, y_train, nTokens)

    # train model:
    print('Training Model...')
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    clf = svm.SVC()
    clf.fit(X_train_scaled, y_train)

    # get prediction and calculate metrics:
    print('Classifying...')
    X_test_scaled = scaler.transform(X_test)
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
    # print('{}: Cross Validation...'.format(ind))
    # scaler = preprocessing.StandardScaler().fit(X)
    # X_scaled = scaler.transform(X)
    # clf = svm.SVC()
    # y_pred = cross_val_predict(clf, X_scaled, y, cv=10)
    # accuracy = (y_pred == y).mean()
    # accuracyList.append(accuracy)
    # print('{0}: Accuracy: {1}'.format(ind, accuracy))
    # if plotConfMat:
    #     conf_mat = confusion_matrix(y, y_pred)
    #     df_cm = pd.DataFrame(conf_mat, index=[label for label in class_to_labels_dict],
    #                          columns=[label.split()[-1] for label in class_to_labels_dict])
    #     plt.figure()
    #     sn.heatmap(df_cm, annot=True, cmap="Blues")
    #     plt.title('Cross Validation Results')
    #     plt.show()
    #
    # plt.figure()
    # plt.plot(np.inf, accuracyList, 'o-')
    # plt.title('Accuracy vs. Text Length')
    # plt.xlabel('Max Number of Words')
    # plt.ylabel('Accuracy')
    # plt.grid()
    # plt.show()
    # print()
