from Preprocess import *
from FeatureExtractor import *
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import svm
import numpy as np
import os

# ======================================================================================================================
if __name__ == '__main__':
    nTokens = [0, 0, 0, 0, 0]
    LM_folderName = "./Language_Models/toy_60/"
    results_folderName = "./results/ppl_classification/toy/"
    pTrain = 0.6
    # maxWords_list = [200, 500, 1000, 2000, 5000, 7000, 10000, 50000, 80000, 100000, 200000]
    maxWords_list = [8000, np.inf]

    # ==================================================================================================================
    os.makedirs(results_folderName, exist_ok=True)

    # load and preprocess dataset:
    print('Preprocessing Data...')
    # dataset, labels = get_test("./datasets/dataset_bmj/test")
    # dataset, labels = get_train("./datasets/dataset_bmj/train")
    dataset, labels = get_train("./datasets/toy_data/train")

    # get labels dictionary:
    class_to_labels_dict = list(set(labels))
    labels_to_class_dict = {label: i for i, label in enumerate(class_to_labels_dict)}

    acq_list_svm = []
    acq_list_argmin = []
    for ind, maxWords in enumerate(maxWords_list):
        # extract features:
        print('{}: Extracting features...'.format(ind))
        feature_ext = FeatureExtractor(maxWords=maxWords)
        feature_ext.fit(dataset, nTokens, LM_foldername=LM_folderName, labels=list(labels_to_class_dict))
        X = feature_ext.transform(dataset)
        y = np.array([labels_to_class_dict[label] for label in labels])
        np.savetxt(results_folderName + "ppl_{}_words.csv".format(maxWords), X, delimiter=",")

        # train svm:
        print('{}: Training Model...'.format(ind))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1-pTrain), random_state=0)
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        clf = svm.SVC(class_weight='balanced')
        clf.fit(X_train_scaled, y_train)

        # get prediction and calculate metrics:
        print('{}: Classifying...'.format(ind))
        y_pred_argmin = np.argmin(X_test, axis=1)
        y_pred_svm = clf.predict(X_test_scaled)
        accuracy_argmin = (y_pred_argmin == y_test).mean()
        accuracy_svm = (y_pred_svm == y_test).mean()
        print('{0}: Argmin Accuracy: {1} | SVM Accuracy: {2}'.format(ind, accuracy_argmin, accuracy_svm))
        acq_list_argmin.append(accuracy_argmin)
        acq_list_svm.append(accuracy_svm)

    np.savetxt(results_folderName + "ppl_acc_argmin_toyProblem.csv", np.array(acq_list_argmin), delimiter=",")
    np.savetxt(results_folderName + "ppl_acc_svm_toyProblem.csv", np.array(acq_list_svm), delimiter=",")
    plt.figure()
    plt.plot(maxWords_list, acq_list_argmin, 'o-b', label="argmin")
    plt.plot(maxWords_list, acq_list_svm, 'o-r', label="SVM")
    plt.title('Accuracy vs. Text Length (Perplexity Only)')
    plt.xlabel('Max Number of Words')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.show()
    print()