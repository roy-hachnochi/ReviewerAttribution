from Preprocess import *
from FeatureExtractor import *
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import svm
from scipy import stats
import numpy as np
import os

# ======================================================================================================================
if __name__ == '__main__':
    nTokens = [0, 0, 0, 0, 0]
    LM_folderName = "./Language_Models/toy_60/"
    results_folderName = "./results/PerplexityClassification/toy/"
    pTrain = 0.6
    maxWords_list = [200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900]
    nSamples = 100  # number of times to calculate accuracy, used for creating confidence interval
    a = 0.05  # confidence interval

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

    acc_mean_argmin = []
    acc_mean_svm = []
    acc_std_argmin = []
    acc_std_svm = []
    for ind, maxWords in enumerate(maxWords_list):
        # extract features:
        print('{}: Extracting features...'.format(ind))
        feature_ext = FeatureExtractor(maxWords=maxWords)
        feature_ext.fit(dataset, nTokens, LM_foldername=LM_folderName, labels=list(labels_to_class_dict))
        X = feature_ext.transform(dataset)
        y = np.array([labels_to_class_dict[label] for label in labels])
        np.savetxt(results_folderName + "ppl_{}_words.csv".format(maxWords), X, delimiter=",")

        print('{}: Classifying...'.format(ind))
        accuracy_argmin_list = []
        accuracy_svm_list = []
        for i in range(nSamples):
            # train svm:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1-pTrain), random_state=i)
            scaler = preprocessing.StandardScaler().fit(X_train)  # TODO: maybe remove scaling
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            clf = svm.SVC(class_weight='balanced')
            clf.fit(X_train_scaled, y_train)

            # get prediction and calculate metrics:
            y_pred_argmin = np.argmin(X_test, axis=1)
            y_pred_svm = clf.predict(X_test_scaled)
            accuracy_argmin = (y_pred_argmin == y_test).mean()
            accuracy_svm = (y_pred_svm == y_test).mean()
            accuracy_argmin_list.append(accuracy_argmin)
            accuracy_svm_list.append(accuracy_svm)

        acc_mean_argmin.append(np.mean(accuracy_argmin_list))
        acc_mean_svm.append(np.mean(accuracy_svm_list))
        acc_std_argmin.append(np.std(accuracy_argmin_list))
        acc_std_svm.append(np.std(accuracy_svm_list))
        print('{0}: Argmin Accuracy: {1} | SVM Accuracy: {2}'.format(ind, acc_mean_argmin[-1], acc_mean_svm[-1]))

    np.savetxt(results_folderName + "ppl_acc_mean_argmin_unscaled.csv", np.array(acc_mean_argmin), delimiter=",")
    np.savetxt(results_folderName + "ppl_acc_mean_svm_unscaled.csv", np.array(acc_mean_svm), delimiter=",")
    np.savetxt(results_folderName + "ppl_acc_std_argmin_unscaled.csv", np.array(acc_std_argmin), delimiter=",")
    np.savetxt(results_folderName + "ppl_acc_std_svm_unscaled.csv", np.array(acc_std_svm), delimiter=",")

    # plot results:
    eps_argmin = stats.norm.ppf(1 - a / 2) * (np.array(acc_std_argmin) / np.sqrt(nSamples))
    eps_svm = stats.norm.ppf(1 - a / 2) * (np.array(acc_std_svm) / np.sqrt(nSamples))
    plt.figure()
    plt.errorbar(maxWords_list, acc_mean_argmin, yerr=eps_argmin, fmt='o-b', label="Argmin")
    plt.errorbar(maxWords_list, acc_mean_svm, yerr=eps_svm, fmt='o-r', label="SVM")
    plt.title('Accuracy vs. Text Length (Perplexity Only)')
    plt.xlabel('Max Number of Words')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.show()