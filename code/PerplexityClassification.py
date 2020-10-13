from Preprocess import *
from FeatureExtractor import *
from sklearn import preprocessing
from sklearn import svm
from sklearn.cluster import OPTICS
import numpy as np
from sklearn.metrics import plot_confusion_matrix

# ======================================================================================================================
if __name__ == '__main__':
    nTokens = [0, 0, 0, 0, 0]
    pTrain = 0.7
    LM_folderName = "./Language_Models/toy_models"
    maxWords_list = [50]  # [200, 500, 1000, 5000, 10000, 20000, 40000, 50000, 70000, 80000, 100000, 150000, 200000, np.inf]

    # ==================================================================================================================
    # load and preprocess dataset:
    print('Preprocessing Data...')
    # dataset, labels = get_test("./datasets/dataset_bmj/test")
    # dataset, labels = get_train("./datasets/dataset_bmj/train")
    dataset, labels = get_train("./datasets/toy_data/train")

    # get labels dictionary:
    class_to_labels_dict = list(set(labels))
    labels_to_class_dict = {label: i for i, label in enumerate(class_to_labels_dict)}

    acq_list = []
    for ind, maxWords in enumerate(maxWords_list):
        # extract features:
        print('{}: Extracting features...'.format(ind))
        feature_ext = FeatureExtractor(maxWords=maxWords)
        feature_ext.fit(dataset, nTokens, LM_foldername=LM_folderName, labels=labels)
        X = feature_ext.transform(dataset)
        y = np.array([labels_to_class_dict[label] for label in labels])

        # get prediction and calculate metrics:
        print('{}: Classifying...'.format(ind))
        y_pred = np.argmax(X, axis=1)
        accuracy = (y_pred == y).mean()
        for i in range(len(y_pred)):
            print('Predicted: {0} | Label: {1}'.format(class_to_labels_dict[y_pred[i]], class_to_labels_dict[y[i]]))
        print('{0}: Accuracy: {1}'.format(ind, accuracy))
        acq_list.append(accuracy)

    plt.figure()
    plt.plot(maxWords_list, acq_list, 'o-')
    plt.title('Accuracy vs. Text Length (Perplexity Only)')
    plt.xlabel('Max Number of Words')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()
    print()