import os
import numpy as np
import csv
from nltk.stem.wordnet import WordNetLemmatizer
import re

UNK_TOKEN = '<UNK>'

# ======================================================================================================================
def load_text(filename):
    text = open(filename, "r", encoding='utf-8-sig').readlines()
    text = ''.join(str(line) for line in text)
    text = text.lower()
    return text

def tokenize(data, maxWords=np.inf):
    # Convert data from string to list of words/tokens.
    signs = ['.', ',', ':', '!', '?', '(', ')', '/', '\'', '\"', '“', '[', ']', '{', '}', ';', '-', '<', '>']
    for sign in signs:
        data = data.replace(sign, " " + sign + " ")
    tokens = data.split()
    tokens = tokens[:min(len(tokens), maxWords)]

    lem = WordNetLemmatizer()
    for i, word in enumerate(tokens):
        word = lem.lemmatize(word, "v")
        word = re.sub(r'[0-9][a-zA-Z]*[0-9]*', UNK_TOKEN, word)  # replace anything containing numbers with UNK
        tokens[i] = UNK_TOKEN if UNK_TOKEN in word else word
    return tokens

# ======================================================================================================================
def get_train(folderName):
    # Load train data.
    # Assume that training data is organized in folders, where the name of the folder is the label of all files inside.
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
                text = load_text(folderName + "/" + author + "/" + file)
                dataset.append(text)
                labels.append(author.lower())
    return dataset, labels

# ======================================================================================================================
def get_test(folderName, fileNameInd=0, labelsInd=1):
    # Load test data. Loads all texts from labels file.
    # Assume that labels are kept in a csv file named "labels.csv".
    with open(folderName + "/labels.csv", mode='r', newline='') as labels_file:
        labels_reader = csv.reader(labels_file, delimiter=',')
        labels_all = [row for row in labels_reader]  # list of lists of all labels

    # get file names and actual labels:
    fileNames = [lbls[fileNameInd] for lbls in labels_all]
    labels = [lbls[labelsInd] for lbls in labels_all]

    # load files:
    dataset = []
    for fileName in fileNames:
        text = load_text(folderName + "/" + fileName + ".txt")
        dataset.append(text)
    return dataset, labels

# ======================================================================================================================
def test_train_split(X, y, pTrain):
    nSamples = len(X)
    randGen = np.random.RandomState(seed=0)

    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for label in sorted(list(set(y))):
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
    print("Test get_train:")
    trainData, trainLabels = get_train("./datasets/dataset_bmj/train")
    print("Number of train samples: {}".format(len(trainData)))
    print("Example of train sample: {}".format(trainData[0]))
    print("Train labels: {}".format(trainLabels))

    print("Test get_test:")
    testData, testLabels = get_test("./datasets/dataset_bmj/test")
    print("Number of test samples: {}".format(len(testData)))
    print("Example of test sample: {}".format(testData[0]))
    print("Test labels: {}".format(testLabels))
