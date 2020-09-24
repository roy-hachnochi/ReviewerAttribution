import os
import numpy as np
import csv
from nltk.stem.wordnet import WordNetLemmatizer
import re

UNK_TOKEN = '<UNK>'

# ======================================================================================================================
def tokenize(filename):
    # Read text from file and convert from string to list of words/tokens.
    data = open(filename, "r", encoding='utf-8-sig').readlines()
    data = ''.join(str(line) for line in data)
    data = data.lower()
    signs = ['.', ',', ':', '!', '?', '(', ')', '/', '\'', '\"', '[', ']', '{', '}', ';', '-']
    for sign in signs:
        data = data.replace(sign, " " + sign + " ")
    tokens = data.split()

    lem = WordNetLemmatizer()
    tokens = [lem.lemmatize(word, "v") for word in tokens]
    tokens = [re.sub(r'[0-9][a-zA-Z]*[0-9]*', UNK_TOKEN, word) for word in tokens]
    tokens = [UNK_TOKEN if UNK_TOKEN in word else word for word in tokens]
    return tokens

# ======================================================================================================================
def get_train(folderName, maxWords=np.inf):
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
                tokens = tokenize(folderName + "/" + author + "/" + file)
                tokens = [tokens[i] for i in range(min(len(tokens), maxWords))]
                dataset.append(tokens)
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
        tokens = tokenize(folderName + "/" + fileName + ".txt")
        dataset.append(tokens)
    return dataset, labels

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
