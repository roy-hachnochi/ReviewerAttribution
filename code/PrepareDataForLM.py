from Preprocess import *
import os

# ======================================================================================================================
if __name__ == '__main__':
    out_folderName = "./datasets/dataset_bmj/forLM/articles_70/"
    endOfText_token = "\n\n<|EndOfText|>\n\n"
    pTrain = 0.7

    os.makedirs(out_folderName, exist_ok=True)

    # load and preprocess dataset:
    print('Preprocessing data...')
    #dataset_train, labels_train = get_test("./datasets/dataset_bmj/test")
    dataset_train, labels_train = get_train("./datasets/dataset_bmj/train")
    # dataset_train, labels_train = get_train("./datasets/toy_data/train")
    dataset_train, labels_train, _, _ = test_train_split(dataset_train, labels_train, pTrain)

    # get labels dictionary:
    class_to_labels_dict = list(set(labels_train))
    labels_to_class_dict = {label: i for i, label in enumerate(class_to_labels_dict)}
    nLabels = len(class_to_labels_dict)

    # join texts for all labels:
    print('Creating corpora...')
    for label in class_to_labels_dict:
        texts = [dataset_train[i] for i, l in enumerate(labels_train) if l == label]  # get all texts for this author
        corpus = endOfText_token.join(texts)
        with open(out_folderName + label + ".txt", 'w+', encoding='utf-8') as file:
            file.write(corpus)
