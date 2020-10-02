from Preprocess import *

# ======================================================================================================================
if __name__ == '__main__':
    out_folderName = "./datasets/dataset_bmj/forLM/"
    endOfText_token = "\n\n<|EndOfText|>\n\n"

    if not os.path.isdir(out_folderName):
        os.mkdir(out_folderName)

    # load and preprocess dataset:
    print('Preprocessing data...')
    # dataset_train, labels_train = get_test("./datasets/dataset_bmj/test")
    dataset_train, labels_train = get_train("./datasets/dataset_bmj/train")
    # dataset_train, labels_train = get_train("./datasets/toy_data/train")

    # get labels dictionary:
    class_to_labels_dict = list(set(labels_train))
    labels_to_class_dict = {label: i for i, label in enumerate(class_to_labels_dict)}
    nLabels = len(class_to_labels_dict)

    # join texts for all labels:
    print('Creating corpora...')
    for label in class_to_labels_dict:
        texts = [' '.join(dataset_train[i]) for i, l in enumerate(labels_train) if l == label]  # get all texts for this author
        corpus = endOfText_token.join(texts)
        with open(out_folderName + label + ".txt", 'w+', encoding='utf-8') as file:
            file.write(corpus)
