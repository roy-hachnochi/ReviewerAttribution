import torch
import os
from LanguageModels import calculate_perplexity, load_lm
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dataset_folder = "./datasets/dataset_bmj/forLM/"
    lm_folder = "./Language_Models/"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_files = os.listdir(dataset_folder)

    ppl_matrix = np.zeros((len(data_files), len(data_files)))
    for iLM, lm_filename in enumerate(data_files):
        lm_author = lm_filename.split(".")[0]
        model, tokenizer = load_lm(lm_folder + lm_author, device)
        for iData, data_filename in enumerate(data_files):
            author = data_filename.split(".")[0]
            text = open(dataset_folder + data_filename, "r").read()
            ppl_matrix[iData, iLM] = calculate_perplexity(text, model, tokenizer, device)

    authors = [filename.split(".")[0] for filename in data_files]
    plt.matshow(-ppl_matrix)
    plt.colorbar()
    plt.title("Author Perplexity vs. Language Model")
    plt.xlabel("LM")
    plt.ylabel("Author")
    plt.xticks([], [])
    plt.yticks(range(len(authors)), authors)
    plt.show()


