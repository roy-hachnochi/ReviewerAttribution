import numpy as np
from Preprocess import *

# ======================================================================================================================
# dataset, labels = get_train("./datasets/dataset_bmj/train")
dataset, labels = get_test("./datasets/dataset_bmj/test")

# ======================================================================================================================
nTokensList = [len(text.split()) for text in dataset]

nTokens_mean = np.mean(nTokensList)
nTokens_std = np.std(nTokensList)
nTokens_median = np.median(nTokensList)

print("Mean number of tokens: {}".format(nTokens_mean))
print("STD number of tokens: {}".format(nTokens_std))
print("Median number of tokens: {}".format(nTokens_median))
