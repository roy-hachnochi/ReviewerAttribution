import heapq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
import math
import torch
from LanguageModels import calculate_perplexity, load_lm
from Preprocess import tokenize
from tqdm import tqdm

nltk.download('stopwords')

# ======================================================================================================================
class FeatureExtractor:
    def __init__(self, ignore=None, maxWords=np.inf):
        self.unigram = WordHistogram()
        self.bigram = WordHistogram()
        self.trigram = WordHistogram()
        self.fourgram = WordHistogram()
        self.fivegram = WordHistogram()
        self.LM_foldernames = []
        self.nTokens = []
        self.ignore = [] if ignore is None else ignore
        self.maxWords = maxWords
        self.nFeatures = 0

    def fit(self, dataset, nTokens, LM_foldername=None, labels=None):
        assert((LM_foldername is not None and labels is not None) or (LM_foldername is None and labels is None))
        datasetClean = [tokenize(text, maxWords=self.maxWords) for text in dataset]
        datasetClean = [[token for token in tokens if token not in self.ignore] for tokens in datasetClean]
        datasetBi = get_ngrams(datasetClean, n=2)
        datasetTri = get_ngrams(datasetClean, n=3)
        datasetFour = get_ngrams(datasetClean, n=4)
        datasetFive = get_ngrams(datasetClean, n=5)
        self.unigram.fit(datasetClean, N=nTokens[0])
        self.bigram.fit(datasetBi, N=nTokens[1])
        self.trigram.fit(datasetTri, N=nTokens[2])
        self.fourgram.fit(datasetFour, N=nTokens[3])
        self.fivegram.fit(datasetFive, N=nTokens[4])
        if LM_foldername is not None:
            self.LM_foldernames = [LM_foldername + label for label in labels]
        self.nTokens = nTokens
        # self.nFeatures = sum(self.nTokens) + len(self.LM_foldernames)
        self.nFeatures = sum(self.nTokens) + 1  # TODO: decide between argmin and vector for ppl

    def transform(self, dataset):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        datasetClean = [tokenize(text, maxWords=self.maxWords) for text in dataset]
        datasetClean = [[token for token in corpus if token not in self.ignore] for corpus in datasetClean]
        datasetBi = get_ngrams(datasetClean, n=2)
        datasetTri = get_ngrams(datasetClean, n=3)
        datasetFour = get_ngrams(datasetClean, n=4)
        datasetFive = get_ngrams(datasetClean, n=5)

        N = len(datasetClean)
        X = np.zeros((N, self.nFeatures))
        featuresInds = np.cumsum(self.nTokens)
        for ind, (corpus, corpusBi, corpusTri, corpusFour, corpusFive) in enumerate(zip(datasetClean, datasetBi, datasetTri, datasetFour, datasetFive)):
            X[ind, :featuresInds[0]] = self.unigram.transform(corpus)
            X[ind, featuresInds[0]:featuresInds[1]] = self.bigram.transform(corpusBi)
            X[ind, featuresInds[1]:featuresInds[2]] = self.trigram.transform(corpusTri)
            X[ind, featuresInds[2]:featuresInds[3]] = self.fourgram.transform(corpusFour)
            X[ind, featuresInds[3]:featuresInds[4]] = self.fivegram.transform(corpusFive)

        datasetReduced = [text.split() for text in dataset]
        datasetReduced = [' '.join(tokens[:min(self.maxWords, len(tokens))]) for tokens in datasetReduced]
        ppl_mat = np.zeros((X.shape[0], len(self.LM_foldernames)))  # TODO: decide between argmin and vector for ppl
        for i_lm, lm_foldername in tqdm(enumerate(self.LM_foldernames)):
            model, tokenizer = load_lm(lm_foldername, device)
            for ind, text in enumerate(datasetReduced):
                ppl = calculate_perplexity(text, model, tokenizer, device)
                # X[ind, featuresInds[4] + i_lm] = ppl  # TODO: check min ppl value
                ppl_mat[ind, i_lm] = ppl  # TODO: decide between argmin and vector for ppl + return min with 5000 when svm only
        X[:, -1] = np.argmin(ppl_mat, axis=1)  # TODO: decide between argmin and vector for ppl
        return X

# ======================================================================================================================
class WordHistogram:
    def __init__(self):
        self.frequentWords = []
        self.idfDict = {}
        self.N = 0

    def fit(self, dataset, N=1000, ignore=None):
        wordHist = {}
        stop_words = set(stopwords.words("english"))
        for corpus in dataset:
            corpus = [word for word in corpus if word not in stop_words]  # TODO: maybe move outside
            for token in corpus:
                if ignore is None or token not in ignore:
                    if token in wordHist:
                        wordHist[token] += 1
                    else:
                        wordHist[token] = 1
        self.frequentWords = heapq.nlargest(N, wordHist, key=wordHist.get)
        self.N = N
        self.idfDict = idf(dataset, self.frequentWords)

    def transform(self, corpus):
        tfDict = tf(corpus, self.frequentWords)
        tfVec = np.array(list(tfDict.values()))
        idfVec = np.array(list(self.idfDict.values()))
        tfidf = tfVec * idfVec
        return tfidf

# ======================================================================================================================
def get_ngrams(dataset, n):
    nGramDataset = []
    for corpus in dataset:
        ngram = ngrams(corpus, n)
        nGramDataset.append([' '.join(gram) for gram in ngram])
    return nGramDataset

# ======================================================================================================================
def idf(dataset, words):
    N = len(dataset)
    idfDict = dict.fromkeys(words, 0)
    for corpus in dataset:
        for word in words:
            if word in corpus:
                idfDict[word] += 1
    for word, val in idfDict.items():
        idfDict[word] = math.log(N / float(val))
    return idfDict


def tf(corpus, words):
    tfDict = dict.fromkeys(words, 0)
    for token in corpus:
        if token in words:
            tfDict[token] += 1
    if len(corpus) == 0:
        print("BAD!")
    for token, val in tfDict.items():
        tfDict[token] = val / float(len(corpus))
    return tfDict


def average_word_length(corpus, ignore=None):
    words = corpus
    if ignore is not None:
        words = [word for word in words if word not in ignore]
    return sum(len(word) for word in words) / len(words)


def num_words(corpus, ignore=None):
    words = corpus
    if ignore is not None:
        words = [word for word in words if word not in ignore]
    return len(words)

# ======================================================================================================================
# TODO: out of date
def plot_features(features, nWords):
    freqWords = features[:nWords]
    avgWordLen = features[-2]
    numWords = features[-1]
    fig, axes = plt.subplots(1, 1)
    axes.bar(np.arange(nWords), freqWords)
    axes.set_title(
        'Average word length = {:.3f}\nNumber of words = {}\nMost Frequent Words Histogram:'.format(avgWordLen,
                                                                                                    numWords))
    axes.grid('on')
    fig.suptitle('Features', fontsize=14, fontweight='bold')
    plt.tight_layout()

# ======================================================================================================================
# TODO: out of date
def plot_features_compare(features, labels, nWords):
    freqWords = features[:, :nWords]
    avgWordLen = features[:, -2]
    numWords = features[:, -1]

    plt.subplot(2, 2, 1)
    plt.bar(np.arange(nWords), freqWords[0, :], alpha=0.5)
    plt.bar(np.arange(nWords), freqWords[1, :], alpha=0.5)
    plt.title('Most Frequent Words - Same Author')
    plt.grid('on')

    plt.subplot(2, 2, 2)
    plt.bar(np.arange(nWords), freqWords[0, :], alpha=0.5)
    plt.bar(np.arange(nWords), freqWords[10, :], alpha=0.5)
    plt.title('Most Frequent Words - Different Authors')
    plt.grid('on')

    plt.subplot(2, 1, 2)
    colors = cm.rainbow(np.linspace(0, 1, len(list(set(labels)))))
    plt.scatter(avgWordLen, numWords, c=colors[labels])
    plt.xlabel('Average Word Length')
    plt.ylabel('Number Of Words')
    plt.grid('on')

    plt.suptitle('Features Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
