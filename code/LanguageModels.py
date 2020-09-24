from Preprocess import *
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import pickle
import math

# ======================================================================================================================
class Vocabulary:
    def __init__(self):
        self.word2int = {}
        self.int2word = {}
        self.nWords = 0

    def fit(self, corpus):
        uniqueWords = list(set(corpus.split()))
        self.int2word = {i: word for i, word in enumerate(uniqueWords)}
        self.word2int = {word: i for i, word in self.int2word.items()}
        self.nWords = len(self.int2word)

    def get_int(self, word):
        try:
            return self.word2int[word]
        except KeyError:
            return -1

    def get_word(self, i):
        try:
            return self.int2word[i]
        except KeyError:
            return ""

    def transform(self, corpus):
        return [self.get_int(word) for word in corpus.split()]

    def save(self, fileName):
        to_save = {'word2int': self.word2int,
                   'int2word': self.int2word}
        with open(fileName, 'wb+') as file:
            pickle.dump(to_save, file)

    def load(self, fileName):
        with open(fileName, 'rb') as file:
            loaded = pickle.load(file)
        self.word2int = loaded['word2int']
        self.int2word = loaded['int2word']
        self.nWords = len(self.int2word)

# ======================================================================================================================
class LanguageModelNN(nn.Module):
    def __init__(self, n_vocab):
        super(LanguageModelNN, self).__init__()
        self.lstm_size = 32
        self.embedding_dim = 256
        self.num_layers = 1

        self.embedding = nn.Embedding(num_embeddings=n_vocab, embedding_dim=self.embedding_dim)
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.lstm_size, num_layers=self.num_layers, dropout=0.2)
        self.fc = nn.Linear(self.lstm_size, n_vocab)

    def forward(self, x, prev_state):
        wordEmbedding = self.embedding(x)
        output, state = self.lstm(wordEmbedding, prev_state)
        logits = self.fc(output)
        return logits, state

    def init_state(self, seq_length):
        return (torch.zeros(self.num_layers, seq_length, self.lstm_size),
                torch.zeros(self.num_layers, seq_length, self.lstm_size))

# ======================================================================================================================
class LanguageModel:
    def __init__(self, fileName=None):
        self.NN = None
        self.vocab = Vocabulary()
        if fileName is not None:
            self.load(fileName)

    def save(self, fileName):
        torch.save(self.NN.state_dict(), fileName + "_state_dict.pt")
        self.vocab.save(fileName + "_vocab")
        params = {'lstm_size': self.NN.lstm_size,
                  'embedding_dim': self.NN.embedding_dim,
                  'num_layers': self.NN.num_layers}
        with open(fileName + "_params", 'wb+') as file:
            pickle.dump(params, file)

    def load(self, fileName):
        self.vocab.load(fileName + "_vocab")
        self.NN = LanguageModelNN(self.vocab.nWords)
        self.NN.load_state_dict(torch.load(fileName + "_state_dict.pt"))
        with open(fileName + "_params", 'rb') as file:
            params = pickle.load(file)
        self.NN.lstm_size = params['lstm_size']
        self.NN.embedding_dim = params['embedding_dim']
        self.NN.num_layers = params['num_layers']

    def calc_perplexity(self, words):
        window_size = 5
        self.NN.eval()

        state_h, state_c = self.NN.init_state(window_size)

        ent = 0
        for i in range(0, len(words) - window_size):
            # insert previous words to model, to get probabilities for next word:
            x = torch.tensor([[self.vocab.get_int(w) for w in words[i:(i + window_size)]]])
            y_pred, (state_h, state_c) = self.NN(x, (state_h, state_c))

            # get probabilities for next word:
            last_word_logits = y_pred[0][-1]
            p = F.softmax(last_word_logits, dim=0).detach().numpy()

            # get probability of true next word:
            wordInd = self.vocab.get_int(words[i + window_size])
            if wordInd >= 0:  # make sure that word exists in vocabulary
                ent -= math.log(p[wordInd])

        return math.exp(ent / (len(words) - window_size))

# ======================================================================================================================
class Dataset(torch.utils.data.Dataset):
    def __init__(self, seq_length):
        self.seq_length = seq_length
        self.vocab = Vocabulary()
        self.data = []

    def prepare(self, texts):
        corpus = [' '.join(text) for text in texts]
        corpus = ' '.join(corpus)
        self.vocab.fit(corpus)
        self.data = self.vocab.transform(corpus)

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        return (torch.tensor(self.data[index:(index + self.seq_length)]),
                torch.tensor(self.data[(index + 1):(index + self.seq_length + 1)]))

# ======================================================================================================================
def train(dataset, model, batch_size, max_epochs, lr, device, seq_length):
    model.train()

    dataloader = DataLoader(dataset, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(max_epochs):
        startTime = time.time()
        state_h, state_c = model.init_state(seq_length)
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            y_pred, (state_h, state_c) = model(x, (state_h, state_c))
            loss = criterion(y_pred.transpose(1, 2), y)  # TODO: why transpose?

            state_h = state_h.detach()
            state_c = state_c.detach()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        endTime = time.time()
        print('Epoch: {} | Loss = {:.3f} | Elapsed time: {:.3f}'.format(epoch, loss.item(), endTime - startTime))
    model.vocab = dataset.vocab

# ======================================================================================================================
def predict(vocab, model, text, next_words=100):
    model.eval()

    words = text.split(' ')
    state_h, state_c = model.init_state(len(words))

    for i in range(0, next_words):
        x = torch.tensor([[vocab.get_int(w) for w in words[i:]]])
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))

        last_word_logits = y_pred[0][-1]
        p = F.softmax(last_word_logits, dim=0).detach().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(vocab.get_word(word_index))

    return ' '.join(words)

# ======================================================================================================================
if __name__ == '__main__':
    # hyper-parameters:
    # TODO: set hyper-parameters
    # TODO: function for calculating perplexity
    seq_length = 32
    batch_size = 32
    max_epochs = 2
    seq_length = 32
    lr = 0.0001
    LM_folderName = "./Language_Models/"

    if not os.path.isdir(LM_folderName):
        os.mkdir(LM_folderName)

    # load and preprocess dataset:
    print('Preprocessing data...')
    dataset_train, labels_train = get_test("./datasets/dataset_bmj/test")

    # get labels dictionary:
    class_to_labels_dict = list(set(labels_train))
    labels_to_class_dict = {label: i for i, label in enumerate(class_to_labels_dict)}
    nLabels = len(class_to_labels_dict)

    # create and train Language Model for each label:
    print('Training Language Model...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device: ' + str(device))
    for label in class_to_labels_dict:
        texts = [dataset_train[i] for i, l in enumerate(labels_train) if l == label]  # get all texts for this author
        dataset = Dataset(seq_length)
        dataset.prepare(texts)
        model = LanguageModel()
        model.NN = LanguageModelNN(dataset.vocab.nWords).to(device)
        model.vocab = dataset.vocab
        train(dataset, model.NN, batch_size, max_epochs, lr, device, seq_length)
        print(predict(dataset.vocab, model.NN, text='i'))
        model.save(LM_folderName + label)
        print("perplexity = {}".format(model.calc_perplexity(texts[0])))

    print()



