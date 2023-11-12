import pandas as pd
import torch
from torch.utils.data import Dataset
import sys
from torch.utils.data import DataLoader

class TextData(Dataset):
    def __init__(self, x, y, mapping):
        self.x = x
        self.y = y
        self.mapping = mapping
        self.max_len = 25
        self.padding_indx = -1

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        raw_text = self.x.iloc[index]['text']
        label = self.y.iloc[index]['target']
        encodings = list()
        splits = raw_text.split()
        for word in splits:
            if len(encodings) >= self.max_len:
                break
            if self.mapping.get(word) is not None:
                encodings.append(self.mapping.get(word))
        while len(encodings) < self.max_len:
            encodings.append(self.padding_indx)
        return torch.LongTensor(encodings), label

def load_emotion():
    path = "data/emotion/"
    xTrain = pd.read_csv(path + "xTrain.csv")
    xTest = pd.read_csv(path + "xTest.csv")
    yTrain = pd.read_csv(path + "yTrain.csv")
    yTest = pd.read_csv(path + "yTest.csv")
    return xTrain, xTest, yTrain, yTest

def load_sentiment():
    return None

def build_vocab(data):
    vocabulary = set()
    for raw_text in data['text']:
        splits = raw_text.split()
        for word in splits:
            vocabulary.add(word)
    return vocabulary

def get_word_mapping(vocab):
    return {word: i for i, word in enumerate(vocab)}

def main():
    dataset = sys.argv[1] if sys.argv[1] is not None else "emotion"
    if dataset == "emotion":
        xTrain, xTest, yTrain, yTest = load_emotion()
    elif dataset == "sentiment":
        train, test = load_sentiment()
    else:
        print("Dataset must be one of emotion or sentiment.")
        exit(1)

    vocabulary = build_vocab(xTrain)
    vocab_map = get_word_mapping(vocabulary)
    emotion_corpus = pd.read_csv("emotion-corpus/corpus.csv")

    ####Remember, padding index is -1.
    traindata = TextData(xTrain, yTrain, vocab_map)
    testdata = TextData(xTest, yTest, vocab_map)

    trainloader = DataLoader(traindata, batch_size=32)
    testloader = DataLoader(testdata, batch_size=32)

if __name__ == "__main__":
    main()
