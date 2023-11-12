import pandas as pd
import sys
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn

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
    
class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dims, padding_index, emote_map):
        super().__init__()
        self.emote_map = emote_map
        self.embeddings = nn.Embedding(vocab_size, embedding_dims, padding_idx=padding_index)

    def get_emote_weights(self, input):
        weights = torch.ones(len(input))
        for index, id in enumerate(input):
            id = id.item()
            if id in self.emote_map:
                weights[index] = self.emote_map[id]
        return weights

    def forward(self, input):
        return None

def test_val(emote_map, input):
    weights = torch.ones(len(input))
    for index, id in enumerate(input):
        id = id.item()
        if emote_map.get(id) is not None:
            weights[index] = emote_map[id]
    return weights

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

def create_emotions_map(corpus, vocab_map):
    mapping = dict()
    for _, row in corpus.iterrows():
        token = row['token']
        if vocab_map.get(token) is not None:
            mapping[vocab_map.get(token)] = row['score']
    return mapping

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
    emotions_map = create_emotions_map(emotion_corpus, vocab_map)
    print(emotions_map)

if __name__ == "__main__":
    main()
