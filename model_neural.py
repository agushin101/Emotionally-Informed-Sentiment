import pandas as pd
import sys
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim


class TextData(Dataset):
    def __init__(self, x, y, mapping):
        self.x = x
        self.y = y
        self.mapping = mapping
        self.max_len = 25
        self.padding_indx = len(mapping)

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
        
        kernel_size = 3
        self.pool = nn.AdaptiveMaxPool1d(kernel_size)
        #self.conv = nn.Conv1d(embedding_dims, 2 * embedding_dims, kernel_size)
        self.lin1 = nn.Linear(embedding_dims, 2 * embedding_dims)
        self.lin_classifier = nn.Linear(2 * embedding_dims, 2)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def get_emote_weights(self, input):
        results = list()
        for sample in input:
            weights = torch.ones(len(sample))
            for index, id in enumerate(sample):
                id = id.item()
                if id in self.emote_map:
                    weights[index] = self.emote_map[id]

            results.append(weights)
        return results
    
    def apply_weights(self, input):
        emote_weights = self.get_emote_weights(input)
        embeddings = self.embeddings(input)
        for i, tensor in enumerate(embeddings):
            weights = emote_weights[i]
            reweighted = self.sigmoid(torch.multiply(tensor, weights))
            embeddings[i] = reweighted
        return embeddings

    def forward(self, input):
        embeddings = self.embeddings(input)
        #pooled = self.pool(self.relu(self.conv(embeddings)))
        #pooled = torch.flatten(pooled, 1)
        lin1 = self.relu(self.lin1(embeddings))
        outputs = self.lin_classifier(self.dropout(lin1))
        outputs = outputs.long()
        return outputs

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

def train(dataloader, model, loss_function, optimizer):
    model.train()
    for x, y in dataloader:
        predicted = model(x)
        print(predicted)
        loss = loss_function(predicted, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def test(dataloader, model, loss_function):
    model.eval()

    with torch.no_grad():
        for x, y in dataloader:
            predicted = model(x)
            print(predicted)

def main():
    dataset = sys.argv[1] if sys.argv[1] is not None else "emotion"
    if dataset == "emotion":
        xTrain, xTest, yTrain, yTest = load_emotion()
    elif dataset == "sentiment":
        xTrain, xTest, yTrain, yTest = load_sentiment()
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

    model = SentimentModel(len(vocabulary) + 1, 25, len(vocabulary), emotions_map)
    lr, epochs = 0.001, 5
    loss = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr)

    for _ in range(epochs):
        train(trainloader, model, loss, optimizer)
    test(testloader, model, loss)

if __name__ == "__main__":
    main()
