import pandas as pd
import torch
from torch.utils.data import Dataset
import sys
from torch.utils.data import DataLoader
import spacy

class TextData(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, index):
        raw_text = self.x.iloc[index]['text']
        label = self.y.iloc[index]['target']

        return raw_text, label

def load_emotion():
    path = "data/emotion/"
    xTrain = pd.read_csv(path + "xTrain.csv")
    xTest = pd.read_csv(path + "xTest.csv")
    yTrain = pd.read_csv(path + "yTrain.csv")
    yTest = pd.read_csv(path + "yTest.csv")
    return xTrain, xTest, yTrain, yTest

def load_sentiment():
    return None

def main():
    dataset = sys.argv[1] if sys.argv[1] is not None else "emotion"
    if dataset == "emotion":
        xTrain, xTest, yTrain, yTest = load_emotion()
    elif dataset == "sentiment":
        train, test = load_sentiment()
    else:
        print("Dataset must be one of emotion or sentiment.")
        exit(1)

    traindata = TextData(xTrain, yTrain)
    testdata = TextData(xTest, yTest)

    trainloader = DataLoader(traindata, batch_size=32)
    testloader = DataLoader(testdata, batch_size=32)

    emotion_corpus = pd.read_csv("emotion-corpus/corpus.csv")
    print(emotion_corpus)

if __name__ == "__main__":
    main()