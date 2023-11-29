import pandas as pd
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import numpy as np


def load_emotion():
    path = "data/emotion/"
    xTrain = pd.read_csv(path + "xTrain.csv")
    xTest = pd.read_csv(path + "xTest.csv")
    yTrain = pd.read_csv(path + "yTrain.csv")
    yTest = pd.read_csv(path + "yTest.csv")
    return xTrain, xTest, yTrain, yTest

def select_k(dataframe, map_function, inverse, k):
    result = list()
    for text in dataframe['text']:
        if len(text.split()) <= k:
            result.append(text)
        else:
            string = ""
            scores = list()
            for word in text.split():
                val = map_function.get(word)
                if val is not None:
                    scores.append(val)
                sort = sorted(scores)
            for i, score in enumerate(sort):
                appender = inverse.get(score)
                string += appender
                if (i + 1) >= k:
                    break
                else:
                    string += " "
            if len(string.split()) < k:
                string += " "
                splits = text.split()
                selects = np.random.choice(len(splits), k - len(string.split()), replace=False).tolist()
                adds = [splits[i] for i in selects]
                for i, word in enumerate(adds):
                    string += word
                    if i < k:
                        string += " "
            result.append(string) if string != "" else result.append(text)
    return result

def select_k_random(dataframe, k):
    result = list()
    for text in dataframe['text']:
        splits = text.split()
        if len(splits) <= k:
            result.append(text)
        else:
            string = ""
            selects = np.random.choice(len(splits), k).tolist()
            choices = [splits[i] for i in selects]
            for i, word in enumerate(choices):
                string += word
                if (i + 1) < k:
                    string += " "
            result.append(string)
    return result

def build_mappings(corpus):
    result = dict()
    inverse = dict()
    for i, word in enumerate(corpus['token']):
        score = abs(corpus.iloc[i]['score'])
        result[word] = score
        inverse[score] = word
    return result, inverse


def main(k):
    xTrain, xTest, yTrain, yTest = load_emotion()
    yTrain = yTrain['target']
    yTest = yTest['target']

    ###Handle the non-random selects first

    emotion_corpus = pd.read_csv("emotion-corpus/corpus.csv")
    map_func, inverse = build_mappings(emotion_corpus)
    xTrain_k = select_k(xTrain, map_func, inverse, k)

    vectorizor = CountVectorizer(input='content', decode_error='ignore')
    train = vectorizor.fit_transform(xTrain_k)
    test = vectorizor.transform(xTest['text'])
    model = MultinomialNB()
    model.fit(train, yTrain)

    y_hat_test = model.predict(test)

    f_test = f1_score(yTest, y_hat_test)
    acc_test = accuracy_score(yTest, y_hat_test)

    ###Now, handle the random selects
    xTrain_rand = select_k_random(xTrain, k)
    train_rand = vectorizor.fit_transform(xTrain_rand)
    test_rand = vectorizor.transform(xTest['text'])

    model_rand = MultinomialNB()
    model_rand.fit(train_rand, yTrain)
    y_hat_rand = model_rand.predict(test_rand)
    f_rand = f1_score(yTest, y_hat_rand)
    acc_rand = accuracy_score(yTest, y_hat_rand)

    model_complete = MultinomialNB()
    xTrain = vectorizor.fit_transform(xTrain['text'])
    xTest = vectorizor.transform(xTest['text'])
    model_complete.fit(xTrain, yTrain)
    y_hat_complete = model_complete.predict(xTest)
    f_complete = f1_score(yTest, y_hat_complete)
    acc_complete = accuracy_score(yTest, y_hat_complete)

    return f_test, f_rand, f_complete, acc_test, acc_rand, acc_complete

import matplotlib.pyplot as plt
if __name__ == "__main__":
    k = list()

    random_f = list()
    random_a = list()
    emotion_f = list()
    emotion_a = list()

    complete_a = 0
    complete_f = 0

    for i in range(1, 31):
        f_test, f_rand, f_complete, acc_test, acc_rand, acc_complete = main(i)
        complete_a = acc_complete
        complete_f = f_complete
        k.append(i)

        random_f.append(f_rand)
        random_a.append(acc_rand)

        emotion_f.append(f_test)
        emotion_a.append(acc_test)

    plt.plot(k, random_a, "b")
    plt.plot(k, emotion_a, "r")
    plt.xlabel("K")
    plt.ylabel("Accuracy Score")
    plt.title("Accuracy Score vs. K")
    plt.axhline(y=acc_complete, color='g', linestyle='-')
    plt.legend(["Random Scores", "Emotion Scores", "Baseline"], loc="lower right")
    plt.show()

    plt.plot(k, f_rand, "b")
    plt.plot(k, f_test, "r")
    plt.xlabel("K")
    plt.ylabel("F-1 Score")
    plt.title("Accuracy Score vs. K")
    plt.axhline(y=f_complete, color='g', linestyle='-')
    plt.legend(["Random Scores", "Emotion Scores", "Baseline"], loc="lower right")
    plt.show()

    
