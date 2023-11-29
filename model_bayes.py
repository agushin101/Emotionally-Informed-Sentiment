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
                if i >= k:
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
            selects = np.random.choice(len(splits), k, replace=False).tolist()
            choices = [splits[i] for i in selects]
            for i, word in enumerate(choices):
                string += word
                if i < k:
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


def main():
    k = int(sys.argv[1])
    xTrain, xTest, yTrain, yTest = load_emotion()
    yTrain = yTrain['target']
    yTest = yTest['target']

    ###Handle the non-random selects first

    emotion_corpus = pd.read_csv("emotion-corpus/corpus.csv")
    map_func, inverse = build_mappings(emotion_corpus)
    xTrain_k = select_k(xTrain, map_func, inverse, k)
    xTest_k = select_k(xTest, map_func, inverse, k)

    vectorizor = CountVectorizer(input='content', decode_error='ignore')
    train = vectorizor.fit_transform(xTrain_k)
    test = vectorizor.transform(xTest['text'])
    model = MultinomialNB()
    model.fit(train, yTrain)

    y_hat_train = model.predict(train)
    y_hat_test = model.predict(test)

    f_test = f1_score(yTest, y_hat_test)
    f_train = f1_score(yTrain, y_hat_train)
    acc_test = accuracy_score(yTest, y_hat_test)
    acc_train = accuracy_score(yTrain, y_hat_train)

    print("The Accuracy Score on the training data is: " + str(acc_train))
    print("The F1 Score on the training data is: " + str(f_train))
    print("The Accuracy Score on the testing data is: " + str(acc_test))
    print("The F1 Score on the testing data is: " + str(f_test))

    ###Now, handle the random selects
    xTest_rand = select_k_random(xTest, k)
    testRand = vectorizor.transform(xTest_rand)

    y_hat_rand = model.predict(testRand)
    f_rand = f1_score(yTest, y_hat_rand)
    acc_rand = accuracy_score(yTest, y_hat_rand)

    print("The Accuracy Score on the randomized testing data is: " + str(acc_rand))
    print("The F1 Score on the randomized testing data is: " + str(f_rand))

if __name__ == "__main__":
    main()
