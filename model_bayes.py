import pandas as pd
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score


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
    for raw_text in data:
        splits = raw_text.split()
        for word in splits:
            vocabulary.add(word)
    return {word: i for i, word in enumerate(vocabulary)}

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
            result.append(string) if string != "" else result.append(text)
    return result

def build_mappings(corpus):
    result = dict()
    inverse = dict()
    for i, word in enumerate(corpus['token']):
        score = corpus.iloc[i]['score']
        result[word] = score
        inverse[score] = word
    return result, inverse

def main():
    dataset = sys.argv[1] if sys.argv[1] is not None else "emotion"
    k = int(sys.argv[2])
    if dataset == "emotion":
        xTrain, xTest, yTrain, yTest = load_emotion()
        yTrain = yTrain['target']
        yTest = yTest['target']
    elif dataset == "sentiment":
        xTrain, xTest, yTrain, yTest = load_sentiment()
    else:
        print("Dataset must be one of emotion or sentiment.")
        exit(1)

    emotion_corpus = pd.read_csv("emotion-corpus/corpus.csv")
    map_func, inverse = build_mappings(emotion_corpus)
    xTrain_k = select_k(xTrain, map_func, inverse, k)
    xTest_k = select_k(xTest, map_func, inverse, k)

    vocab = build_vocab(emotion_corpus['token'])
    vectorizor = CountVectorizer(input='content', decode_error='ignore', vocabulary=vocab)
    train = vectorizor.fit_transform(xTrain_k)
    test = vectorizor.transform(xTest_k)
    model = MultinomialNB()
    model.fit(train, yTrain)

    y_hat_train = model.predict(train)
    y_hat_test = model.predict(test)

    f_test = f1_score(yTest, y_hat_test)
    f_train = f1_score(yTrain, y_hat_train)

    print("The F1 Score on the training data is: " + str(f_train))
    print("The F1 Score on the testing data is: " + str(f_test))

if __name__ == "__main__":
    main()
