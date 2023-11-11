import pandas as pd
import os.path
import spacy
import numpy as np

def dict_add(dictionary, key, value):
    if dictionary.get(key) is None:
        dictionary[key] = value
    else:
        dictionary[key] += value

def z_normalize(dictionary, mu, sigma):
    for key in dictionary:
        x = dictionary[key]
        x -= mu
        z = x / sigma
        dictionary[key] = z

def main():
    if not os.path.isfile("emotion-corpus/emotion-phrases.csv"):
        df1 = pd.read_csv("emotion-corpus/test.csv")
        df2 = pd.read_csv("emotion-corpus/training.csv")
        df3 = pd.read_csv("emotion-corpus/validation.csv")
        pd.concat([df1, df2, df3]).to_csv("emotion-corpus/emotion-phrases.csv", index=False)

    wordmap = dict()

    dataloader = pd.read_csv("emotion-corpus/emotion-phrases.csv")
    nlp = spacy.load("en_core_web_md")
    for index, data in dataloader.iterrows():
        phrase = data['text']
        label = data['label']
        jitter = np.random.uniform(.95, 1)
        score = 0
        if label == 0 or label == 3:
            score = -1 * jitter
        elif label == 1 or label == 2:
            score = 1 * jitter
        else:
            continue
        
        tokens = nlp(phrase)
        for token in tokens:
            if not token.is_stop:
                dict_add(wordmap, str(token), score)

    vals = list(wordmap.values())
    mean = np.mean(vals)
    stdev = np.std(vals)
    z_normalize(wordmap, mean, stdev)

    outdf = pd.DataFrame(data={'token': wordmap.keys(), 'score': wordmap.values()},)
    outdf.to_csv('emotion-corpus/corpus.csv', index=False)
    
        
if __name__ == "__main__":
    main()
