import pandas as pd

def prep_emotion():
    train = "emotion-corpus/training.csv"
    test = "emotion-corpus/test.csv"

    loadTrain = pd.read_csv(train)
    loadTest = pd.read_csv(test)

    binary_targets_train = []
    binary_targets_test = []
    text_train = []
    text_test = []

    for _, sample in loadTrain.iterrows():
        target = sample['label']
        text = sample['text']
        if target == 0 or target == 3:
            binary_targets_train.append(0)
            text_train.append(text)
        elif target == 1 or target == 2:
            binary_targets_train.append(1)
            text_train.append(text)
        else:
            continue
    for _, sample in loadTest.iterrows():
        target = sample['label']
        text = sample['text']
        if target == 0 or target == 3:
            binary_targets_test.append(0)
            text_test.append(text)
        elif target == 1 or target == 2:
            binary_targets_test.append(1)
            text_test.append(text)
        else:
            continue

    out = "data/emotion/"
    pd.Series(binary_targets_train, name='target').to_csv(out + "yTrain.csv", index=False)
    pd.Series(binary_targets_test, name='target').to_csv(out + "yTest.csv", index=False)
    pd.Series(text_test, name='text').to_csv(out + "xTest.csv", index=False)
    pd.Series(text_train, name='text').to_csv(out + "xTrain.csv", index=False)

def prep_sentiment():
    return None

def main():
    prep_emotion()
    prep_sentiment()

if __name__ == "__main__":
    main()