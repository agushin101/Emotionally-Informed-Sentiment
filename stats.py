import pandas as pd
import numpy as np

def main():
    loader = pd.read_csv("data/emotion/xTrain.csv")
    lens = list()
    for i, row in loader.iterrows():
        counts = row['text'].split()
        lens.append(len(counts))
    avg = np.mean(lens)
    large = max(lens)
    med = np.median(lens)

    print("Statistics:")
    print("Average: " + str(avg))
    print("Median: " + str(med))
    print("Max: " + str(large))

if __name__ == "__main__":
    main()
