import pandas as pd
import os.path

def main():
    if not os.path.isfile("emotion-corpus/corpus.csv"):
        df1 = pd.read_csv("emotion-corpus/test.csv")
        df2 = pd.read_csv("emotion-corpus/training.csv")
        df3 = pd.read_csv("emotion-corpus/validation.csv")
        pd.concat([df1, df2, df3]).to_csv("emotion-corpus/emotion-phrases.csv", index=False)
        
if __name__ == "__main__":
    main()
