import pandas as pd
import sys

def process_emotion():
    
    return None

def process_sentiment():
    return None

def main():
    dataset = sys.argv[1] if sys.argv[1] is not None else "emotion"
    train, test = None
    if dataset == "emotion":
        train, test = process_emotion()
    elif dataset == "sentiment":
        train, test = process_sentiment()
    else:
        print("Dataset must be one of emotion or sentiment.")
        exit(1)

if __name__ == "__main__":
    main()