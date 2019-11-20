import pandas as pd

filename = "data/raw/democratvsrepublicantweets/ExtractedTweets.csv"
df = pd.read_csv(filename)
del df['Party']
del df['Handle']

def assign_label(row):
    return 0


df['label'] = df.apply(lambda row: assign_label(row), axis=1)
df.columns = ['tweet', 'label']
df.to_csv("data/cleaned/dem_rep_tweets.csv", index=False)



