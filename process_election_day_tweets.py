import pandas as pd

filename = "data/raw/election-day-tweets/election_day_tweets.csv"
df = pd.read_csv(filename)
df = df[df.lang == 'en']

def assign_label(row):
    return 0
df['label'] = df.apply(lambda row: assign_label(row), axis=1)

df = df[['text', 'label']]
df.columns = ['tweet', 'label']

df.to_csv("data/cleaned/election_day_tweets.csv", index=False)