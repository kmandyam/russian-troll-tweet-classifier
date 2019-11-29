import pandas as pd

filename = "data/raw/australian-election-2019-tweets/auspol2019.csv"
df = pd.read_csv(filename)

def assign_label(row):
    return 0
df['label'] = df.apply(lambda row: assign_label(row), axis=1)

df = df[['full_text', 'label']]
df.columns = ['tweet', 'label']

df.to_csv("data/cleaned/aus_election_tweets.csv", index=False)