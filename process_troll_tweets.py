import os
import pandas as pd

directory = "data/raw/troll_tweets"
frames = []
for filename in os.listdir(directory):
    print()
    if filename.endswith(".csv"):
        print("processing " + os.path.join(directory, filename))
        df = pd.read_csv(os.path.join(directory, filename))
        df = df[df.language == 'English']
        sample_size = min(53845, df.shape[0])
        df = df.sample(sample_size)
        frames.append(df)

df = pd.concat(frames)

def assign_label(row):
    return 1

df['label'] = df.apply(lambda row: assign_label(row), axis=1)

del df['external_author_id']
del df['author']
del df['region']
del df['following']
del df['followers']
del df['updates']
del df['post_type']
del df['account_type']

del df['retweet']
del df['account_category']
del df['new_june_2018']
del df['alt_external_id']
del df['tweet_id']
del df['article_url']

del df['tco1_step1']
del df['tco2_step1']
del df['tco3_step1']

df.to_csv("data/cleaned/troll_tweets.csv", index=True)