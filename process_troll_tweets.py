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
        sample_size = min(50000, df.shape[0])
        df = df.sample(sample_size)
        frames.append(df)

df = pd.concat(frames)

def assign_label(row):
    return 1

df['label'] = df.apply(lambda row: assign_label(row), axis=1)
df = df[['content', 'label']]
df.columns = ['tweet', 'label']
df.to_csv("data/cleaned/troll_tweets.csv", index=False)