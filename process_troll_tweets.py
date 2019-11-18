import os
import pandas as pd

directory = "data/raw/troll_tweets"
frames = []
for filename in os.listdir(directory):
    print()
    if filename.endswith(".csv"):
        print("processing " + os.path.join(directory, filename))
        df = pd.read_csv(os.path.join(directory, filename))
        sample_size = min(53845, df.shape[0])
        df = df.sample(sample_size)
        frames.append(df)

df = pd.concat(frames)
df.to_csv("data/cleaned/troll_tweets.csv", index=True)



