import pandas as pd

troll_csv = "data/cleaned/troll_tweets.csv"
dem_rep_csv = "data/cleaned/dem_rep_tweets.csv"
elec_day_csv = "data/cleaned/election_day_tweets.csv"
aus_elec_csv = "data/cleaned/aus_election_tweets.csv"


troll_df = pd.read_csv(troll_csv)
dem_rep_df = pd.read_csv(dem_rep_csv)
elec_day_df = pd.read_csv(elec_day_csv)
aus_elec_df = pd.read_csv(aus_elec_csv)

print("troll tweets: ", len(troll_df))
print("dem rep tweets: ", len(dem_rep_df))
print("elec day tweets: ", len(elec_day_df))
print("aus elec tweets: ", len(aus_elec_df))

df = pd.concat([troll_df, dem_rep_df, elec_day_df, aus_elec_df])

df['tweet'] = df['tweet'].astype('str')
df['label'] = df['label'].astype('str')
df = df[df.label != 'nan']

all_data_used = len(df)
print("all data used: ", len(df))

def data_splits(df):
    # shuffle the df
    df = df.sample(frac=1)
    num_training = int(len(df) * 0.7)
    num_validation = int(len(df) * 0.15)
    num_test = int(len(df) * 0.15)
    train_df = df[:num_training]
    validate_df = df[num_training:num_training+num_validation]
    test_df = df[num_training+num_validation:]

    return train_df, validate_df, test_df

tr, val, te = data_splits(df)

print("Training Data: ", len(tr))
print("Validation Data: ", len(val))
print("Testing Data: ", len(te))

print("ALL DATA: ", all_data_used)
print("DATA IN DF: ", len(tr) + len(val) + len(te))
assert all_data_used == len(tr) + len(val) + len(te)

tr.to_csv("data/data_splits/train.csv", index=False)
val.to_csv("data/data_splits/validation.csv", index=False)
te.to_csv("data/data_splits/test.csv", index=False)
