import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


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

troll_df = troll_df.sample(n=10000)
# dem_rep_df = dem_rep_df.sample(n=10000)
#
# df = pd.concat([troll_df, dem_rep_df])
# df['tweet'] = df['tweet'].astype('str')
# df['label'] = df['label'].astype('str')
#
# tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2',
#                         encoding='latin-1', ngram_range=(1, 2),
#                         stop_words='english')
#
# features = tfidf.fit_transform(df.tweet).toarray()
# labels = df.label
# print(features.shape)
#
# X_train, X_test, y_train, y_test = train_test_split(df['tweet'], df['label'], random_state=0)
# count_vect = CountVectorizer()
# X_train_counts = count_vect.fit_transform(X_train)
# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# clf = MultinomialNB().fit(X_train_tfidf, y_train)
#
# X_test_counts = count_vect.transform(X_test)
# y_pred = clf.predict(X_test_counts)
#
# print(metrics.classification_report(y_test, y_pred, target_names=df['label'].unique()))
