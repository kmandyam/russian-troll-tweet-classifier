import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


troll_csv = "data/cleaned/troll_tweets.csv"
pol_csv = "data/cleaned/dem_rep_tweets.csv"

troll_df = pd.read_csv(troll_csv)
pol_df = pd.read_csv(pol_csv)

troll_df = troll_df.sample(n=10000)
pol_df = pol_df.sample(n=10000)

df = pd.concat([troll_df, pol_df])
df['tweet'] = df['tweet'].astype('str')
df['label'] = df['label'].astype('str')

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2',
                        encoding='latin-1', ngram_range=(1, 2),
                        stop_words='english')

features = tfidf.fit_transform(df.tweet).toarray()
labels = df.label
print(features.shape)

X_train, X_test, y_train, y_test = train_test_split(df['tweet'], df['label'], random_state=0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train)

X_test_counts = count_vect.transform(X_test)
y_pred = clf.predict(X_test_counts)

print(metrics.classification_report(y_test, y_pred, target_names=df['label'].unique()))
