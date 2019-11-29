import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

train_df = pd.read_csv("data/data_splits/train.csv")
val_df = pd.read_csv("data/data_splits/validation.csv")
test_df = pd.read_csv("data/data_splits/test.csv")

dfs = [train_df, val_df, test_df]

for df in dfs:
    df['tweet'] = df['tweet'].astype('str')
    df['label'] = df['label'].astype('str')

total_df = pd.concat(dfs)

print("unique labels: ", total_df['label'].unique())
assert len(total_df['label'].unique()) == 2

print("all data used: ", len(total_df))

X_train = train_df['tweet']
X_test = test_df['tweet']

y_train = train_df['label']
y_test = test_df['label']

print("Train Size: ", len(X_train))
print("Test Size: ", len(X_test))

start_time = time.time()
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2',
                        encoding='latin-1', ngram_range=(1, 2),
                        stop_words='english')

features = tfidf.fit_transform(total_df.tweet).toarray()
labels = total_df.label
print(features.shape)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = LogisticRegression(random_state=0).fit(X_train_tfidf, y_train)
print("--- %s seconds ---" % (time.time() - start_time))
X_test_counts = count_vect.transform(X_test)
y_pred = clf.predict(X_test_counts)

print(metrics.classification_report(y_test, y_pred, target_names=total_df['label'].unique()))
