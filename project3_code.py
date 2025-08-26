import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')


df_name = "path/to/the/dataset.csv" 
df = pd.read_csv(df_name)
df_labeled = df.dropna(subset=["popularity"]).copy()
df_unlabeled = df[df["popularity"].isna()].copy()

def tokenize(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text.split()

df_labeled["tokens"] = df_labeled["text"].apply(tokenize)
df_unlabeled["tokens"] = df_unlabeled["text"].apply(tokenize)
print("Preprocessed text.")

w2v_model = Word2Vec(
    sentences=df_labeled["tokens"],
    vector_size=100,
    window=4,
    min_count=2,
    workers=4,
    sg=0 
)
print("Trained Word2Vec")

'''
  Uncomment below to use Word2Vec embeddings
'''
# def embed_song(tokens, model):
#     vectors = [model.wv[word] for word in tokens if word in model.wv]
#     if len(vectors) == 0:
#         return np.zeros(model.vector_size)
#     return np.mean(vectors, axis=0)

# X = np.vstack(df_labeled["tokens"].apply(lambda t: embed_song(t, w2v_model)))
# y = df_labeled["popularity"].astype(int).values
# print("Got DF with not NaN values and converted all ys to int type")
'''
  Uncomment below to use TF-IDF features
'''
# lyrics_as_strings = df_labeled["tokens"].apply(lambda tokens: " ".join(tokens))
# tfidf = TfidfVectorizer(max_features=200, stop_words='english')
# X = tfidf.fit_transform(lyrics_as_strings)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=13)
print("Splitted df")

'''
  Uncomment below to use word2vec embeddings
'''
# # Scale features for SVM
# scaler = StandardScaler(with_mean=False) # with_mean=False for sparse matrix
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

print("Scaled features for SVM.")



xgb_param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 1.0]
}

svm_param_grid = {
    'kernel': ['rbf', 'linear'],
    'C': [0.1, 1, 10],
    'epsilon': [0.01, 0.1, 0.5],
    'gamma': ['scale', 'auto']
}


xgb = XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=6, random_state=13)
grid_xgb = GridSearchCV(
    estimator=xgb,
    param_grid=xgb_param_grid,
    scoring='neg_mean_squared_error',  # or 'neg_mean_absolute_error'
    cv=3,
    n_jobs=-1
)
grid_xgb.fit(X_train, y_train)
print("Best XGB params:", grid_xgb.best_params_)

y_pred_xgb = grid_xgb.predict(X_test)


print('''
# -----------------------
# SVM Classifier
# -----------------------
''')
svm = SVR(kernel="rbf")

grid_svm = GridSearchCV(
    estimator=svm,
    param_grid=svm_param_grid,
    scoring='neg_mean_squared_error',
    cv=3,
    n_jobs=-1
)

grid_svm.fit(X_train, y_train)

print("Best SVM params:", grid_svm.best_params_)
y_pred_svm = grid_svm.predict(X_test)

print('''
-----------------------
SVM Classifier
-----------------------
''')
print("SVM Results:\n", mean_squared_error(y_test, y_pred_svm))
print("SVM Results mae:\n", mean_absolute_error(y_test, y_pred_svm))
print('''
-----------------------
XGBoost Classifier
-----------------------
''')
print("XGBoost Results mse:\n", mean_squared_error(y_test, y_pred_xgb))
print("XGBoost Results mae:\n", mean_absolute_error(y_test, y_pred_xgb))


'''
# -----------------------
# Plotting
# -----------------------
'''

popularities = df_labeled["popularity"]
mean = df_labeled["popularity"].mean()
print(mean)
plt.figure(figsize=(10,6))
sns.histplot(popularities, bins=30, kde=True, color='skyblue')
plt.title("Distribution of Song Popularity")
plt.xlabel("Popularity")
plt.ylabel("Number of Songs")
plt.show()

stop_words = set(stopwords.words('english'))
all_words = [word for tokens in df_labeled['tokens'] for word in tokens if word not in stop_words]
word_freq = Counter(all_words)
for word, _ in word_freq.most_common(10):
    print(word)

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_xgb, alpha=0.5, color='magenta')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()])
plt.xlabel("True Popularity")
plt.ylabel("Predicted Popularity (XGBoost)")
plt.title("XGBoost Predictions vs True Values")
plt.show()

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_svm, alpha=0.5, color='magenta')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()])
plt.xlabel("True Popularity")
plt.ylabel("Predicted Popularity (SVR)")
plt.title("SVR Predictions vs True Values")
plt.show()