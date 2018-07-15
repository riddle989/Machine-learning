from sklearn import metrics
import glob
import errno
import codecs
path1 = 'C:/Users/NoT/Desktop/ML/Project/Stylogenetics/stylogenetics/Hasan Mahbub/*.doc'
path2 = 'C:/Users/NoT/Desktop/ML/Project/Stylogenetics/stylogenetics/MZI/MZI/*.doc'
path3 = 'C:/Users/NoT/Desktop/ML/Project/Stylogenetics/stylogenetics/Nir Shondhani/Nir Shondhani/*.doc'
labels, texts = [], []
val_x, val_y = [],[]
files = glob.glob(path1)

for name in files:
    try:
        with codecs.open(name, 'r', encoding='utf-8') as f:
            str = f.read()
            # str = re.sub(' +', ' ', str)
            str = " ".join(str.split())
            labels.append("hm")
            texts.append(str)


    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise

files = glob.glob(path2)
for name in files:
    try:
        with codecs.open(name, 'r', encoding='utf-8') as f:
            str = f.read()
            #  str = re.sub(' +', ' ', str)
            str = " ".join(str.split())

            labels.append("mzi")
            texts.append(str)

    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise
files = glob.glob(path3)
for name in files:
    try:
        with codecs.open(name, 'r', encoding='utf-8') as f:
            str = f.read()
            #  str = re.sub(' +', ' ', str)
            str = " ".join(str.split())

            labels.append("ns")
            texts.append(str)

    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise


import pandas as pd
import numpy as np
df = pd.DataFrame({'texts':texts, 'labels': labels})
df.head()


df['category_id'] = df['labels'].factorize()[0]

category_id_df = df[['labels', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'labels']].values)


import re, nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()

def normalizer(str):

    tokens = nltk.word_tokenize(str)
    lower_case = [l.lower() for l in tokens]
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))
    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]
    return lemmas



pd.set_option('display.max_colwidth', -1) # Setting this so we can see the full content of cells
df['normalized_data'] = df.texts.apply(normalizer)


import collections
def count_words(input):
    cnt = collections.Counter()
    for row in input:
        for word in row:
            cnt[word] += 1
    return cnt


import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer(ngram_range=(1,2))

vectorized_data = count_vectorizer.fit_transform(df.texts)
indexed_data = hstack((np.array(range(0,vectorized_data.shape[0]))[:,None], vectorized_data))


def sentiment2target(sentiment):
    return {
        'hm': 0,
        'mzi': 1,
        'ns': 2,

    }[sentiment]



targets = df.labels.apply(sentiment2target)

from sklearn.model_selection import train_test_split
data_train, data_test, targets_train, targets_test = train_test_split(indexed_data, targets, test_size=0.2, random_state=0)
data_train_index = data_train[:,0]
data_train = data_train[:,1:]
data_test_index = data_test[:,0]
data_test = data_test[:,1:]


#svm
print("svm")
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
clf = OneVsRestClassifier(svm.SVC(gamma=0.01, C=100., probability=True, class_weight='balanced', kernel='linear'))
clf_output = clf.fit(data_train, targets_train)
print(clf.score(data_test, targets_test))


#Naive bayes
print("Naive bayes")
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(data_train, targets_train)
predictions = nb.predict(data_test)
print(metrics.accuracy_score(targets_test,predictions))


#KNN
print("KNN")
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors = 3)
KNN.fit(data_train, targets_train)
predictions = KNN.predict(data_test)
print(metrics.accuracy_score(targets_test,predictions))

#Decision Tree
print("Decision Tree")
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(data_train, targets_train)
predictions=clf.predict(data_test)
print(metrics.accuracy_score(targets_test,predictions))
