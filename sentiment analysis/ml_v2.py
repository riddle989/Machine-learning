
# In[1]:

import pandas as pd
from io import StringIO
df = pd.read_csv('C:/Users/NoT/Desktop/ML/Project/sentiment analysis/sentiment.csv')
df.head()

# In[2]:

col = ['data', 'title' ]
df = df[col]
df.columns = ['data', 'title']
df['category_id'] = df['title'].factorize()[0]
category_id_df = df[['title', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'title']].values)
df.head()


# In[9]:
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.data).toarray()
labels = df.category_id
features.shape


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



df['normalized_data'] = df.data.apply(normalizer)

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
        'Like (ভাল)': 0,
        'Smiley (স্মাইলি)': 1,
        'HaHa(হা হা)' : 2,
        'Sad (দু: খিত)': 3,
        'Skip ( বোঝতে পারছি না )': 4,
        'Love(ভালবাসা)': 5,
        'WOW(কি দারুন)': 6,
        'Blush(গোলাপী আভা)': 7,
        'Consciousness (চেতনাবাদ)': 8,
        'Rocking (আন্দোলিত হত্তয়া)': 9,
        'Bad (খারাপ)': 10,
        'Angry (রাগান্বিত)': 11,
        'Fail (ব্যর্থ)': 12,
        'Provocative (উস্কানিমুলক)': 13,
        'Shocking (অতিশয় বেদনাদায়ক)': 14,
        'Protestant (প্রতিবাদমূলক)': 15,
        'Evil (জঘন্য)': 16,
        'Skeptical (সন্দেহপ্রবণ)': 17,

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
