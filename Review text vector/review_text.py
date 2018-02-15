import numpy as np
import urllib
import scipy.optimize
import random
from collections import defaultdict
import nltk
import string
from nltk.stem.porter import *
from sklearn import linear_model
import pickle
from sklearn.metrics import mean_squared_error


##################################################
# reading the data                               #
##################################################

def parseData(fname):
    for l in urllib.urlopen(fname):
        yield eval(l)


### Just the reviews

print "Reading data..."
data = list(parseData("yelp_training_set_review.json"))
print "done"

train_set = data[:200000]
test_set = data[200000:229906]

##################################################
# information upon unigrams and bigrams          #
##################################################

### Ignore capitalization and remove punctuation
wordCount = defaultdict(int)
punctuation = set(string.punctuation)
bigramCounter = defaultdict(int)
unigramCounter = defaultdict(int)
reviews = []

for d in train_set:
    r = ''.join([c for c in d['text'].lower() if not c in punctuation])
    reviews.append(r)
    allWords = r.split()
    for w in r.split():
        unigramCounter[w] += 1
    for i in xrange(1, len(allWords)):
        word1 = allWords[i - 1]
        word2 = allWords[i]
        bigramCounter[word1 + " " + word2] += 1

print 'the number of unique bigrams = ', len(bigramCounter)
print 'the number of unique unigrams = ', len(unigramCounter)

unigramCount = [(unigramCounter[w], w) for w in unigramCounter]
unigramCount.sort()
unigramCount.reverse()

print '20 most frequently occurring unigrams =', unigramCount[:20]


bigramCount = [(bigramCounter[w], w) for w in bigramCounter]
bigramCount.sort()
bigramCount.reverse()

print '20 most frequently occurring bigrams =', bigramCount[:20]

##################################################
# most 1000 bigrams vector                       #
##################################################

bigramWords_sample = [x[1] for x in bigramCount[:1000]]
bigramWordId_sample = dict(zip(bigramWords_sample, range(len(bigramWords_sample))))


def feature(datum):
    feat = [0] * len(bigramWords_sample)
    r = ''.join([c for c in datum['text'].lower() if not c in punctuation])
    allWords = r.split()
    bigramWordsData = []
    for i in xrange(1, len(allWords)):
        word1 = allWords[i - 1]
        word2 = allWords[i]
        bigramWordsData.append(word1 + " " + word2)
    for w in bigramWordsData:
        if w in bigramWords_sample:
            feat[bigramWordId_sample[w]] += 1
    feat.append(1)  # offset
    return feat


X_train = [feature(d) for d in train_set]
y_train = [d['stars'] for d in train_set]
X_test = [feature(d) for d in test_set]
y_test = [d['stars'] for d in test_set]

# With regularization
clf = linear_model.Ridge(1.0, fit_intercept=False)
clf.fit(X_train, y_train)
theta = clf.coef_
predictions = clf.predict(X_test)
print 'new predictor mean_squared_error using 1000 most common bigrams =', mean_squared_error(predictions, y_test)

##################################################
#  most 2000 unigram and bigrams vector          #
##################################################

unigramCount = [(unigramCounter[w], w) for w in unigramCounter]
bigramCount = [(bigramCounter[w], w) for w in bigramCounter]
countBothWord = unigramCount + bigramCount
countBothWord.sort()
countBothWord.reverse()
bothWords = [x[1] for x in countBothWord[:2000]]
both_key_freq = dict(zip(bothWords, range(len(bothWords))))
both_freq_key = dict(zip(range(len(bothWords)), bothWords))


def feature(datum):
    feat = [0] * len(bothWords)
    r = ''.join([c for c in datum['text'].lower() if not c in punctuation])
    words_temp = r.split()
    bigramWordsData = []
    for i in xrange(1, len(words_temp)):
        word1 = words_temp[i - 1]
        word2 = words_temp[i]
        bigramWordsData.append(word1 + " " + word2)
    for w in words_temp:
        if w in bothWords:
            feat[both_key_freq[w]] += 1
    for w in bigramWordsData:
        if w in bothWords:
            feat[both_key_freq[w]] += 1
    feat.append(1)  # offset
    return feat


X_train = [feature(d) for d in train_set]
y_train = [d['stars'] for d in train_set]
X_test = [feature(d) for d in test_set]
y_test = [d['stars'] for d in test_set]

# With regularization
clf = linear_model.Ridge(1.0, fit_intercept=False)
clf.fit(X_train, y_train)
theta = clf.coef_
predictions = clf.predict(X_test)
print 'new predictor mean_squared_error using both 2000 most common unigram and bigrams =', mean_squared_error(predictions, y_test)

##################################################
#  the idea of visulization of ratings           #
##################################################

weightCounter = []
for i in xrange(len(theta) - 1):
    weightCounter.append((theta[i], i))
weightCounter.sort()
weightCounter.reverse()

print 'most positive 20 statement(unigrams and bigrams) in sense of stars rating:'
for i in xrange(20):
    print both_freq_key[weightCounter[i][1]]

weightCounter.reverse()

print 'most negative 20 statement(unigrams and bigrams) in sense of stars rating:'
for i in xrange(20):
    print both_freq_key[weightCounter[i][1]]

##################################################
#  with 2000-dimensional tf-idf representations  #
##################################################

temp = defaultdict(list)
freq_temp = defaultdict(int)

words = []
for r in reviews:
    words += r.split()
uniqueWords = set(words)
for i in xrange(len(reviews)):
    r = reviews[i]
    for w in r.split():
        if len(temp[w]) == 0 or i != temp[w][-1]:
            temp[w].append(i)

for w in uniqueWords:
    freq_temp[w] = len(temp[w])

def calTfIdf(word, r):
    tf = 0
    r_temp = r.split()
    for w in r_temp:
        if word == w:
            tf += 1
    N = len(reviews)

    idf = np.log10(N * 1.0 / freq_temp[word])
    tfidf = tf * idf
    return tfidf


counts = [(unigramCounter[w], w) for w in unigramCounter]
counts = countBothWord
counts.sort()
counts.reverse()

words = [x[1] for x in counts[:2000]]
wordId = dict(zip(words, range(len(words))))
wordSet = set(words)


def feature(datum):
    feat = [0] * len(words)
    r = ''.join([c for c in datum['text'].lower() if not c in punctuation])
    for w in r.split():
        if w in words:


            feat[wordId[w]] = calTfIdf(w, r)

    feat.append(1)  # offset
    return feat


X_train = [feature(d) for d in train_set]
y_train = [d['stars'] for d in train_set]
X_test = [feature(d) for d in test_set]
y_test = [d['stars'] for d in test_set]

# With regularization
clf = linear_model.Ridge(1.0, fit_intercept=False)
clf.fit(X_train, y_train)
theta = clf.coef_
predictions = clf.predict(X_test)

print 'the new model with 1000-dimensional unigrams and 1000-dimensional bigrams tf-idf representations MSE =', mean_squared_error(predictions, y_test)
