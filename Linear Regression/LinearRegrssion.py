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
import json as js


def parseData(fname):
    for l in urllib.urlopen(fname):
        yield eval(l)


print "Reading data..."
dataReview = list(parseData("yelp_training_set_review.json"))
dataBusiness = list(parseData("yelp_training_set_business.json"))
dataUser = list(parseData("yelp_training_set_user.json"))
dataCheck = list(parseData("yelp_training_set_checkin.json"))
# constructing train set and test set
dataReviewTrain = dataReview[:100000]
dataReviewTest = dataReview[200000:220000]
dataReviewValid = dataReview[100000:200000]
print "done"
print  len(dataReviewValid)
print  len(dataReviewTest)
# construct dictionary for checking by business id
openBusiness = defaultdict(bool)
reviewCountBusiness = defaultdict(int)
starsBusiness = defaultdict(float)
longitudeBusiness = defaultdict(float)
latitudeBusiness = defaultdict(float)
# construct dictionary for checking by user id
voteUser1 = defaultdict(int)
voteUser2 = defaultdict(int)
voteUser3 = defaultdict(int)
reviewCountUser = defaultdict(int)
starsUser = defaultdict(float)
# store data in dict
sum = 0
count = 0
for d in dataBusiness:
    id = d['business_id']
    openBusiness[id] = 0
    openBusiness[id] = d['open']
    reviewCountBusiness[id] = 20.192857762
    reviewCountBusiness[id] = d['review_count']
    starsBusiness[id] = 3.67452543989
    starsBusiness[id] = d['stars']
    longitudeBusiness[id] = -111.98889452
    longitudeBusiness[id] = d['longitude']
    latitudeBusiness[id] = 33.4878330886
    latitudeBusiness[id] = d['latitude']
for d in dataUser:
    uid = d['user_id']
    voteUser1[uid] = 0
    voteUser1[uid] = d['votes']['funny']
    voteUser2[uid] = 0
    voteUser2[uid] = d['votes']['useful']
    voteUser3[uid] = 0
    voteUser3[uid] = d['votes']['cool']
    reviewCountUser[uid] = 0
    reviewCountUser[uid] = d['review_count']
    starsUser[uid] = 0
    starsUser[uid] = d['average_stars']

    sum += d['average_stars']
    count += 1
avg = sum * 1.0 / count
print ("avg = " + str(avg))
def feature(datum):
    user_id = datum['user_id']
    business_id = datum['business_id']
    openInfo = openBusiness[business_id]
    reviewCountInfo = reviewCountBusiness[business_id]
    starsBusinessInfo = starsBusiness[business_id]
    # longitudeInfo = longitudeBusiness[business_id]
    # latitudeInfo = latitudeBusiness[business_id]
    reviewCountUserInfo = reviewCountUser[user_id]
    voteUser1Info = voteUser1[user_id]
    voteUser2Info = voteUser2[user_id]
    voteUser3Info = voteUser3[user_id]
    starsUserInfo = starsUser[user_id]
    feat = [1]
    feat.append(datum['votes']['funny'])
    feat.append(datum['votes']['useful'])
    feat.append(datum['votes']['cool'])
    feat.append(openInfo)
    feat.append(reviewCountInfo)
    feat.append(starsBusinessInfo)
    # feat.append(longitudeInfo)
    # feat.append(latitudeInfo)
    feat.append(reviewCountUserInfo)
    feat.append(voteUser1Info)
    feat.append(voteUser2Info)
    feat.append(voteUser3Info)
    feat.append(starsUserInfo)
    return feat
X = [feature(d) for d in dataReviewTrain]
y = [d['stars'] for d in dataReviewTrain]
# print X[:3]
def calMSE(x, y):
    sum = 0
    for i in range(0,len(y)):
     sum += (y[i]-x[i])**2
    mse = sum*1.0/len(y)
    return mse
#With regularization
clf = linear_model.Ridge(0.01, fit_intercept=False)
clf.fit(X, y)
theta = clf.coef_
predictions = clf.predict(X)
print ("MSE for train set  = " + str(calMSE(predictions, y)))
print ("theta = " + str(theta))

# compute MSE on test set
def featureTest(d):
    business_id = d['business_id']
    user_id = d['user_id']
    openInfo = openBusiness[business_id]
    reviewCountInfo = reviewCountBusiness[business_id]
    starsBusinessInfo = starsBusiness[business_id]
    # longitudeInfo = longitudeBusiness[business_id]
    # latitudeInfo = latitudeBusiness[business_id]
    reviewCountUserInfo = reviewCountUser[user_id]
    voteUser1Info = voteUser1[user_id]
    voteUser2Info = voteUser2[user_id]
    voteUser3Info = voteUser3[user_id]
    starsUserInfo = starsUser[user_id]
    feat = [1]
    feat.append(d['votes']['funny'])
    feat.append(d['votes']['useful'])
    feat.append(d['votes']['cool'])
    feat.append(openInfo)
    feat.append(reviewCountInfo)
    feat.append(starsBusinessInfo)
    # feat.append(longitudeInfo)
    # feat.append(latitudeInfo)
    feat.append(reviewCountUserInfo)
    feat.append(voteUser1Info)
    feat.append(voteUser2Info)
    feat.append(voteUser3Info)
    feat.append(starsUserInfo)
    return feat
X_test = [feature(d) for d in dataReviewTest]
y_test = [d['stars'] for d in dataReviewTest]
X_valid = [feature(d) for d in dataReviewValid]
y_valid = [d['stars'] for d in dataReviewValid]
X_test = np.matrix(X_test)
y_test = np.matrix(y_test)
theta = np.matrix(theta)
X_valid = np.matrix(X_valid)
y_valid = np.matrix(y_valid)

def mse(X,y,theta):
  sum=0
  M = y.T - X * (theta.T)
  for i in range(y.shape[1]):
    sum+=M[i]**2
  return sum/y.shape[1]
print ("MSE for test set  = " + str(mse(X_test, y_test,theta)))
print ("MSE for valid set  = " + str(mse(X_valid, y_valid,theta)))



