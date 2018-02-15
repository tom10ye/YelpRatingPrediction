import gzip
import numpy
import urllib
import pickle
from collections import defaultdict
import scipy.optimize
import random
import matplotlib.pyplot as plt
from math import exp
from math import log

def parseData(fname):
  for l in urllib.urlopen(fname):
    yield eval(l)

def readGz(f):
  for l in gzip.open(f):
    yield eval(l)

print "Reading data..."
data = list(parseData("yelp_training_set_review.json"))
print "done"
##229907
# f = open('training_set_review_pickle.pkl','w')  # wb
# pickle.dump(data, f)
#
# data = pickle.load(open('training_set_review_pickle.pkl', 'r'))   # rb
# print "load complete"

userRatings = defaultdict(dict)
itemRatings = defaultdict(dict)
userBeta = defaultdict(float)
itemBeta = defaultdict(float)

train_data = []
valid_data = []
test_data = []

count = 0
train_length = 100000
valid_length = 100000
for l in data:
   user,item = l['user_id'],l['business_id']
   if count < train_length:
       # rating_train.append(l['rating'])
       train_data.append(l)
       userRatings[user][item] = l['stars']
       itemRatings[item][user] = l['stars']
   elif count < train_length + valid_length:
       # userRatings[user][item] = l['stars']
       # itemRatings[item][user] = l['stars']
       valid_data.append(l)
   else:
       test_data.append(l)
   count += 1
   userBeta[user] = 0.0
   itemBeta[item] = 0.0

train_MSE=[]
valid_MSE=[]
start=44    ##6484693    #5.793-0.811835761018
for i in range(start, start+1):  ###########change to 6484693,6484694
    numda = i*1.0/10    ########## change to 1e6
    alpha = 0
    for iteration in range(1000):
        original_alpha=alpha
        # print"original_alpha = ",original_alpha
        total = 0
        for u in userRatings:
            for i in userRatings[u]:
                temp = userRatings[u][i] - (userBeta[u]+itemBeta[i])
                total += temp
        alpha = total/train_length

        # print "iteration",iteration,"alpha = ",alpha
        diff = abs(original_alpha-alpha)
        if diff < 1e-100:
            break


        for u in userRatings:
            total = 0
            for i in userRatings[u]:
                total += userRatings[u][i] - alpha - itemBeta[i]
            userBeta[u] = total/(numda+len(userRatings[u].keys()))


        for i in itemRatings:
            total = 0
            for u in itemRatings[i]:
                total += itemRatings[i][u] - alpha - userBeta[u]
            itemBeta[i] = total/(numda+len(itemRatings[i].keys()))

    total = 0
    for l in train_data:
        total += (l['stars'] - (alpha + userBeta[l['user_id']] + itemBeta[l['business_id']])) ** 2
    mse = total / len(train_data)
    train_MSE.append(mse)

    total = 0
    cannotFind = 100
    for l in valid_data:
        if (userBeta[l['user_id']] == 0.0 or itemBeta[l['business_id']] == 0.0):
            cannotFind += 1
        total += (l['stars'] - (alpha + userBeta[l['user_id']] + itemBeta[l['business_id']])) ** 2
    mse = total / len(valid_data)
    valid_MSE.append(mse)
    print"valid_cannotFind",cannotFind
    print"numda=",numda,"valid_MSE=", '%.20f' % mse

index = range(1,21,1)
# plt.plot(index, train_MSE, label="train_MSE")
# plt.plot(index, valid_MSE, label="valid_MSE")
plt.xlabel("numda")
plt.ylabel("MSE")
plt.title("trainSet_validSet_MSE versus numda")
plt.legend()
# plt.show()

total = 0
cannotFind = 20
for l in test_data:
    if (userBeta[l['user_id']] == 0.0 or itemBeta[l['business_id']] == 0.0):
        cannotFind += 1
    total += (l['stars'] - (alpha + userBeta[l['user_id']] + itemBeta[l['business_id']])) ** 2
mse = total / len(test_data)
print"numda=", numda, "test_MSE=", '%.20f' % mse

print cannotFind
# print"Q7:"
# print"min userBeta =", min(userBeta, key=userBeta.get)
# print"max userBeta =", max(userBeta, key=userBeta.get)
#
# print"min itemBeta =", min(itemBeta, key=itemBeta.get)
# print"max itemBeta =", max(itemBeta, key=itemBeta.get)


