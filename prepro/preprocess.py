#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 16:28:55 2016

@author: Chaofan
"""

import pandas as pd
import numpy as np
import datetime
import gzip
import json
from  sklearn.decomposition import PCA
from myria import *

def parse(path):
  f = gzip.open(path, 'r')
  for l in f:
    yield eval(l)

review_data = parse('kcore_5.json.gz')
productID = []
userID = []
score = []
reviewTime = []
rowCount = 0

while True:
    try:
        entry = next(review_data)
        productID.append(entry['asin'])
        userID.append(entry['reviewerID'])
        score.append(entry['overall'])
        reviewTime.append(entry['reviewTime'])
        rowCount += 1
        if rowCount % 1000000 == 0:
            print 'Already read %s observations' % rowCount
    except StopIteration, e:
        print 'Read %s observations in total' % rowCount
        entry_list = pd.DataFrame({'productID': productID,
                                   'userID': userID,
                                   'score': score,
                                   'reviewTime': reviewTime})
        filename = 'review_data.csv'
        entry_list.to_csv(filename, index=False)
        print 'Save the data in the file %s' % filename
        break

entry_list = pd.read_csv('review_data.csv')

def filterReviewsByField(reviews, field, minNumReviews):
    reviewsCountByField = reviews.groupby(field).size()
    fieldIDWithNumReviewsPlus = reviewsCountByField[reviewsCountByField >= minNumReviews].index
    #print 'The number of qualified %s: ' % field, fieldIDWithNumReviewsPlus.shape[0]
    if len(fieldIDWithNumReviewsPlus) == 0:
        print 'The filtered reviews have become empty'
        return None
    else:
        return reviews[reviews[field].isin(fieldIDWithNumReviewsPlus)]

def checkField(reviews, field, minNumReviews):
    return np.mean(reviews.groupby(field).size() >= minNumReviews) == 1

def filterReviews(reviews, minItemNumReviews, minUserNumReviews):
    filteredReviews = filterReviewsByField(reviews, 'productID', minItemNumReviews)
    if filteredReviews is None:
        return None
    if checkField(filteredReviews, 'userID', minUserNumReviews):
        return filteredReviews
    
    filteredReviews = filterReviewsByField(filteredReviews, 'userID', minUserNumReviews)
    if filteredReviews is None:
        return None
    if checkField(filteredReviews, 'productID', minItemNumReviews):
        return filteredReviews
    else:
        return filterReviews(filteredReviews, minItemNumReviews, minUserNumReviews)
        
def filteredReviewsInfo(reviews, minItemNumReviews, minUserNumReviews):    
    t1 = datetime.datetime.now()
    filteredReviews = filterReviews(reviews, minItemNumReviews, minUserNumReviews)
    print 'Mininum num of reviews in each item: ', minItemNumReviews
    print 'Mininum num of reviews in each user: ', minUserNumReviews
    print 'Dimension of filteredReviews: ', filteredReviews.shape if filteredReviews is not None else '(0, 4)'
    print 'Num of unique Users: ', filteredReviews['userID'].unique().shape[0]
    print 'Num of unique Product: ', filteredReviews['productID'].unique().shape[0]
    t2 = datetime.datetime.now()
    print 'Time elapsed: ', t2 - t1
    return filteredReviews

allReviewData = filteredReviewsInfo(entry_list, 100, 10)
smallReviewData = filteredReviewsInfo(allReviewData, 150, 15)

smallReviewData['whetherSmall'] = 1
allReviewData = pd.merge(allReviewData, smallReviewData[['whetherSmall']], left_index=True, right_index=True, how='left')
allReviewData['whetherSmall'].fillna(0, inplace=True)
allReviewData['whetherSmall'] = allReviewData['whetherSmall'].astype(np.int64)
allReviewData['score'] = allReviewData['score'].astype(np.int64)

allReviewData = allReviewData.ix[np.random.permutation(allReviewData.index)]
allReviewData['whetherTest'] = np.zeros(allReviewData.shape[0], dtype=int)
splitTrainTest = []
for name, data in allReviewData.groupby('whetherSmall'):
    data['whetherTest'][:int(data.shape[0]*0.1)] = 1
    splitTrainTest.append(data)
allReviewData = pd.concat(splitTrainTest, axis=0)

allReviewData = allReviewData.sort_values(['whetherTest', 'whetherSmall'], ascending=[True, False])
allReviewData['batchID'] = np.zeros(allReviewData.shape[0], dtype=int)
allReviewData['batchID'][allReviewData['whetherTest'] == 0] = np.arange(np.sum(allReviewData['whetherTest'] == 0))
allReviewData['batchID'][allReviewData['whetherTest'] == 1] = np.arange(np.sum(allReviewData['whetherTest'] == 1))

productData = pd.DataFrame(allReviewData['productID'].unique(), columns=['productID'])
productData['pid'] = np.arange(productData.shape[0])
userData = pd.DataFrame(allReviewData['userID'].unique(), columns=['userID'])
userData['uid'] = np.arange(userData.shape[0])

allReviewData = pd.merge(allReviewData, productData, on='productID', how='left')
allReviewData = pd.merge(allReviewData, userData, on='userID', how='left')
allReviewData['rid'] = np.arange(allReviewData.shape[0])
del allReviewData['productID'], allReviewData['userID']
allReviewData = allReviewData[['rid', 'pid', 'uid', 'score', 'reviewTime', 'whetherTest', 'whetherSmall', 'batchID']]

productData.to_csv('product_data.csv', index=False)
userData.to_csv('user_data.csv', index=False)
allReviewData.to_csv('final_review_data.csv', index=False, header=False)

uAvgScoreTrain = allReviewData[allReviewData['whetherTest'] == 0][['uid', 'score']].groupby('uid').mean()
pAvgScoreTrain = allReviewData[allReviewData['whetherTest'] == 0][['pid', 'score']].groupby('pid').mean()

uAvgScoreTrainSmall = allReviewData[(allReviewData['whetherTest'] == 0) & 
                                    (allReviewData['whetherSmall'] == 1)][['uid', 'score']].groupby('uid').mean()
pAvgScoreTrainSmall = allReviewData[(allReviewData['whetherTest'] == 0) & 
                                    (allReviewData['whetherSmall'] == 1)][['pid', 'score']].groupby('pid').mean()

uAvgScoreTrain = uAvgScoreTrain.rename(columns={'score': 'avgScoreTrain'})
uAvgScoreTrainSmall = uAvgScoreTrainSmall.rename(columns={'score': 'avgScoreTrainSmall'})
pAvgScoreTrain = pAvgScoreTrain.rename(columns={'score': 'avgScoreTrain'})
pAvgScoreTrainSmall = pAvgScoreTrainSmall.rename(columns={'score': 'avgScoreTrainSmall'})

uAvgScore = pd.concat([uAvgScoreTrain, uAvgScoreTrainSmall], axis=1)
pAvgScore = pd.concat([pAvgScoreTrain, pAvgScoreTrainSmall], axis=1)
uAvgScore.fillna(-1, inplace=True)
pAvgScore.fillna(-1, inplace=True)
pAvgScore = pAvgScore.applymap(lambda x: round(x, 6))
uAvgScore = uAvgScore.applymap(lambda x: round(x, 6))

uAvgScore.to_csv('uAvgScore.csv', header=False)
pAvgScore.to_csv('pAvgScore.csv', header=False)

connection = MyriaConnection(rest_url='http://demo.myria.cs.washington.edu:8753')

name_uAvgScore = {'userName': 'public',
                  'programName': 'CSE544_SM_CH',
                  'relationName': 'uAvgScore'} 
schema_uAvgScore = {"columnNames": ['uid', 'avgScoreTrain', 'avgScoreTrainSmall'],
                    "columnTypes": ['LONG_TYPE', 'FLOAT_TYPE', 'FLOAT_TYPE']}
with open('uAvgScore.csv') as f:
    connection.upload_fp(name_uAvgScore, schema_uAvgScore, f)

    
name_pAvgScore = {'userName': 'public',
                  'programName': 'CSE544_SM_CH',
                  'relationName': 'pAvgScore'} 
schema_pAvgScore = {"columnNames": ['pid', 'avgScoreTrain', 'avgScoreTrainSmall'],
                    "columnTypes": ['LONG_TYPE', 'FLOAT_TYPE', 'FLOAT_TYPE']}
with open('pAvgScore.csv') as f:
    connection.upload_fp(name_pAvgScore, schema_pAvgScore, f)
    

name_review = {'userName': 'public',
               'programName': 'CSE544_SM_CH',
               'relationName': 'ReviewData'} 
schema_review = {"columnNames": ['rid', 'pid', 'uid', 'score', 'reviewTime', 'whetherTest', 'whetherSmall', 'batchID'],
                 "columnTypes": ['LONG_TYPE', 'LONG_TYPE', 'LONG_TYPE', 'FLOAT_TYPE', 'STRING_TYPE', 'INT_TYPE', 'INT_TYPE', 'LONG_TYPE']}
with open('final_review_data.csv') as f:
    connection.upload_fp(name_review, schema_review, f)

