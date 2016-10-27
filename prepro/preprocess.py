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
    print n1, n2, 
    print 'Dimension of filteredReviews: ', filteredReviews.shape if filteredReviews is not None else '(0, 4)'
    print 'Num of unique Users: ', finalReviews['userID'].unique().shape
    print 'Num of unique Product: ', finalReviews['productID'].unique().shape
    t2 = datetime.datetime.now()
    print t2 - t1
    return filteredReviews

filteredReviewsBig = filteredReviewsInfo(entry_list, 100, 10)
filteredReviewsSmall = filteredReviewsInfo(filteredReviewsBig, 150, 15)

filteredReviewsSmall['whetherSmall'] = 1
filteredReviewsSmall = filteredReviewsSmall[['whetherSmall']]
filteredReviewsBig = pd.merge(filteredReviewsBig, filteredReviewsSmall, left_index=True, right_index=True, how='left')
filteredReviewsBig['whetherSmall'].fillna(0, inplace=True)

indexYesSmall = np.array(filteredReviewsBig[filteredReviewsBig['whetherSmall'] == 1].index)
indexNoSmall = np.array(filteredReviewsBig[filteredReviewsBig['whetherSmall'] == 0].index)
np.random.shuffle(indexYesSmall)
np.random.shuffle(indexNoSmall)
indexYesSmallTest = indexYesSmall[:int((len(indexYesSmall)*0.1))]
indexNoSmallTest = indexNoSmall[:int((len(indexNoSmall)*0.1))]
indexTest = np.array(list(indexYesSmallTest) + list(indexNoSmallTest))
filteredReviewsBig['whetherTest'] = 0
filteredReviewsBig['whetherTest'][filteredReviewsBig.index.isin(indexTest)] = 1
filteredReviewsBig.to_csv('final_review_data.csv', index=False)

userTable = filteredReviewsBig[['userID', 'score']].groupby('userID').mean()
userTable = userTable.rename(columns={'score': 'avgScore'})
userTable.to_csv('user_data.csv')

productTable = filteredReviewsBig[['productID', 'score']].groupby('productID').mean()
productTable = productTable.rename(columns={'score': 'avgScore'})
productTable.to_csv('product_data.csv')

connection = MyriaConnection(rest_url='http://demo.myria.cs.washington.edu:8753')

# Initialize a name and schema for the new relation
name_user = {'userName': 'public',
             'programName': 'CSE544_SM_CH',
             'relationName': 'user'} 
schema_user = { "columnNames" : ["userID", 'avgScore'],
                "columnTypes" : ["STRING_TYPE", "STRING_TYPE"] }

# Now upload that file to Myria
with open('user_data.csv') as f:
    connection.upload_fp(name_user, schema_user, f)
    
name_product = {'userName': 'public',
                'programName': 'CSE544_SM_CH',
                'relationName': 'product'} 
schema_product = { "columnNames" : ["productID", 'avgScore'],
                   "columnTypes" : ["STRING_TYPE", "STRING_TYPE"] }

# Now upload that file to Myria
with open('product_data.csv') as f:
    connection.upload_fp(name_product, schema_product, f)
    
name_review = {'userName': 'public',
                'programName': 'CSE544_SM_CH',
                'relationName': 'review'} 
schema_review = { "columnNames" : ["productID", 'reviewTime', 'score', 'userID', 'whetherSmall', 'whetherTest'],
                  "columnTypes" : ["STRING_TYPE", "STRING_TYPE", "STRING_TYPE", "STRING_TYPE", "STRING_TYPE", "STRING_TYPE"] }

# Now upload that file to Myria
with open('final_review_data.csv') as f:
    connection.upload_fp(name_review, schema_review, f)
    
