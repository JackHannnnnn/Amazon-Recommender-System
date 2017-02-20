#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 16:28:55 2016

@author: Chaofan
"""

# Import libraries
import pandas as pd
import numpy as np
import datetime
import gzip
from  sklearn.decomposition import PCA
import MySQLdb as mdb

# Helper functions
def parse(path):
  f = gzip.open(path, 'r')
  for l in f:
    yield eval(l)
    
def filterReviewsByField(reviews, field, minNumReviews):
    reviewsCountByField = reviews.groupby(field).size()
    fieldIDWithNumReviewsPlus = reviewsCountByField[reviewsCountByField >= minNumReviews].index
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

def convert_category(category):
    if category == '' or category == -1:
        return 'Unknown'
    return category
    
    
    
def preprocessData():   
    # Extract review data from the original data set
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
        except StopIteration:
            print 'Read %s observations in total' % rowCount
            entry_list = pd.DataFrame({'productID': productID,
                                       'userID': userID,
                                       'score': score,
                                       'reviewTime': reviewTime})
            filename = 'review_data.csv'
            entry_list.to_csv(filename, index=False)
            print 'Save the data in the file %s' % filename
            break
    
    
    # Prepare the review data for training and testing the algorithms
    entry_list = pd.read_csv('review_data.csv')
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
    allReviewData.to_csv('final_review_data.txt', sep='\t', index=False, header=False)
    
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
    
    uAvgScore.to_csv('user_data.txt', sep='\t', header=False)
    pAvgScore.to_csv('pAvgScore.csv', header=False)
    
    
    # Preprocess product data for Content-based Recommender System
    product_data = productData
    product_ids = product_data['productID'].tolist()
    products = {}
    products['productID'] = []
    products['price'] =[]
    products['num_also_bought'] = []
    products['num_also_viewed'] = []
    products['num_bought_together'] = []
    products['num_buy_after_viewing'] = []
    products['sales_rank'] = []
    products['category'] = []
    products['brand'] = []
    
    for l in open('metadata.json'):
        product = eval(l)
        if product['asin'] in product_ids:
            products['productID'].append(product['asin'])
            products['price'].append(product.get('price', np.nan))
            if product.get('related', -1) == -1:
                products['num_also_bought'].append(0)
                products['num_also_viewed'].append(0)
                products['num_bought_together'].append(0)
                products['num_buy_after_viewing'].append(0)
            else:
                also_bought = product['related'].get('also_bought', [])
                also_viewed = product['related'].get('also_viewed', [])
                bought_together = product['related'].get('bought_together', [])
                buy_after_viewing = product['related'].get('buy_after_viewing', [])
                products['num_also_bought'].append(len(also_bought))
                products['num_also_viewed'].append(len(also_viewed))
                products['num_bought_together'].append(len(bought_together))
                products['num_buy_after_viewing'].append(len(buy_after_viewing))
            if product.get('salesRank', -1) == -1 or len(product['salesRank']) == 0:
                products['sales_rank'].append(np.nan)
                products['category'].append(product.get('categories', [[-1]])[0][0])
            else:
                sales_rank = product['salesRank'].popitem()
                products['sales_rank'].append(sales_rank[1])
                products['category'].append(sales_rank[0])
            products['brand'].append(product.get('brand', -1))
    
    
    products = pd.DataFrame(products)
    product_data = pd.merge(product_data, products, on='productID', how='left')
    del product_data['productID'], product_data['brand']
    del products
    
    product_data['category'] = product_data['category'].map(convert_category)
    product_data['category'].fillna('Unknown', inplace=True)
    product_data['price'].fillna(product_data['price'].median(), inplace=True)
    product_data['sales_rank'].fillna(product_data['sales_rank'].median(), inplace=True)
    product_data['num_also_bought'].fillna(product_data['num_also_bought'].median(), inplace=True)
    product_data['num_also_viewed'].fillna(product_data['num_also_viewed'].median(), inplace=True)
    product_data['num_bought_together'].fillna(product_data['num_bought_together'].median(), inplace=True)
    product_data['num_buy_after_viewing'].fillna(product_data['num_buy_after_viewing'].median(), inplace=True)
    
    categories = pd.get_dummies(product_data['category']).astype(np.int8)
    pca = PCA(n_components=0.85)
    categories_pca = pca.fit_transform(categories)
    super_categories = ['super_category_1',
                        'super_category_2',
                        'super_category_3',
                        'super_category_4',
                        'super_category_5',
                        'super_category_6']
    category_data = pd.DataFrame(categories_pca, columns=super_categories)
    category_data = pd.concat([product_data[['pid']], category_data], axis=1)
    category_data.to_csv('category_data.txt', sep='\t', index=False, header=False)
    
    
    pAvgScore = pd.read_csv('pAvgScore.csv', header=None, 
                            names=['pid', 'avgScoreTrain', 'avgScoreTrainSmall'])
    product_data = pd.merge(product_data, pAvgScore, on='pid')
    product_data = product_data[['pid', 'price', 'category',
                                 'num_also_bought', 'num_also_viewed',
                                 'num_bought_together', 'num_buy_after_viewing',
                                 'sales_rank', 'avgScoreTrain', 'avgScoreTrainSmall']]
    product_data.to_csv('product_data.txt', sep='\t', index=False, header=False)
    
    
    # Upload the data to the MySQL Database on an AWS EC2 instance
    con = mdb.connect('localhost', 'testuser', 'test623', 'testdb', local_infile=1)
    cur = con.cursor()
    
    cur.execute('''create table User (
                       uid int, 
                       avgScoreTrain float, 
                       avgScoreTrainSmall float);''')
    cur.execute('''LOAD DATA LOCAL INFILE 'user_data.txt' 
                       INTO TABLE User 
                       fields terminated by '\t' 
                       lines terminated by '\n';''')
    cur.execute('''create table Product (
                       pid int, 
                       price float, 
                       category varchar(60),
                       numAlsoBought int,
                       numAlsoViewed int,
                       numBoughtTogether int,
                       numBuyAfterViewing int,
                       salesRank int,
                       avgScoreTrain float,
                       avgScoreTrainSmall float);''')
    cur.execute('''LOAD DATA LOCAL INFILE 'product_data.txt' 
                       INTO TABLE Product 
                       fields terminated by '\t' 
                       lines terminated by '\n';''')
    cur.execute('''create table Category (
                       pid int,
                       super_category_1 float,
                       super_category_2 float,
                       super_category_3 float,
                       super_category_4 float,
                       super_category_5 float,
                       super_category_6 float);''')
    cur.execute('''LOAD DATA LOCAL INFILE 'category_data.txt' 
                       INTO TABLE Category 
                       fields terminated by '\t' 
                       lines terminated by '\n';''')
    cur.execute('''create table Review (
                       rid int,
                       pid int,
                       uid int,
                       score float,
                       reviewTime varchar(60),
                       whetherTest int,
                       whetherSmall int,
                       batchID int);''')
    cur.execute('''LOAD DATA LOCAL INFILE 'final_review_data.txt' 
                       INTO TABLE Review 
                       fields terminated by '\t' 
                       lines terminated by '\n';''')
    con.commit()
    con.close()


if __name__ == '__main__':
    preprocessData()
    
