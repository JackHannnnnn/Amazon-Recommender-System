#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 20:28:07 2016

@author: Chaofan
"""

import pandas as pd
import numpy as np
import MySQLdb as mdb

class DataReader(object):
    def __init__(self, small, batch_size):  
        self.small = 1 if small else 0
        self.batch_size = batch_size
        self.ith_train_batch = 0
        self.ith_test_batch = 0
        self.con = mdb.connect('localhost', 'testuser', 'test623', 'testdb')
        self.cur = self.con.cursor()
        self.user_num = None
        self.prod_num = None
        self.batch_num_train = None
        self.batch_num_test = None
        
    def get_user_num(self):
        ''' int: return the number of unique users'''
        if self.user_num is not None:
            return self.user_num
        
        if self.small == 1:
            queryStr = 'select count(*) from User where avgScoreTrainSmall != -1;'
        else:
            queryStr = 'select count(*) from User;'
        self.cur.execute(queryStr)
        self.user_num = self.cur.fetchall()[0][0]
        return self.user_num

    def get_prod_num(self):
        ''' int: return the number of unique products'''
        if self.prod_num is not None:
            return self.prod_num
        
        if self.small == 1:
            queryStr = 'select count(*) from Product where avgScoreTrainSmall != -1;'
        else:
            queryStr = 'select count(*) from Product;'
        self.cur.execute(queryStr)
        self.prod_num = self.cur.fetchall()[0][0]
        return self.prod_num

    def get_batch_num_train(self):
        ''' int: retun the total number of batches in the training set'''
        if self.batch_num_train is not None:
            return self.batch_num_train

        if self.small == 1:
            queryStr = 'select count(*) from Review where whetherSmall = 1 and whetherTest = 0;'
        else:
            queryStr = 'select count(*) from Review where whetherTest = 0;'
        self.cur.execute(queryStr)
        self.batch_num_train = int(np.floor(self.cur.fetchall()[0][0] / float(self.batch_size)))
        return self.batch_num_train  

    def get_batch_num_test(self):
        ''' int: retun the total number of batches in the test set'''
        if self.batch_num_test is not None:
            return self.batch_num_test
    
        if self.small == 1:
            queryStr = 'select count(*) from Review where whetherSmall = 1 and whetherTest = 1;'
        else:
            queryStr = 'select count(*) from Review where whetherTest = 1;'
        self.cur.execute(queryStr)
        self.batch_num_test = int(np.floor(self.cur.fetchall()[0][0] / float(self.batch_size)))
        return self.batch_num_test 

    def get_next_train(self):
        '''
        return the training set of review data of the next batch
        return a tuple of tuples whose schema is ['uid', 'pid', 'score'] (types: [int, int, float])
        the dimension of this tuple of tuples is batch_size * 3
        '''
        if self.ith_train_batch == 0:
            print 'This is the first batch of the training data'
        
        if self.small == 1:
            queryStr = '''select uid, pid, score 
                              from Review 
                              where batchID >= %d and batchID < %d and whetherSmall = 1 and whetherTest = 0;
                       ''' % (self.ith_train_batch*self.batch_size, (self.ith_train_batch+1)*self.batch_size)
        else:
            queryStr = '''select uid, pid, score 
                              from Review 
                              where batchID >= %d and batchID < %d and whetherTest = 0;
                       ''' % (self.ith_train_batch*self.batch_size, (self.ith_train_batch+1)*self.batch_size)
        self.cur.execute(queryStr)
        self.ith_train_batch += 1
        if self.ith_train_batch == self.batch_num_train:
            self.ith_train_batch = 0
	data = self.cur.fetchall()
        return data

    def get_next_test(self):
        '''
        return the test set of review data of the next batch
        return a tuple of tuples whose schema is ['uid', 'pid', 'score'] (types: [int, int, float])
        the dimension of this tuple of tuples is batch_size * 3
        '''
        if self.ith_test_batch == 0:
            print 'This is the first batch of the test data'
        
        if self.small == 1:
            queryStr = '''select uid, pid, score 
                              from Review 
                              where batchID >= %d and batchID < %d and whetherSmall = 1 and whetherTest = 1;
                       ''' % (self.ith_test_batch*self.batch_size, (self.ith_test_batch+1)*self.batch_size)
        else:
            queryStr = '''select uid, pid, score 
                              from Review 
                              where batchID >= %d and batchID < %d and whetherTest = 1;
                       ''' % (self.ith_test_batch*self.batch_size, (self.ith_test_batch+1)*self.batch_size)
        self.cur.execute(queryStr)
        self.ith_test_batch += 1
        if self.ith_test_batch == self.batch_num_test:
            self.ith_test_batch = 0
        return self.cur.fetchall()
    
    def get_avg_rating(self):
        ''' float: return the the average rating of all reviews on the training set'''
        if self.small == 1:
            queryStr = 'select avg(score) from Review where whetherSmall = 1 and whetherTest = 0;'
        else:
            queryStr = 'select avg(score) from Review where whetherTest = 0;'
        self.cur.execute(queryStr)
        return self.cur.fetchall()[0][0]

    def get_user_avg_rating(self, uids):
        '''
        input: a list of integers
        output: a list of floats: return a list of avg_ratings for the list of users
        '''
        uidsStr = str(tuple([int(uid) for uid in uids]))
        if self.small == 1:
            queryStr = '''select uid, avgScoreTrainSmall 
                              from User 
                              where uid in %s;''' % uidsStr
        else:
            queryStr = '''select uid, avgScoreTrain 
                              from User 
                              where uid in %s;''' % uidsStr
        self.cur.execute(queryStr)
        rating_dict = {row[0]:row[1] for row in self.cur.fetchall()}
        return [rating_dict[uid] for uid in uids]
        
    def get_prod_avg_rating(self, pids):
        '''
        input: a list of integers
        output: a list of floats: return a list of avg_ratings for the list of products
        '''
        pidsStr = str(tuple([ int(pid) for pid in pids]))
        if self.small == 1:
            queryStr = '''select pid, avgScoreTrainSmall 
                              from Product 
                              where pid in %s;''' % pidsStr
        else:
            queryStr = '''select pid, avgScoreTrain 
                              from Product 
                              where pid in %s;''' % pidsStr
        self.cur.execute(queryStr)
        rating_dict = {row[0]:row[1] for row in self.cur.fetchall()}
        return [rating_dict[pid] for pid in pids]
    
    def get_user_rating(self, uids):
        '''
        input: a list of integers (user_id)
        output: return a tuple (prod_ids, ratings)
        which are Each user's previously used products and their ratings (their length should be same)
        '''
        uidsStr = str(tuple([int(uid) for uid in uids]))
        if self.small == 1:
            queryStr = '''select uid, pid, score 
                              from Review 
                              where uid in %s and whetherSmall = 1 and whetherTest = 0;''' % (uidsStr, )
        else:
            queryStr = '''select uid, pid, score 
                              from Review 
                              where uid in %s and whetherTest = 0;''' % uidsStr
        self.cur.execute(queryStr)
        data = pd.DataFrame(list(self.cur.fetchall()), columns=['uid', 'pid', 'score'])
        prod_dict = {}
        rating_dict = {}
        for name, group in data.groupby('uid'):
            prod_dict[name] = list(group['pid'])
            rating_dict[name] = list(group['score'])
        return ([prod_dict[uid] for uid in uids], [rating_dict[uid] for uid in uids])
    
    def get_prod_rating(self, pids):
        '''
        input: a list of integers (product_id)
        output: return a tuple (user_ids, ratings)
        which are users that previously rated this product and their ratings (their length should be same)
        '''
        pidsStr = str(tuple([int(pid) for pid in pids]))
        if self.small == 1:
            queryStr = '''select uid, pid, score 
                              from Review 
                              where pid in %s and whetherSmall = 1 and whetherTest = 0;''' % pidsStr
        else:
            queryStr = '''select uid, pid, score 
                              from Review 
                              where pid in %s and whetherTest = 0;''' % pidsStr
        self.cur.execute(queryStr)
        data = pd.DataFrame(list(self.cur.fetchall()), columns=['uid', 'pid', 'score'])
        user_dict = {}
        rating_dict = {}
        for name, group in data.groupby('pid'):
            user_dict[name] = list(group['uid'])
            rating_dict[name] = list(group['score'])
        return ([user_dict[pid] for pid in pids], [rating_dict[pid] for pid in pids])
    
    def get_prod_profile(self, pids):
        '''
        input: a list of integers (product_id)
        output: a tuple of tuples which are profile features of each product
        '''
        pidsStr = str(tuple([int(pid) for pid in pids]))
        queryStr = '''select price, numAlsoBought, numAlsoViewed, numBoughtTogether, numBuyAfterViewing,
                             salesRank, super_category_1, super_category_2, super_category_3, 
                             super_category_4, super_category_5, super_category_6
                          from Product p inner join Category c on p.pid = c.pid
                          where p.pid in %s;''' % pidsStr
        self.cur.execute(queryStr)
        return self.cur.fetchall()
    
