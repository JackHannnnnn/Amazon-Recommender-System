import pandas as pd
import numpy as np
from myria import *

class DataReader(object):
    def __init__(self, small, batch_size):  
        self.small = 1 if small else 0
        self.batch_size = batch_size
        self.connection = MyriaConnection(rest_url='http://demo.myria.cs.washington.edu:8753')
        self.ith_train_batch = 0
        self.ith_test_batch = 0
        
    def get_user_num(self):
        # retun the number of unique users 
        if self.small == 1:
            queryStr = """
                   review = scan(public:CSE544_SM_CH:ReviewTable);
                   q = [from review 
                        where whetherSmall == %d
                        emit uid];
                   store(q, data);""" % self.small
            query = MyriaQuery.submit(queryStr, connection=self.connection)
        else:
            queryStr = """
                   review = scan(public:CSE544_SM_CH:ReviewTable);
                   q = [from review
                        emit uid];
                   store(q, data);"""
            query = MyriaQuery.submit(queryStr, connection=self.connection)
        return len(pd.DataFrame(query.to_dict())['uid'].unique())
    
    def get_prod_num(self):
        # return the number of unique products
        if self.small == 1:
            queryStr = """
                   review = scan(public:CSE544_SM_CH:ReviewTable);
                   q = [from review 
                        where whetherSmall == %d
                        emit pid];
                   store(q, data);""" % self.small
            query = MyriaQuery.submit(queryStr, connection=self.connection)
        else:
            queryStr = """
                   review = scan(public:CSE544_SM_CH:Review);
                   q = [from review
                        emit pid];
                   store(q, data);"""
            query = MyriaQuery.submit(queryStr, connection=self.connection)
        return len(pd.DataFrame(query.to_dict())['pid'].unique())
    
    def get_batch_num_train(self):
        # retun the total number of batches in the training set
        if self.small == 1:
            queryStr = """
                   review = scan(public:CSE544_SM_CH:ReviewTable);
                   q = [from review 
                        where whetherSmall == %d and whetherTest == 0
                        emit COUNT(*) as total];
                   store(q, data);""" % self.small
            query = MyriaQuery.submit(queryStr, connection=self.connection)
        else:
            queryStr = """
                   review = scan(public:CSE544_SM_CH:ReviewTable);
                   q = [from review
                        where whetherTest == 0
                        emit COUNT(*) as total];
                   store(q, data);"""
            query = MyriaQuery.submit(queryStr, connection=self.connection)
        return int(query.to_dict()[0]['total'] / float(self.batch_size))
    
    def get_batch_num_test(self):
        # retun the total number of batches in the test set
        if self.small == 1:
            queryStr = """
                   review = scan(public:CSE544_SM_CH:ReviewTable);
                   q = [from review 
                        where whetherSmall == %d and whetherTest == 1
                        emit COUNT(*) as total];
                   store(q, data);""" % self.small
            query = MyriaQuery.submit(queryStr, connection=self.connection)
        else:
            queryStr = """
                   review = scan(public:CSE544_SM_CH:ReviewTable);
                   q = [from review
                        where whetherTest == 1
                        emit COUNT(*) as total];
                   store(q, data);"""
            query = MyriaQuery.submit(queryStr, connection=self.connection)
        return int(query.to_dict()[0]['total'] / float(self.batch_size))

    def get_next_train(self):
        # return next_batch
        # return (user_ids, prod_ids, ratings)
        # (each are a list with length batch_size)
        if self.small == 1:
            queryStr = """
                       review = scan(public:CSE544_SM_CH:ReviewTable);
                       q = [from review 
                            where batchID >= %d and batchID < %d and whetherSmall == %d and whetherTest == 0
                            emit uid, pid, score];
                       store(q, data);""" % (self.ith_train_batch*self.batch_size, (self.ith_train_batch+1)*self.batch_size, self.small)
            query = MyriaQuery.submit(queryStr, connection=self.connection)
        else:
            queryStr = """
                       review = scan(public:CSE544_SM_CH:ReviewTable);
                       q = [from review 
                            where batchID >= %d and batchID < %d and whetherTest == 0
                            emit uid, pid, score];
                       store(q, data);""" % (self.ith_train_batch*self.batch_size, (self.ith_train_batch+1)*self.batch_size)
            query = MyriaQuery.submit(queryStr, connection=self.connection)
        self.ith_train_batch += 1
        train_data = pd.DataFrame(query.to_dict())[['uid', 'pid', 'score']]
        print 'Get %d rows of data' % train_data.shape[0]
	return train_data

    def get_next_test(self):
        # return next_batch
        # return (user_ids, prod_ids, ratings)
        # (each are a list with length batch_size)
        if self.small == 1:
            queryStr = """
                       review = scan(public:CSE544_SM_CH:ReviewTable);
                       q = [from review 
                            where batchID >= %d and batchID < %d and whetherSmall == %d and whetherTest == 1
                            emit uid, pid, score];
                       store(q, data);""" % (self.ith_test_batch*self.batch_size, (self.ith_test_batch+1)*self.batch_size, self.small)
            query = MyriaQuery.submit(queryStr, connection=self.connection)
        else:
            queryStr = """
                       review = scan(public:CSE544_SM_CH:ReviewTable);
                       q = [from review 
                            where batchID >= %d and batchID < %d and whetherTest == 1
                            emit uid, pid, score];
                       store(q, data);""" % (self.ith_test_batch*self.batch_size, (self.ith_test_batch+1)*self.batch_size)
            query = MyriaQuery.submit(queryStr, connection=self.connection)
        self.ith_test_batch += 1
        test_data = pd.DataFrame(query.to_dict())[['uid', 'pid', 'score']]
        print 'Get %d rows of data' % test_data.shape[0]
        return test_data
    
    def get_avg_rating(self):
        # return the the average rating of all reviews on the training set
        if self.small == 1:
            queryStr = """
                       review = scan(public:CSE544_SM_CH:ReviewTable);
                       q = [from review 
                            where whetherSmall == %d and whetherTest == 0
                            emit AVG(score) as avgScore];
                       store(q, data);""" % self.small
            query = MyriaQuery.submit(queryStr, connection=self.connection)
        else:
            queryStr = """
                       review = scan(public:CSE544_SM_CH:ReviewTable);
                       q = [from review 
                            where whetherTest == 0
                            emit AVG(score) as avgScore];
                       store(q, data);"""
            query = MyriaQuery.submit(queryStr, connection=self.connection)
        return query.to_dict()[0]['avgScore']

    def get_user_avg_rating(self, uids):
        # return a list of avg_ratings for the list of users
        avg_rating = []
        if self.small == 1:
            for uid in uids:
                queryStr = """
                           review = scan(public:CSE544_SM_CH:ReviewTable);
                           q = [from review 
                                where whetherSmall == %d and uid == %d and whetherTest == 0
                                emit AVG(score) as avgScore];
                           store(q, data);""" % (self.small, uid)
                query = MyriaQuery.submit(queryStr, connection=self.connection)
                avg_rating.append(query.to_dict()[0]['avgScore'])
        else:
            for uid in uids:
                queryStr = """
                           review = scan(public:CSE544_SM_CH:ReviewTable);
                           q = [from review 
                                where uid == %d and whetherTest == 0
                                emit AVG(score) as avgScore];
                           store(q, data);""" % uid
                query = MyriaQuery.submit(queryStr, connection=self.connection)
                avg_rating.append(query.to_dict()[0]['avgScore'])
        return avg_rating
        
        
    def get_prod_avg_rating(self, pids):
        # return a list of avg_ratings for the list of products
        avg_rating = []
        if self.small == 1:
            for pid in pids:
                queryStr = """
                           review = scan(public:CSE544_SM_CH:ReviewTable);
                           q = [from review 
                                where whetherSmall == %d and pid == %d and whetherTest == 0
                                emit AVG(score) as avgScore];
                           store(q, data);""" % (self.small, pid)
                query = MyriaQuery.submit(queryStr, connection=self.connection)
                avg_rating.append(query.to_dict()[0]['avgScore'])
        else:
            for pid in pids:
                queryStr = """
                           review = scan(public:CSE544_SM_CH:ReviewTable);
                           q = [from review 
                                where pid == %d and whetherTest == 0
                                emit AVG(score) as avgScore];
                           store(q, data);""" % pid
                query = MyriaQuery.submit(queryStr, connection=self.connection)
                avg_rating.append(query.to_dict()[0]['avgScore'])
        return avg_rating

    def get_user_rating(self, uids):
        # param user_ids : list of user_ids
        # return a tuple (prod_ids, ratings)
        # which are Each user's previously used products and their ratings (their length should be same)
        prod_list = []
        rating_list = []
        if self.small == 1:
            for uid in uids:
                queryStr = """
                           review = scan(public:CSE544_SM_CH:ReviewTable);
                           q = [from review 
                                where whetherSmall == %d and uid == %d and whetherTest == 0
                                emit pid, score];
                           store(q, data);""" % (self.small, uid)
                query = MyriaQuery.submit(queryStr, connection=self.connection)
                data = pd.DataFrame(query.to_dict())
                prod_list.append(list(data['pid']))
                rating_list.append(list(data['score']))
        else:
            for uid in uids:
                queryStr = """
                           review = scan(public:CSE544_SM_CH:ReviewTable);
                           q = [from review 
                                where uid == %d and whetherTest == 0
                                emit pid, score];
                           store(q, data);""" % uid
                query = MyriaQuery.submit(queryStr, connection=self.connection)
                data = pd.DataFrame(query.to_dict())
                prod_list.append(list(data['pid']))
                rating_list.append(list(data['score']))
        return (prod_list, rating_list)
        
