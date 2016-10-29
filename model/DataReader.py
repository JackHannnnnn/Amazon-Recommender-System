import pandas as pd
import numpy as np
from myria import *

class DataReader(object):
    def __init__(self, small, batch_size, test=False):  
        self.small = 1 if small else 0
        self.batch_size = batch_size
        self.test = 1 if test else 0
        self.connection = MyriaConnection(rest_url='http://demo.myria.cs.washington.edu:8753')
        queryStr = """
                   review = scan(public:CSE544_SM_CH:Review);
                   q = [from review 
                        where batchID < %d and whetherSmall == %d and whetherTest == %d
                        emit uid, pid, score];
                   store(q, data);""" % (self.batch_size, self.small, self.test)
        query = MyriaQuery.submit(queryStr, connection=self.connection)
        self.ith_batch = 1
        self.data = pd.DataFrame(query.to_dict())[['uid', 'pid', 'score']]
        self.user_num = self.data['uid'].unique().shape[0]
        self.prod_num = self.data['pid'].unique().shape[0]

    def get_data(self):
        return self.data
        
    def get_user_num(self):
        return self.user_num
    
    def get_user_ids(self):
        return self.data['uid'].unique()
    
    def get_prod_num(self):
        return self.prod_num
    
    def get_prod_ids(self):
        return self.data['pid'].unique()

    def get_ith_batch(self):
        return self.ith_batch
    
    def get_batch_size(self):
        return self.batch_size

    def get_next(self):
        # return next_batch
        # return (user_ids, prod_ids, ratings)
        # (each are a list with length batch_size)
        queryStr = """
                   review = scan(public:CSE544_SM_CH:Review);
                   q = [from review 
                        where batchID >= %d and batchID < %d and whetherSmall == %d and whetherTest == %d
                        emit uid, pid, score];
                   store(q, data);""" % (self.ith_batch * self.batch_size, (self.ith_batch + 1) * self.batch_size, self.small, self.test)
        query = MyriaQuery.submit(queryStr, connection=self.connection)
        self.ith_batch += 1
        self.data = pd.DataFrame(query.to_dict())[['uid', 'pid', 'score']]
        print 'Get %d rows of data' % self.data.shape[0]
        return self.data

    def get_avg_rating(self):
        return np.mean(self.data['score'])

    def get_user_avg_rating(self, uid):
        # return avg_rating for one user      
        queryStr = """
                   user = scan(public:CSE544_SM_CH:User);
                   q = [from user 
                        where uid == %f
                        emit uid, avgScore];
                   store(q, data);""" % uid
        query = MyriaQuery.submit(queryStr, connection=self.connection)
        return query.to_dict()[0]['avgScore']
        
        
    def get_prod_avg_rating(self, pid):
        # return avg_rating for one product
        queryStr = """
                   prod = scan(public:CSE544_SM_CH:Product);
                   q = [from prod 
                        where pid == %f
                        emit pid, avgScore];
                   store(q, data);""" % pid
        query = MyriaQuery.submit(queryStr, connection=self.connection)
        return query.to_dict()[0]['avgScore']

    def get_user_rating(self, user_ids):
        # param user_ids : single user_id or list of user_ids
        # return list of (prod_ids, ratings)
        # which are Each user's previously used products and their ratings (their length should be same)
        if isinstance(user_ids, (int, float, long)):
            return self.data[self.data['uid'] == user_ids][['pid', 'score']]
        return self.data[self.data['uid'].isin(user_ids)][['pid', 'score']]
