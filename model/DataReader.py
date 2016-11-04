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
        # int: return the number of unique users 
        if self.small == 1:
            queryStr = """
                   review = scan(public:CSE544_SM_CH:ReviewData);
                   q = [from review 
                        where whetherSmall = %d
                        emit uid];
                   store(q, data);""" % self.small
            query = MyriaQuery.submit(queryStr, connection=self.connection)
        else:
            queryStr = """
                   review = scan(public:CSE544_SM_CH:ReviewData);
                   q = [from review
                        emit uid];
                   store(q, data);"""
            query = MyriaQuery.submit(queryStr, connection=self.connection)
        return len(pd.DataFrame(query.to_dict())['uid'].unique())
    
    def get_prod_num(self):
        # int: return the number of unique products
        if self.small == 1:
            queryStr = """
                   review = scan(public:CSE544_SM_CH:ReviewData);
                   q = [from review 
                        where whetherSmall = %d
                        emit pid];
                   store(q, data);""" % self.small
            query = MyriaQuery.submit(queryStr, connection=self.connection)
        else:
            queryStr = """
                   review = scan(public:CSE544_SM_CH:ReviewData);
                   q = [from review
                        emit pid];
                   store(q, data);"""
            query = MyriaQuery.submit(queryStr, connection=self.connection)
        return len(pd.DataFrame(query.to_dict())['pid'].unique())
    
    def get_batch_num_train(self):
        # int: retun the total number of batches in the training set
        if self.small == 1:
            queryStr = """
                   review = scan(public:CSE544_SM_CH:ReviewData);
                   q = [from review 
                        where whetherSmall = %d and whetherTest = 0
                        emit COUNT(*) as total];
                   store(q, data);""" % self.small
            query = MyriaQuery.submit(queryStr, connection=self.connection)
        else:
            queryStr = """
                   review = scan(public:CSE544_SM_CH:ReviewData);
                   q = [from review
                        where whetherTest = 0
                        emit COUNT(*) as total];
                   store(q, data);"""
            query = MyriaQuery.submit(queryStr, connection=self.connection)
        return int(np.floor(query.to_dict()[0]['total'] / float(self.batch_size)))
    
    def get_batch_num_test(self):
        # int: retun the total number of batches in the test set
        if self.small == 1:
            queryStr = """
                   review = scan(public:CSE544_SM_CH:ReviewData);
                   q = [from review 
                        where whetherSmall = %d and whetherTest = 1
                        emit COUNT(*) as total];
                   store(q, data);""" % self.small
            query = MyriaQuery.submit(queryStr, connection=self.connection)
        else:
            queryStr = """
                   review = scan(public:CSE544_SM_CH:ReviewData);
                   q = [from review
                        where whetherTest = 1
                        emit COUNT(*) as total];
                   store(q, data);"""
            query = MyriaQuery.submit(queryStr, connection=self.connection)
        return int(np.floor(query.to_dict()[0]['total'] / float(self.batch_size)))

    def get_next_train(self):
        '''
        return the training set of review data of the next batch
        return a list of lists whose schema is ['uid', 'pid', 'score'] (format: [float, float, float])
        the dimension of this list of lists is batch_size * 3
        '''
        if self.small == 1:
            queryStr = """
                       review = scan(public:CSE544_SM_CH:ReviewData);
                       q = [from review 
                            where batchID >= %d and batchID < %d and whetherSmall = %d and whetherTest = 0
                            emit uid, pid, score];
                       store(q, data);""" % (self.ith_train_batch*self.batch_size, (self.ith_train_batch+1)*self.batch_size, self.small)
            query = MyriaQuery.submit(queryStr, connection=self.connection)
        else:
            queryStr = """
                       review = scan(public:CSE544_SM_CH:ReviewData);
                       q = [from review 
                            where batchID >= %d and batchID < %d and whetherTest = 0
                            emit uid, pid, score];
                       store(q, data);""" % (self.ith_train_batch*self.batch_size, (self.ith_train_batch+1)*self.batch_size)
            query = MyriaQuery.submit(queryStr, connection=self.connection)
        self.ith_train_batch += 1
        train_data = query.to_dataframe()[['uid', 'pid', 'score']]
        print 'Get %d rows of data' % train_data.shape[0]
        return train_data.values.tolist()

    def get_next_test(self):
        '''
        return the training set of review data of the next batch
        return a list of lists whose schema is ['uid', 'pid', 'score'] (format: [float, float, float])
        the dimension of this list of lists is batch_size * 3
        '''
        if self.small == 1:
            queryStr = """
                       review = scan(public:CSE544_SM_CH:ReviewData);
                       q = [from review 
                            where batchID >= %d and batchID < %d and whetherSmall = %d and whetherTest = 1
                            emit uid, pid, score];
                       store(q, data);""" % (self.ith_test_batch*self.batch_size, (self.ith_test_batch+1)*self.batch_size, self.small)
            query = MyriaQuery.submit(queryStr, connection=self.connection)
        else:
            queryStr = """
                       review = scan(public:CSE544_SM_CH:ReviewData);
                       q = [from review 
                            where batchID >= %d and batchID < %d and whetherTest = 1
                            emit uid, pid, score];
                       store(q, data);""" % (self.ith_test_batch*self.batch_size, (self.ith_test_batch+1)*self.batch_size)
            query = MyriaQuery.submit(queryStr, connection=self.connection)
        self.ith_test_batch += 1
        test_data = query.to_dataframe()[['uid', 'pid', 'score']]
        print 'Get %d rows of data' % test_data.shape[0]
        return test_data.values.tolist()
    
    def get_avg_rating(self):
        # float: return the the average rating of all reviews on the training set
        if self.small == 1:
            queryStr = """
                       review = scan(public:CSE544_SM_CH:ReviewData);
                       q = [from review 
                            where whetherSmall = %d and whetherTest = 0
                            emit AVG(score) as avgScore];
                       store(q, data);""" % self.small
            query = MyriaQuery.submit(queryStr, connection=self.connection)
        else:
            queryStr = """
                       review = scan(public:CSE544_SM_CH:ReviewData);
                       q = [from review 
                            where whetherTest = 0
                            emit AVG(score) as avgScore];
                       store(q, data);"""
            query = MyriaQuery.submit(queryStr, connection=self.connection)
        return query.to_dict()[0]['avgScore']

    def get_user_avg_rating(self, uids):
        '''
        input: a list of integers
        output: a list of floats: return a list of avg_ratings for the list of users
        '''
        if self.small == 1:
            queryStr = """
                           uAvgScore = scan(public:CSE544_SM_CH:uAvgScore);
                           q = [from uAvgScore
                                emit uid, avgScoreTrainSmall];
                           store(q, data);"""
            query = MyriaQuery.submit(queryStr, connection=self.connection)
            query = query.to_dataframe()
            u_data = dict(zip(query['uid'], query['avgScoreTrainSmall']))
        else:
            queryStr = """
                           uAvgScore = scan(public:CSE544_SM_CH:uAvgScore);
                           q = [from uAvgScore
                                emit uid, avgScoreTrain];
                           store(q, data);"""
            query = MyriaQuery.submit(queryStr, connection=self.connection)
            query = query.to_dataframe()
            u_data = dict(zip(query['uid'], query['avgScoreTrain']))
        return [u_data[uid] for uid in uids]
        
        
    def get_prod_avg_rating(self, pids):
        '''
        input: a list of integers
        output: a list of floats: return a list of avg_ratings for the list of products
        '''
        if self.small == 1:
            queryStr = """
                           pAvgScore = scan(public:CSE544_SM_CH:pAvgScore);
                           q = [from pAvgScore
                                emit pid, avgScoreTrainSmall];
                           store(q, data);"""
            query = MyriaQuery.submit(queryStr, connection=self.connection)
            query = query.to_dataframe()
            p_data = dict(zip(query['pid'], query['avgScoreTrainSmall']))
        else:
            queryStr = """
                           pAvgScore = scan(public:CSE544_SM_CH:pAvgScore);
                           q = [from pAvgScore
                                emit pid, avgScoreTrain];
                           store(q, data);"""
            query = MyriaQuery.submit(queryStr, connection=self.connection)
            query = query.to_dataframe()
            p_data = dict(zip(query['pid'], query['avgScoreTrain']))
        return [p_data[pid] for pid in pids]

    def get_user_rating(self, uids):
        '''
        input: a list of integers (user_id)
        output: return a tuple (prod_ids, ratings)
        which are Each user's previously used products and their ratings (their length should be same)
        '''
        if self.small == 1:
            queryStr = """
                           review = scan(public:CSE544_SM_CH:ReviewData);
                           q = [from review
                                where whetherSmall = %d and whetherTest = 0
                                emit uid, pid, score];
                           store(q, data);""" % self.small
            query = MyriaQuery.submit(queryStr, connection=self.connection)
            data = query.to_dataframe()
            prod_dict = {}
            rating_dict = {}
            data = data[data['uid'].isin(uids)]
            for name, group in data.groupby('uid'):
                prod_dict[name] = list(group['pid'])
                rating_dict[name] = list(group['score'])
        else:
            queryStr = """
                           review = scan(public:CSE544_SM_CH:ReviewData);
                           q = [from review
                                where whetherTest = 0
                                emit uid, pid, score];
                           store(q, data);"""
            query = MyriaQuery.submit(queryStr, connection=self.connection)
            data = query.to_dataframe()
            prod_dict = {}
            rating_dict = {}
            data = data[data['uid'].isin(uids)]
            for name, group in data.groupby('uid'):
                prod_dict[name] = list(group['pid'])
                rating_dict[name] = list(group['score'])
        return ([prod_dict[uid] for uid in uids], [rating_dict[uid] for uid in uids])
        
