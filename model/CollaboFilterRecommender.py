#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 17:28:55 2016

@author: Chaofan
"""

from model.BaseRecommender import BaseRecommender
import pandas as pd
import numpy as np
import datetime


class CollaboFilterRecommender(BaseRecommender):
    def __init__(self, small, batch_size):
        BaseRecommender.__init__(self, small, batch_size)
        self.user_num = self.reader.get_user_num()
        self.prod_num = self.reader.get_prod_num()
        self.batch_num_train = self.reader.get_batch_num_train()
        self.batch_num_test = self.reader.get_batch_num_test()
        print "Number of Users : %d\tNumber of Prods : %d" \
            % (self.user_num, self.prod_num)
        print "Number of train batches : %d\tNumber of test batches : %d" \
            % (self.batch_num_train, self.batch_num_test)
        
        self.overall_avg_score = None
        self.prod_avg = None
        self.similarity_matrix = None
        
    def build(self):
        """
        build recommender system
        """
        self.overall_avg_score = self.reader.get_avg_rating()
        self.prod_avg = self.reader.get_prod_avg_rating(range(self.prod_num))
        self.similarity_matrix = np.zeros([self.prod_num, self.prod_num])
        self.build_sim_matrix()
        print 'Recommender is built!'

    def test(self):
        """
        :return performance on test set (Root Mean Squared Error)
        """
        start = datetime.datetime.now()
        sse = 0
        for num in xrange(self.batch_num_test):
            t1 = datetime.datetime.now()
            batch_data = self.reader.get_next_test()
            true_ratings = [x[2] for x in batch_data]
            userIds, productIds, prev_productIds, user_avg, prev_ratings = self.get_pred_data(batch_data)
            pred_ratings = self.predict(userIds, productIds, prev_productIds, user_avg, prev_ratings)
            sse += np.sum((np.array(true_ratings) - np.array(pred_ratings)) ** 2)
            print '%dth batch is done!' % num, '\tTime elapsed for this batch:', datetime.datetime.now() - t1
        rmse = np.sqrt(sse / float(self.batch_num_test * self.batch_size))
        print 'Test error (RMSE): %f' % rmse
        print 'Time elapsed:', datetime.datetime.now() - start
        return rmse

    def predict(self, userIds, productIds, prev_productIds, user_avg, prev_ratings):
        """
        :param userIds : list of userIds
        :params productIds : list of productIds
        :params prev_productIds : list of (list of products that users previsouly used)
        :params user_avg : list of average of previous ratings that users gave
        :params prev_ratings : list of (list of ratings that users previously gave)
        :return ratings : list of predicted ratings
        """
        assert len(userIds) == len(productIds) and len(productIds) == len(user_avg) and len(user_avg) == len(prev_productIds) and len(prev_productIds) == len(prev_ratings)
        ratings = []
        for i, (uid, pid) in enumerate(zip(userIds, productIds)):
            base_estimate_i = user_avg[i] + self.prod_avg[pid] - self.overall_avg_score
            numerator = denominator = 0
            for j, prev_pid in enumerate(prev_productIds[i]):
                sim = self.similarity_matrix[pid, prev_pid]
                if sim > 0:
                    base_estimate_j = user_avg[i] + self.prod_avg[prev_pid] - self.overall_avg_score
                    numerator += sim * (prev_ratings[i][j] - base_estimate_j)
                    denominator += sim
            if denominator == 0:
                ratings.append(0)
            else:
                ratings.append(base_estimate_i + numerator / float(denominator))
        return ratings

    def get_pred_data(self, batch_data):
        '''
        :return userIds, productIds, prev_productIds, user_avg, prev_ratings
        '''
        userIds = [x[0] for x in batch_data]
        productIds = [x[1] for x in batch_data]
        user_avg = self.reader.get_user_avg_rating(userIds)
        prev_productIds, prev_ratings = self.reader.get_user_rating(userIds)
        return userIds, productIds, prev_productIds, user_avg, prev_ratings
        
    def build_sim_matrix(self):
        sim_uids, sim_ratings = self.reader.get_prod_rating(range(self.prod_num))
        t1 = datetime.datetime.now()
        count = 0
        for pid1 in range(self.prod_num):
            for pid2 in range(pid1+1, self.prod_num):
                uids = [sim_uids[pid1], sim_uids[pid2]]
                ratings = [sim_ratings[pid1], sim_ratings[pid2]]
                pid1_ratings = np.array(ratings[0])
                pid2_ratings = np.array(ratings[1])
                norm_ratings1 = pid1_ratings - np.mean(pid1_ratings)
                norm_ratings2 = pid2_ratings - np.mean(pid2_ratings)

                comm_uids = list(set(uids[0])&set(uids[1]))
                comm_r1 = np.array([norm_ratings1[uids[0].index(uid)] for uid in comm_uids])
                comm_r2 = np.array([norm_ratings2[uids[1].index(uid)] for uid in comm_uids])

                numerator = sum(comm_r1 * comm_r2)
                denominator = np.sqrt(np.sum(comm_r1 ** 2)) * np.sqrt(np.sum(comm_r2 ** 2))
                sim = 0 if denominator == 0 else numerator / float(denominator)
                self.similarity_matrix[pid1, pid2] = self.similarity_matrix[pid2, pid1] = sim
                count += 1
                if count % 10000 == 0:
                    print count, 'times', 'Time elapsed: ', datetime.datetime.now() - t1
    
        
