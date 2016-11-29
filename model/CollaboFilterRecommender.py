from model.BaseRecommender import BaseRecommender

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
        for i in xrange(self.prod_num):
            for j in xrange(self.prod_num):
                if self.similarity_matrix[i, j] == 0:
                    sim = self.similarity(i, j)
                    self.similarity_matrix[i, j] = sim
                    self.similarity_matrix[j, i] = sim
        sse = 0
        for num in xrange(self.batch_num_train):
            batch_data = self.reader.get_next_train()
            true_ratings = [x[2] for x in batch_data]
            userIds, productIds, prev_productIds, user_avg, prev_ratings = get_pred_data(batch_data)
            pred_ratings = self.predict(userIds, productIds, prev_productIds, user_avg, prev_ratings)
            sse += np.sum((np.array(true_ratings) - np.array(pred_ratings)) ** 2)
        rmse = np.sqrt(sse / float(self.batch_num_train * self.batch_size))
        print 'The training error (RMSE): %d' % rmse
        print 'Recommender is built!'

    def test(self):
        """
        :return performance on test set (Root Mean Squared Error)
        """
        sse = 0
        for num in xrange(self.batch_num_test):
            batch_data = self.reader.get_next_test()
            true_ratings = [x[2] for x in batch_data]
            userIds, productIds, prev_productIds, user_avg, prev_ratings = get_pred_data(batch_data)
            pred_ratings = self.predict(userIds, productIds, prev_productIds, user_avg, prev_ratings)
            sse += np.sum((np.array(true_ratings) - np.array(pred_ratings)) ** 2)
        rmse = np.sqrt(sse / float(self.batch_num_test * self.batch_size))
        print 'The test error (RMSE): %d' % rmse
        

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
            numerator = 0
            denominator = 0
            for j, prev_pid in enumerate(prev_productIds[i]):
                sim = self.similarity_matrix[pid, prev_pid]
                if sim > 0:
                    base_estimate_j = user_avg[i] + self.prod_avg[prev_pid] - self.overall_avg_score
                    numerator += sim * (prev_ratings[j] - base_estimate_j)
                    denominator += sim
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
        
    def similarity(self, pid1, pid2):
        uids, ratings = self.reader.get_prod_rating([pid1, pid2])
        pid1_ratings = np.array(ratings[0])
        pid2_ratings = np.array(ratings[1])
        norm_ratings1 = pid1_ratings - np.mean(pid1_ratings)
        norm_ratings2 = pid2_ratings - np.mean(pid2_ratings)
        numerator = 0
        for index, uid in enumerate(uids[0]):
            if uid in uids[1]:
                numerator += norm_ratings1[index] * norm_ratings2[uids[1].index(uid)]
        denominator = np.sqrt(np.sum(norm_ratings1 ** 2)) * np.sqrt(np.sum(norm_ratings2 ** 2))
        if denominator == 0:
            print 'No one has rated at least one of the two products'
            return 0
        return numerator / float(denominator)
    
        


