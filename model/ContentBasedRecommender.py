from model.BaseRecommender import BaseRecommender

class ContentBasedRecommender(BaseRecommender):
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
            
        self.prods_profile = None
        self.users_profile = None

    def build(self):
        """
        build recommender system
        """
        self.prods_profile = self.reader.get_prod_profile(range(self.prod_num))
        self.users_profile = self.get_user_profile(range(self.user_num))
        sse = 0
        for num in xrange(self.batch_num_train):
            batch_data = self.reader.get_next_train()
            true_ratings = [x[2] for x in batch_data]
            userIds = [x[0] for x in batch_data]
            productIds = [x[1] for x in batch_data]
            pred_ratings = self.predict(userIds, productIds)
            sse += np.sum((np.array(true_ratings) - np.array(pred_ratings)) ** 2)
        rmse = np.sqrt(sse / float(self.batch_num_train * self.batch_size))
        print 'The training error (RMSE): %d' % rmse
        print 'Recommender is built!'
        

    def test(self):
        """
        :return performance on test set (Mean Square Root Error)
        """
        sse = 0
        for num in xrange(self.batch_num_test):
            batch_data = self.reader.get_next_test()
            true_ratings = [x[2] for x in batch_data]
            userIds = [x[0] for x in batch_data]
            productIds = [x[1] for x in batch_data]
            pred_ratings = self.predict(userIds, productIds)
            sse += np.sum((np.array(true_ratings) - np.array(pred_ratings)) ** 2)
        rmse = np.sqrt(sse / float(self.batch_num_test * self.batch_size))
        print 'The test error (RMSE): %d' % rmse
        
        
    def predict(self, userIds, productIds):
        """
        :param userIds : list of userIds
        :params productIds : list of productIds
        :return ratings : list of predicted ratings
        """
        assert len(userIds) == len(productIds)
        ratings = []
        for uid, pid in zip(userIds, productIds):
            ratings.append(self.similarity(self.users_profile[uid], self.prods_profile[pid])*5)
        return ratings
        
    def get_user_profile(self, userIds):
        user_profile = []
        prev_productIds, prev_ratings = self.reader.get_user_rating(userIds)
        for i, uid in enumerate(userIds):
            numerator = 0
            denominator = 0
            for prev_productId, prev_rating in zip(prev_productIds[i], prev_ratings[i]):
                numerator += prev_rating * np.array(self.prods_profile[prev_productId])
                denominator += prev_rating
            user_profile.append(list(numerator / denominator))
        return user_profile
    
    def similarity(self, user_profile, prod_profile):
        user_profile = np.array(user_profile)
        prod_profile = np.array(prod_profile)
        return np.sum(user_profile*prod_profile) / (np.sqrt(np.sum(user_profile**2)) * np.sqrt(np.sum(prod_profile**2)))



