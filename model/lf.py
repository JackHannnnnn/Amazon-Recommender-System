from model.reader import DataReader
from model.base import BaseRecommender


class LatentFactorRecommender(BaseRecommender):
    def __init__(self, small, batch_size, factor_num):
	BaseRecommender.__init__(self, small, batch_size)
	self.user_num = self.reader.get_user_num()
	self.prod_num = self.reader.get_prod_num()
	self.batch_num = self.reader.get_batch_num()
	self.factor_num = factor_num

    def build():
	"""
	build recommender system
	"""

    def eval():
	"""
	:return performance on test set (Mean Square Root Error)
	"""
	
    def predict(userIds, productIds):
	"""
	:param userIds : list of userIds
	:params productIds : list of productIds
	:params prev_productIds : list of (list of products that users previsouly used)
	:params user_avg : list of average of previous ratings that users gave
	:params prev_ratings : list of (list of ratings that users previously gave)
	:return ratings : list of predicted ratings
	"""
	assert len(userIds) == len(productIds) and len(productIds) == len(user_avg) and len(user_avg) == len(prev_productIds) and len(prev_productIds) == len(prev_ratings)



