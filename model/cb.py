from model.reader import DataReader
from model.base import BaseRecommender

class ContentBasedRecommender(BaseRecommender):
    def __init__(self, small, batch_size):
	BaseRecommender.__init__(self, small, batch)

    def build(self):
	"""
	build recommender system
	"""

    def eval(self):
	"""
	:return performance on test set (Mean Square Root Error)
	"""
	
    def predict(self, userIds, productIds):
	"""
	:param userIds : list of userIds
	:params productIds : list of productIds
	:params prev_productIds : list of (list of products that users previsouly used)
	:params user_avg : list of average of previous ratings that users gave
	:params prev_ratings : list of (list of ratings that users previously gave)
	:return ratings : list of predicted ratings
	"""
	assert len(userIds) == len(productIds) and len(productIds) == len(user_avg) and len(user_avg) == len(prev_productIds) and len(prev_productIds) == len(prev_ratings)



