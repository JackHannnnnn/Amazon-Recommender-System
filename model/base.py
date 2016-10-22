
"""

This is a class for base recommender - Classes supporting algorithms should inherit this class.

"""

class BaseRecommender(object):
    def build(train_set):
	"""
	"""

    def eval():
	"""
	:return performance (Mean Square Root Error)
	"""
	
    def predict(userIds, productIds):
	"""
	:param userIds : list of userIds
	:params productIds : list of productIds
	:return ratings : list of predicted ratings
	"""

    def recommend(userIds, prodIds, ratings, number):
	"""
	:param userIds : user to recommend products. It can be a list of userId, or a single userId
	:param prodIds : which product did user use. It can be a list of (list of prodIds), or a single list of prodIds
	:param ratings : rating. It can be a list of (list of ratings), or a single list of prodIds
	:params number : numbers of recommend products for each users
	:return product : Amazon products of recommend. It can be a list of (list of prodIds), or a single list of prodIds


	It can be evaluated by using predict function (Evaluate all predicted ratings of all products and select k products of highest ratings)
	"""
