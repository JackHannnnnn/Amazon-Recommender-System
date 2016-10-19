
"""

This is a class for base recommender - Classes supporting algorithms should inherit this class.

"""

class BaseRecommender(object):
    def build(train_set):
	"""
	:param train_set : a dictionary of train_data which contains
		'rating_list'  : a list of rating_id
		'num_user' : number of distinct users
		'num_prod' : number of distinct products
	"""

    def eval(test_set):
	"""
	:param test_set : a dictionary of test_set to evaluate performance
	:return performance (Mean Square Root Error)
	"""
	
    def predict(userIds, productIds):
	"""
	:param userIds : list of userIds
	:params productIds : list of productIds
	:return ratings : list of predicted ratings
	"""

    def recommend(user, number):
	"""
	:param user : user to recommend products. It can be a list of userId, or a single userId
	:params number : numbers of recommend products for each users
	:return product : Amazon products of recommend. It can be a list of (list of prodIds), or a single list of prodIds


	It can be evaluated by using predict function (Evaluate all predicted ratings of all products and select k products of highest ratings)
	"""
