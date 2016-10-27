"""
Created on Thu Oct 27 2016

@author: Sewon
"""

from model.reader import DataReader

"""

This is a class for base recommender - Classes supporting algorithms should inherit this class.

"""

class BaseRecommender(object):
    def __init__(self, small, batch_size):
	self.small = small
	self.batch_size = batch_size
	self.dataReader = DataReader(small, batch_size)

    def build(self): # This is an abstract method
	"""
	build recommender system
	"""
	return None

    def eval(self):	# This is an abstract method
	"""
	:return performance on test set (Mean Square Root Error)
	"""
	return None
	
    def predict(self, userIds, productIds): # This is an abstract method
	"""
	:param userIds : list of userIds
	:params productIds : list of productIds
	:params prev_productIds : list of (list of products that users previsouly used)
	:params user_avg : list of average of previous ratings that users gave
	:params prev_ratings : list of (list of ratings that users previously gave)
	:return ratings : list of predicted ratings
	"""
	return None

    def recommend(self, userIds, prodIds, ratings, number):
	"""
	:param userIds : user to recommend products. It can be a list of userId, or a single userId
	:param prodIds : which product did user use. It can be a list of (list of prodIds), or a single list of prodIds
	:param ratings : rating. It can be a list of (list of ratings), or a single list of prodIds
	:params number : numbers of recommend products for each users
	:return product : Amazon products of recommend. It can be a list of (list of prodIds), or a single list of prodIds


	It can be evaluated by using predict function (Evaluate all predicted ratings of all products and select k products of highest ratings)
	"""

	if not type(userIds) is 'list': userIds = [userIds]
	if not type(prodIds) is 'list': prodIds = [prodIds]
	if not type(ratings) is 'list': ratings = [ratings]
	assert len(userIds) == len(prodIds) and len(prodIds) == len(ratings)


 	eval_userIds, eval_productIds, eval_user_avg, eval_prev_productIds, eval_prev_ratings = [], [], [], [], []
	user_dict = {}

	for (user, prevProds, prevRates) in zip(userIds, prodIds, ratings):
	    candi_prods = [i for i in range(self.prod_num) if i not in prevProds]
	    avg = np.mean(prevRates)
	    for candi_prod in candi_prods:
		eval_userIds.append(user)
		eval_prodictIds.append(candi_prod)
		eval_user_avg.append(avg)
		eval_prev_productIds.append(prevProds)
		eval_prev_ratings.append(prevRates)
	    user_dict[user] = (len(eval_userIds) - len(candi_prods), len(eval_userIds), candi_prods)

 	ratings = self.predict(eval_userIds, eval_productIds, eval_user_avg, eval_prev_productIds, eval_prev_ratings)

	recommend_prods = []
	for user in userIds:
	    start, end, candi_prods = user_dict[user]
	    curr_ratings = ratings[start:end]
	    pairs = [(prod, rate) for prod, rate in zip(candi_prods, curr_ratings)]
	    topPairs = sorted(pairs, key=itemgetter(1), reverse=True)[:number]
	    recommend_prods.append([i for i, _ in topPairs])

	if len(recommend_prods) == 1:
	    return recommend_prods[0]
	return recommend_prods



