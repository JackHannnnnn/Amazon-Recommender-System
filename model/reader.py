
class DataReader(object):
    def __init__(self, small, batch_size):

	self.user_num = # TODO
	self.prod_num = # TODO
	

    def get_user_num(self):
	# return number of unique users
	return self.user_num

    def get_prod_num(self):
	# return number of unique prods
	return self.prod_num

    def get_batch_num(self):
	# return number of batches (number of ratings / batch_size)

    def get_next_train(self):
	# return next_batch
	# return (user_ids, prod_ids, ratings)
	# (each are a list with length batch_size)
	

    def get_next_test(self):
	# return next batch
	# return (user_ids, prod_ids, ratings)
	# (each are a list with length batch_size)

    def get_avg_rating(self):
	# return avg rating

    def get_user_avg_rating(self, user_ids = range(self.user_num)):
	# param user_ids : single user_id or list of user_ids (default : all user_ids)
	# return single or list of each user's avg rating

    def get_prod_avg_rating(self, prod_ids = range(self.prod_num)):
	# param prod_ids : single prod_id or list of prod_ids (default : all_prod_ids)
	# return single list of each prod's avg rating

    # NOTE: these three methods above should calculate average ony with training set - test set should not be included

    def get_user_rating(self, user_ids):
	# param user_ids : single user_id or list of user_ids
	# return list of (prod_ids, ratings)
	# which are Each user's previously used products and their ratings (their length should be same)

	# NOTE: only rating in training set should be return - test set should not be included



