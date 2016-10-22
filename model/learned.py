from model.reader import DataReader
from model.base import BaseRecommender


import tensorflow as tf
import numpy as np
from operator import itemgetter


class LearnedRecommender(BaseRecommender):

    def __init__(self, small, batch_size, epoch_num):
	BaseRecommender.__init__(self, small, batch_size)
	self.user_num = self.reader.get_user_num()
	self.prod_num = self.reader.get_prod_num()
	self.batch_num = self.reader.get_batch_num()
	self.epoch_num = epoch_num

    def build(self):
	"""
	build recommender system
	"""

	# Variables

	self.W = tf.Variable(tf.random_uniform([self.prod_num, self.prod_num], -1.0, 1.0), name='W')
	self.b_p = tf.Variable(tf.zeros([self.prod_num]), name='b_prod')
	self.b = tf.Variable(0, name='b')

	# Placeholders

	self.x_bias = tf.placeholder('int', shape=[self.batch_size])
	self.x_rate = tf.placeholder('float32', shape=[self.batch_size, self.prod_num])
	self.x_mask = tf.placeholder('int', shape=[self.batch_size, self.prod_num])
	self.y_rate = tf.placeholder('float32', shape=[self.batch_size])
	self.y_ind = tf.placholder('int', shape=[self.batch_size])

	# models

	b_x = self.x_bias
	b = tf.reshape(tf.tile(self.b, self.batch_size*self.prod_num), [self.batch_size, self.prod_num])
	b_j = tf.reshape(tf.tile(self.b_p, self.batch_size), [self.batch_size, self.prod_num])
	r_xj = (x_rate - b - b_x - b_j) * self.x_mask # [self.batch_size, self.prod_num]
	r_xi = b + b_j + b_x + tf.matmul(r_xj, W) # [self.batch_size, self.prod_num]
	self.pred_y_rate = tf.gather(r_xi, self.y_ind) # [self.batch_size]

	# optimizing
	self.loss = tf.reduce_sum(tf.sqrt(tf.squared_difference(self.y_rate, self.pred_y_rate)))/self.batch_size
	optimizer = tf.train.AdamOptimizer(0.01)
	train_op = optimizer.minimize(loss)

	# training

	self.sess=tf.Session()
	self.sess.run(tf.initialize_all_variables())

	for epoch in range(epoch_num):
	    tot_loss = 0.0
	    for i in range(self.batch_num):
		user_ids, prod_ids, ratings = self.reader.get_next_batch_train()
		_, loss = sess.run([train_op, self.loss], feed_dict = self.get_feed_dict(user_ids, prod_ids, ratings))
		tot_loss += loss

	    avg_loss = tot_loss / self.batch_num
	    print ("Epoch %d : Loss : %.2f" % (epoch, avg_loss))

	print ("Recommender is built!")


    def eval(self):
	"""
	:return performance on test set (Mean Square Root Error)
	"""
	tot_loss = 0.0
	for i in range(self.batch_num):
	    user_ids, prod_ids, ratings = self.reader_get_next_batch_text()
	    loss = sess.run(self.loss, feed_dict = self.get_feed_dict(user_ids, prod_ids, ratings))
	    tot_loss += loss
	return tot_loss / self.batch_num

    def get_feed_dict(self, user_ids, prod_ids, ratings, user_avg = None, prev_prods = None, prev_rates = None):
	if user_avg is None or prev_prods is None or prev_rates is None:
	    prev_prods, prev_rates = self.reader.get_user_rating(user_ids)
	    user_avg = self.reader.get_user_avg(user_ids)

	xr = np.zeros([self.batch_size, self.prod_num])
	xm = np.zeros([self.batch_size, self.prod_num])
		
	for i, prod_id, prev_prod, prev_rate, in enumerate(prod_ids, prev_prods, prev_rates):
	    for j, r in zip(prev_prod, prev_rate):
		if prod_id == j: continue
		xr[i][j] = r
		xm[i][j] = 1
	feed_dict = {
		slef.x_bias : user_avg,
		self.x_rate : xr,
		self.x_mask : xm,
		self.y_rate : ratings,
		self.y_ind : prod_ids
	}
	return feed_dict
	
    def predict(self, userIds, productIds, user_avg, prev_productIds, prev_ratings):
	"""
	:param userIds : list of userIds
	:params productIds : list of productIds
	:params prev_productIds : list of (list of products that users previsouly used)
	:params user_avg : list of average of previous ratings that users gave
	:params prev_ratings : list of (list of ratings that users previously gave)
	:return ratings : list of predicted ratings
	"""
	assert len(userIds) == len(productIds) and len(productIds) == len(user_avg) and len(user_avg) == len(prev_productIds) and len(prev_productIds) == len(prev_ratings)

	example_num = len(userIds)
	if example_num % self.batch_size > 0:
	    add_num = self.batch_size - (example_num % self.batch_size)
	    userIds += [0]*add_num
	    productIds += [0]*add_num
	    user_avg += [0]*add_num
	    prev_productIds += [[0]]*add_num
	    prev_ratings += [[0]]*add_num

	assert len(userIds) % self.batch_size == 0
	batch_num = len(userIds) / self.batch_size
	dummy_ratings = [0]*len(userIds)

	pred_ratings = []
	for i in range(batch_num):
	    start = i * self.batch_size
	    end = (i+1) * self.batch_size
	    curr_userIds, curr_productIds, curr_dummy_ratings, curr_user_avg, curr_prev_productIds, curr_prev_ratings = \
		tuple([data[start:end] for data in [userIds, productIds, dummy_ratings, user_avg, prev_productIds, prev_ratings])
	    curr_pred_rating = sess.run(self.pred_rating, feed_dict = \
		self.get_feed_dict(curr_userIds, curr_prodIds, curr_dummy_ratings, curr_user_avg, curr_prev_productIds, curr_prev_ratings))
	    pred_ratings += list(curr_pred_rating)

	return pred_ratings[:example_num]


