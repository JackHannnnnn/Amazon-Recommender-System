"""
Created on Thu Oct 27 2016

@author: Sewon
"""

from model.BaseRecommender import BaseRecommender


import tensorflow as tf
import numpy as np
from operator import itemgetter
from IPython import embed
import time

class LearnedRecommender(BaseRecommender):

    def __init__(self, small, batch_size):
	BaseRecommender.__init__(self, small, batch_size)
	self.user_num = self.reader.get_user_num()
	self.prod_num = self.reader.get_prod_num()
	self.batch_num_train = self.reader.get_batch_num_train()
	self.batch_num_test = self.reader.get_batch_num_test()
	self.epoch_num = 20
	print "Number of Users : %d\tNumber of Prods : %d" \
		% (self.user_num, self.prod_num)
	print "Number of train batches : %d\tNumber of test batches : %d" \
		% (self.batch_num_train, self.batch_num_test)
	self.tmp = {}

    def build(self):
	"""
	build recommender system
	"""

	# Variables

	self.W = tf.Variable(tf.random_uniform([self.prod_num, self.prod_num], -1.0, 1.0), name='W')
	self.b_p = tf.Variable(tf.zeros([self.prod_num]), name='b_prod')
	self.b = tf.Variable(0.0, name='b')

	# Placeholders

	self.x_bias = tf.placeholder('float32', shape=[self.batch_size])
	self.x_rate = tf.placeholder('float32', shape=[self.batch_size, self.prod_num])
	self.x_mask = tf.placeholder('float32', shape=[self.batch_size, self.prod_num])
	self.y_rate = tf.placeholder('float32', shape=[self.batch_size, self.prod_num])
	self.y_mask = tf.placeholder('float32', shape=[self.batch_size, self.prod_num])

	# models

	b_x_ = tf.tile(self.x_bias, [self.prod_num])
	b_x = tf.reshape(b_x_, [self.batch_size, self.prod_num])
	b_expand = tf.expand_dims(self.b, 0)
	b_ = tf.tile(b_expand, [self.batch_size*self.prod_num])
	b = tf.reshape(b_, [self.batch_size, self.prod_num])
	b_j = tf.reshape(tf.tile(self.b_p, [self.batch_size]), [self.batch_size, self.prod_num])
	r_xj = (self.x_rate - b - b_x - b_j) * self.x_mask # [self.batch_size, self.prod_num]
	self.pred_y_rate = b + b_j + b_x + tf.matmul(r_xj, self.W) # [self.batch_size, self.prod_num]

	# optimizing
	loss_m = tf.sqrt(tf.squared_difference(self.y_rate, self.pred_y_rate))
	self.loss = tf.reduce_sum(loss_m * self.y_mask)/self.batch_size
	optimizer = tf.train.AdamOptimizer(0.01)
	train_op = optimizer.minimize(self.loss)

	# training

	self.sess=tf.Session()
	self.sess.run(tf.initialize_all_variables())

	for epoch in range(self.epoch_num):
	    t = time.time()
	    tot_loss = 0.0
	    for i in range(self.batch_num_train):
		_, loss = self.sess.run([train_op, self.loss], feed_dict = self.get_feed_dict(self.reader.get_next_train()))
		tot_loss += loss
		print "loss : %.2f" %(loss)

	    avg_loss = tot_loss / self.batch_num
	    print ("Epoch %d\tLoss\t%.2f\tTime %dmin" \
		% (epoch, avg_loss, (time.time()-t)/60))

	print ("Recommender is built!")


    def test(self):
	"""
	:return performance on test set (Mean Square Root Error)
	"""
	tot_loss = 0.0
	for i in range(self.batch_num_test):
	    loss = self.sess.run(self.loss, feed_dict = self.get_feed_dict(self.reader_get_next_test()))
	    tot_loss += loss
	return tot_loss / self.batch_num

    def get_feed_dict(self, batch, user_avg = None, prev_prods = None, prev_rates = None):
	user_ids = list(batch['uid'])
	prod_ids = list(batch['pid'])
	ratings = list(batch['score'])

	def manage(prods):
	    l = []
	    for prod in prods:
		if not prod in self.tmp:
		    self.tmp[prod] = len(self.tmp)
		l.append(self.tmp[prod])
	    return l
	prod_ids = manage(prod_ids)

	if user_avg is None or prev_prods is None or prev_rates is None:
	    prev_prods, prev_rates = self.reader.get_user_rating(user_ids)
	    user_avg = self.reader.get_user_avg_rating(user_ids)
	prev_prods = [manage(prev_prod) for prev_prod in prev_prods]	

	xr = np.zeros([self.batch_size, self.prod_num])
	xm = np.zeros([self.batch_size, self.prod_num])
	r  = np.zeros([self.batch_size, self.prod_num])
	ym = np.zeros([self.batch_size, self.prod_num])		

	for i, (prod_id, rating, prev_prod, prev_rate) in enumerate(zip(prod_ids, ratings, prev_prods, prev_rates)):
	    for j, pr in zip(prev_prod, prev_rate):
		if prod_id == j: continue
		xr[i][j] = pr
		xm[i][j] = 1
	    r[i][prod_id] = rating
	    ym[i][prod_id] = 1

	feed_dict = {
		self.x_bias : user_avg,
		self.x_rate : xr,
		self.x_mask : xm,
		self.y_rate : r,
		self.y_mask : ym
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
		tuple([data[start:end] for data in [userIds, productIds, dummy_ratings, user_avg, prev_productIds, prev_ratings]])
	    curr_pred_rating = sess.run(self.pred_rating, feed_dict = \
		self.get_feed_dict(curr_userIds, curr_prodIds, curr_dummy_ratings, curr_user_avg, curr_prev_productIds, curr_prev_ratings))
	    pred_ratings += list(curr_pred_rating)

	return pred_ratings[:example_num]


