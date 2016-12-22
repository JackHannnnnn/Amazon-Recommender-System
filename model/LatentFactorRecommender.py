"""
Created on Thu Oct 27 2016

@author: Sewon
"""

from model.BaseRecommender import BaseRecommender

import numpy as np
import tensorflow as tf
import time

class LatentFactorRecommender(BaseRecommender):
    def __init__(self, small, batch_size, factor_num, wd = 0.001):
	BaseRecommender.__init__(self, small, batch_size)
	self.user_num = self.reader.get_user_num()
	self.prod_num = self.reader.get_prod_num()
	self.batch_num_train = self.reader.get_batch_num_train()
	self.batch_num_test = self.reader.get_batch_num_test()
	self.factor_num = factor_num
	self.user_dic = {}

	self.epoch_num = 20
	self.wd = 0
	print "Number of Users : %d\tNumber of Prods : %d" \
		% (self.user_num, self.prod_num)
	print "Number of train batches : %d\tNumber of test batches : %d" \
		% (self.batch_num_train, self.batch_num_test)
	
    def build(self):
	"""
	build recommender system
	"""

	# Variables

	self.Wp = tf.Variable(tf.random_uniform([self.prod_num, self.factor_num], -1.0, 1.0), name='Wp')
	self.Wu = tf.Variable(tf.random_uniform([self.user_num, self.factor_num], -1.0, 1.0), name='Wu')
	
	# Placeholders

	
	self.u_ind = tf.placeholder('int32', shape=[self.batch_size])
	self.y_rate = tf.placeholder('float32', shape=[self.batch_size, self.prod_num])
	self.y_mask = tf.placeholder('float32', shape=[self.batch_size, self.prod_num])

	# models

	for i in range(self.batch_size):
	    ind = tf.gather(self.u_ind, i)
	    currU = tf.gather(self.Wu, ind)
	    if i==0: U=currU
	    else: U=tf.concat(0, [U, currU])
	# U : [batch_size, factor_num]
	# Wp : [prod_num, factor_num]
	self.pred_y_rate = tf.matmul(tf.reshape(U, [self.batch_size, self.factor_num]), tf.transpose(self.Wp))
	_loss = tf.sqrt(tf.squared_difference(self.y_rate, self.pred_y_rate))
	self.l2_loss = tf.nn.l2_loss(self.Wp) + tf.nn.l2_loss(self.Wu)
	self.loss = tf.reduce_sum(_loss * self.y_mask)/self.batch_size
	self.tot_loss = self.loss + self.wd * self.l2_loss
	optimizer = tf.train.AdamOptimizer(0.01)
	train_op = optimizer.minimize(self.tot_loss)

	#training
	self.sess = tf.Session()
	self.sess.run(tf.initialize_all_variables())

	for epoch in range(self.epoch_num):
	    t = time.time()
	    tot_loss = 0.0
	    for i in range(self.batch_num_train):
		_, loss = self.sess.run([train_op, self.loss], feed_dict = self.get_feed_dict(self.reader.get_next_train()))
		tot_loss += loss
		#print (loss)
	    avg_loss = tot_loss / self.batch_num_train
	    print ("Epoch %d\tLoss\t%.4f\tTest-Loss:\t%.4f\tTime %dmin" \
		% (epoch, avg_loss, self.test(), (time.time()-t)/60))


	print ("Recommender is built!")

    def test(self):
	"""
	:return performance on test set (Mean Square Root Error)
	"""
	tot_loss = 0.0
	for i in range(self.batch_num_test):
	    loss = self.sess.run(self.loss, feed_dict = self.get_feed_dict(self.reader.get_next_test()))
	    tot_loss += loss
	return tot_loss / self.batch_num_test

    def get_feed_dict(self, batch):
	user_ids = [l[0] for l in batch]
	prod_ids = [int(l[1]) for l in batch]
	ratings = [l[2] for l in batch]

	def manage(users):
	    l = []
	    for user in users:
		if not user in self.user_dic:
		    self.user_dic[user] = len(self.user_dic)
		l.append(self.user_dic[user])
	    return l
	user_ids = manage(user_ids)

	yr = np.zeros((self.batch_size, self.prod_num))
	ym = np.zeros((self.batch_size, self.prod_num))

	for i, (prod_id, rating) in enumerate(zip(prod_ids, ratings)):
	    yr[i][prod_id] = rating
	    ym[i][prod_id] = 1

	feed_dict = {
		self.u_ind : user_ids,
		self.y_rate : yr,
		self.y_mask : ym
	}
	return feed_dict

	
    def recommend(self, userIds, prodIds, ratings, number):

	user_num = len(set(userIds))
	example_num = sum([len(l) for l in ratings])
	
	Wp = tf.Variable(self.Wp, trainable=False)
	Wu = tf.Variable(tf.random_uniform([self.user_num, self.factor_num], -1.0, 1.0), name='Wu')
	
	y_rate = tf.placeholder('float32', shape=[user_num, self.prod_num])
	y_mask = tf.placeholder('int', shape=[user_num, self.prod_num])

	# models
	pred_y_rate = tf.matmul(Wu, tf.transpose(Wp))
	loss_m = tf.sqrt(tf.squared_difference(y_rate, pred_y_rate))
	loss_op = tf.reduce_sum(loss_m * self.y_mask)/example_num
	optimizer = tf.train.AdamOptimizer(0.01)
	train_op = optimizer.minimize(loss_op)

	#training
	self.sess.run(tf.initialize_variables([Wp, Wu]))

	yr = np.zeros((user_num, self.prod_num))
	ym = np.zeros((user_num, self.prod_num))
	for i, _ in enumerate(userIds):
	    for prodId, rating in zip(prodIds[i], ratings[i]):
		yr[i][prodId] = rating
		ym[i][prodId] = 1
	feed_dict = {y_rate : yr, y_mask : ym}
	for epoch in range(self.epoch_num):
	    _, loss = self.sess.run([train_op, loss_op], feed_dict = feed_dict)
	    print ("epoch %d loss %.3f" %(epoch, loss))
	tot_y_rate = self.sess.run(pred_y_rate, feed_dict = feed_dict)
	
	recommend_prods
	for i, y in enumerate(tot_y_rate):
	    pairs = [(prod, rate) for prod, rate in enumerate(y) \
		if prod not in prodIds[i]]
	    topPairs = sorted(pairs, key=itemgetter(1), reverse=True)[:number]
	    recommend_prods.append([i for i, _ in topPairs])

	return recommend_prods

