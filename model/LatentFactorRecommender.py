"""
Created on Thu Oct 27 2016

@author: Sewon
"""

from model.BaseRecommender import BaseRecommender


class LatentFactorRecommender(BaseRecommender):
    def __init__(self, small, batch_size, factor_num):
	BaseRecommender.__init__(self, small, batch_size)
	self.user_num = self.reader.get_user_num()
	self.prod_num = self.reader.get_prod_num()
	self.batch_num = self.reader.get_batch_num()
	self.factor_num = factor_num


    def build(self):
	"""
	build recommender system
	"""

	# Variables

	self.Wp = tf.Variable(tf.random_uniform([self.prod_num, self.factor_num], -1.0, 1.0), name='Wp')
	self.Wu = tf.Variable(tf.random_uniform([self.user_num, self.factor_num], -1.0, 1.0), name='Wu')
	
	# Placeholders

	
	self.u_ind = tf.placeholder('int', shape=[self.batch_size])
	self.y_rate = tf.placeholder('float32', shape=[self.batch_size, self.prod_num])
	self.y_mask = tf.placeholder('int', shape=[self.batch_size, self.prod_num])

	# models

	for i in range(self.batc_size):
	    currU = tf.gather(Wu, i)
	    if i==0: U=currU
	    else: U=tf.concat(0, [U, currU])
	# U : [batch_size, factor_num]
	# Wp : [prod_num, factor_num]
	self.prod_y_rate = tf.matmul(U, tf.transpose(Wp))
	loss_m = tf.sqrt(tf.squared_difference(self.y_rate, self.pred_y_rate))
	self.loss = tf.reduce_sum(loss_m * self.y_mask)/self.batch_size
	optimizer = tf.train.AdamOptimizer(0.01)
	train_op = optimizer.minimize(loss)

	#training
	self.sess = tf.Session()
	self.sess.run(tf.initialize_all_variables())

	for epoch in range(epoch_num):
	    tot_loss = 0.0
	    for i in range(self.batch_num):
		_, loss = self.sess.run([train_op, self.loss], feed_dict = self.get_feed_dict(self.reader.get_next_train()))
		tot_loss += loss
	    avg_loss = tot_loss / self.batch_num
	    print ("Epoch %d : Loss : %.2f" % (epoch, avg_loss))

	print ("Recommender is built!")

    def test(self):
	"""
	:return performance on test set (Mean Square Root Error)
	"""
	tot_loss = 0.0
	for i in range(self.bach_num):
	    loss = self.sess.run(self.loss, feed_dict = self.get_feed_dict(self.reader.get_next_test()))
	    tot_loss += loss
	return tot_loss / self.batch_num

    def get_feed_dict(self, user_ids, prod_ids, ratings):
	user_ids = list(batch['uid'])
	prod_ids = list(batch['pid'])
	ratings = list(batch['score'])

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
	for epoch in range(20):
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

