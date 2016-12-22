"""
Created on Thu Oct 27 2016

@author: Sewon
"""

import numpy as np
import time

# List of recommenders
from model.ContentBasedRecommender import ContentBasedRecommender
from model.CollaboFilterRecommender import CollaboFilterRecommender
from model.LearnedRecommender import LearnedRecommender
from model.LatentFactorRecommender import LatentFactorRecommender
from model.BiasLatentFactorRecommender import BiasLatentFactorRecommender

class Evaluator(object):
    def __init__(self, recom, hidden, small, batch_size):
	self.recommender_dic = {
		'cb' : ContentBasedRecommender,
		'cf' : CollaboFilterRecommender,
		'l' : LearnedRecommender,
		'lf' : LatentFactorRecommender,
		'blf' : BiasLatentFactorRecommender
	}
	self.recom_list = self.recommender.keys() if recom == 'all' else recom
	self.recommenders = {}
	self.hidden_list = hidden
	self.batch_size = batch_size
	self.small = small
	print ("Evaluator of %s Dataset with batch size %d" \
		%('small' if small else 'large', batch_size))

    def evaluate(self):
	configs = []
	for recom in self.recom_list:
	    if recom == 'lf' or recom == 'blf':
		for h in self.hidden_list:
		    configs.append([recom, h])
	    else:
		configs.append(recom)
	for config in configs:
	    self.single_eval(config)

    def single_eval(self, config):
	if type(config) == list:
	    recommender = self.recommender_dic[config[0]](self.small, self.batch_size, config[1])
	    config = config[0]+"("+str(config[1])+" factors)"
	else:
	    recommender = self.recommender_dic[config](self.small, self.batch_size)
	print "Start Building Recommender System [%s]" %(config)
	t1 = time.time()
	recommender.build()
	t2 = time.time()
	print "Start Evaluating Performance"
	error = recommender.test()
	t3 = time.time()
	build_time = (t2-t1)/60
	eval_time = (t3-t2)/60
	print ("Recommender [%s]\t: Error : %.4f (build %.3fmin eval %.3fmin)" \
		%(config, error, build_time, eval_time))
	self.recommenders[config] = recommender

    def run_demo(self):
	print ("Which recommender systems do you want?")
	print ("Enter numbers with space")
	for i, name in enumerate(self.recommenders.keys()):
	    print "[%d] %s" %(i, name)
	req = raw_input()
	reqlist = [int(word) for word in req.split(' ')]
	print ("How many products you want?")
	number = raw_input()
	print ("How many users?")
	user_num = raw_input()
	userIds, prodIds, ratings = [], [], []
	for i in range(1, user_num+1):
	    uid = "[User %d]" %i
	    plist, rlist = [], []
	    print ("Enter Previous Rating of %s" %uid)
	    print ("format : prodId1 rating1 [tab] prodId2 rating2 [tab] ...")
	    line = raw_input()
	    phases = line.split('\t')
	    for phase in phases:
		words = phase.split(' ')
		plist.append(words[0])
		plist.append(words[1])
	    userIds.append(uid)
	    prodIds.append(plist)
	    ratings.append(rlist)
	print ("Running Recommender Systems for %d Users ..." % user_num)
	for i in reqlist:
	    recommend_prods = self.recommenders.values()[i].recommend( \
		userIds, prodIds, ratings, number)
	    print ("%s :")
	    for prods in recommend_prods:
		print prods

