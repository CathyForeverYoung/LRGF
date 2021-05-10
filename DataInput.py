import numpy as np
from sklearn import preprocessing
from math import sqrt,log
from pandas import Series,DataFrame
import pandas as pd
import random


class DataInput(object):
	def __init__(self, m, n, name,topK):
		self.m = m
		self.n = n
		self.name = name
		self.topK = topK

	def load_data(self,fold=1):
		path_train = self.name + '/data/train'+ str(fold) + '.txt'
		path_test = self.name + '/data/test'+ str(fold) + '.txt'
		with open(path_train,'r') as f:
			data_train = eval(f.read())
		with open(path_test,'r') as f:
			data_test = eval(f.read())
		
		"""评分矩阵、训练物品集合、测试物品集合"""
		user_ratings = {}
		train_mat = np.zeros((self.m,self.n)) 
		train_items = set()
		for u,v,r in data_train:
			user_ratings.setdefault(u,set())
			user_ratings[u].add(v)
			train_mat[u,v] = r
			train_items.add(v)
		self.mat = DataFrame(train_mat) #训练集构成的额评分矩阵
		self.train_items = train_items  #训练物品集合

		user_ratings_test = {}
		test_mat = np.zeros((self.m,self.n)) 
		test_items = set()
		for u,v,r in data_test:
			user_ratings_test.setdefault(u,set())
			user_ratings_test[u].add(v)
			test_mat[u,v] = r
			test_items.add(v)
		self.test_mat = DataFrame(test_mat)
		self.test_items = test_items    #测试物品集合

		#drop 如果用户在测试集中的物品个数<topK,则丢掉他
		print("原始测试用户",len(user_ratings_test))	
		dropped_users = []
		for u in user_ratings_test:
			if len(user_ratings_test[u])<self.topK:
				dropped_users.append(u)
		for u in dropped_users:
			user_ratings_test.pop(u)		
		print("实际测试用户",len(user_ratings_test))

		self.user_ratings = user_ratings
		self.user_ratings_test = user_ratings_test


		return (data_train,data_test)


class ML100k(DataInput):

	def load_data(self,fold=1):
		print("load data")
		data_train,data_test = [],[]
		path_train = self.name + '/u'+ str(fold) + '.base'
		path_test = self.name + '/u'+ str(fold) + '.test'
		
		with open(path_train) as f:
			for line in f:
				a = line.split("\t")
				data_train.append((int(a[0]) - 1, int(a[1]) - 1, int(a[2])))  # 此处已经把id变成索引（-1）
		with open(path_test) as f:
			for line in f:
				a = line.split("\t")
				data_test.append((int(a[0]) - 1, int(a[1]) - 1, int(a[2])))  # 此处已经把id变成索引（-1）
		
		"""评分矩阵、训练物品集合、测试物品集合"""
		user_ratings = {}
		train_mat = np.zeros((self.m,self.n)) 
		train_items = set()
		for u,v,r in data_train:
			user_ratings.setdefault(u,set())
			user_ratings[u].add(v)
			train_mat[u,v] = r
			train_items.add(v)
		self.mat = DataFrame(train_mat) #训练集构成的额评分矩阵
		self.train_items = train_items  #训练物品集合

		user_ratings_test = {}
		test_mat = np.zeros((self.m,self.n)) 
		test_items = set()
		for u,v,r in data_test:
			user_ratings_test.setdefault(u,set())
			user_ratings_test[u].add(v)
			test_mat[u,v] = r
			test_items.add(v)
		self.test_mat = DataFrame(test_mat)
		self.test_items = test_items    #测试物品集合			

		#drop
		print("原始测试用户",len(user_ratings_test))	
		dropped_users = []
		for u in user_ratings_test:
			if len(user_ratings_test[u])<self.topK:
				dropped_users.append(u)
		for u in dropped_users:
			user_ratings_test.pop(u)		
		print("实际测试用户",len(user_ratings_test))

		self.user_ratings = user_ratings
		self.user_ratings_test = user_ratings_test

		return data_train,data_test