"""
在同构图上进行随机游走
"""
import numpy as np
import multiprocessing
from sklearn import preprocessing
from math import sqrt,log
import time
import os
import pickle
from pandas import Series,DataFrame
import pandas as pd
import random
import sys

from estimator import *
from anchorselector import *
from pgBPR import *


#模板模式
#工厂模式

class Template(object):
	def __init__(self, m, n, p, name,topK):
		self.m = m
		self.n = n
		self.para = p
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
	
	def anchor_select(self,data,q,TransU,TransV):
		#同构图随机游走
		rw_anchor_selector = IsomorphicRWalkAnchorSelector(self.m,self.n)
		anchors = rw_anchor_selector.anchor_select(q,TransU,TransV)
		return anchors

	def construct_isomorphic(self):
		UVmat = self.mat.values
		user_isomorphic = np.zeros((self.m,self.m))
		for u1 in range(self.m):
			for u2 in range(u1+1,self.m):
				user_isomorphic[u1][u2] = np.sum(UVmat[u1]*UVmat[u2])/sqrt(np.sum(UVmat[u1])*np.sum(UVmat[u2])+1)
				user_isomorphic[u2][u1] = user_isomorphic[u1][u2]

		item_isomorphic = np.zeros((self.n,self.n))
		for v1 in range(self.n):
			for v2 in range(v1+1,self.n):
				item_isomorphic[v1][v2] = np.sum(UVmat[:,v1]*UVmat[:,v2])/sqrt(np.sum(UVmat[:,v1])*np.sum(UVmat[:,v2])+1)
				item_isomorphic[v2][v1] = item_isomorphic[v1][v2]
		return user_isomorphic,item_isomorphic

	def preprocess(self,user_isomorphic,item_isomorphic):
		#归一化得到转移矩阵
		user_isomorphic_sum = user_isomorphic.sum(axis=0)
		item_isomorphic_sum = item_isomorphic.sum(axis=0) 
		
		#转移概率矩阵
		TransU,TransV = np.zeros((self.m,self.m)),np.zeros((self.n,self.n))
		for i in range(self.m):
			if user_isomorphic_sum[i]>0:
				TransU[:,i] = user_isomorphic[:,i]/user_isomorphic_sum[i]
		for i in range(self.n):
			if item_isomorphic_sum[i]>0:
				TransV[:,i] = item_isomorphic[:,i]/item_isomorphic_sum[i] 
		return (TransU,TransV)
		
	def random_walk(self,TransU,TransV,anchors):
		"""不是随机开始，从某个点开始"""
		print('start random walk')
		q = len(anchors)
		alpha = 0.5

		#初始节点分布矩阵
		probU,probV = np.zeros((self.m, q)),np.zeros((self.n, q))
		#重启动矩阵
		restartU,restartV = np.zeros((self.m,q)),np.zeros((self.n,q))
		for i in range(q):
			au,av = anchors[i][0],anchors[i][1]
			restartU[au][i],restartV[av][i] = 1,1
			probU[au][i],probV[av][i] = 1,1

		while True:
			probU_t = alpha*np.dot(TransU,probU) + (1-alpha)*restartU
			residual = np.sum(abs(probU-probU_t))
			probU = probU_t
			if abs(residual)<1e-8:
				break 

		while True:
			probV_t = alpha*np.dot(TransV,probV) + (1-alpha)*restartV
			residual = np.sum(abs(probV-probV_t))
			probV = probV_t
			if abs(residual)<1e-8:
				break 

		return (probU,probV)
	
	def submatrix_const(self,prob,q):
		print('start constructing submatrices')
		#分别得到用户物品的稳态概率矩阵
		probU,probV = prob
		anchor_neighuser = {} #每个锚点的用户邻域
		anchor_neighitem = {} #每个锚点的物品邻域
		
		for u in range(self.m):
			index_val = np.argsort(probU[u])[::-1][:int(q*self.para)]
			for p in index_val:
				anchor_neighuser.setdefault(p,[])
				anchor_neighuser[p].append(u)
		for v in range(self.n):
			index_val = np.argsort(probV[v])[::-1][:int(q*self.para)]
			for p in index_val:
				anchor_neighitem.setdefault(p,[])
				anchor_neighitem[p].append(v)

		#abstract local matrices
		nargs = [(anchor_neighuser[i],anchor_neighitem[i],i) for i in range(q) if (i in anchor_neighuser.keys()) and (i in anchor_neighitem.keys())]
		multiprocessing.freeze_support()
		cores = multiprocessing.cpu_count()
		pool = multiprocessing.Pool(processes=cores - 1)
		submatrice = {}
		for y in pool.imap(self.get_submat,nargs):
			submat, i = y
			submatrice[i] = submat
		pool.close()
		pool.join()

		self.submat = submatrice
					
		return anchor_neighuser,anchor_neighitem

	def local_train(self,q):
		print('start local train')
		multiprocessing.freeze_support()
		cores = multiprocessing.cpu_count()
		pool = multiprocessing.Pool(processes=cores - 1)
	
		nargs = [(i,self.submat[i],self.test_items,self.user_ratings_test,self.topK) for i in range(q)]
		rec = {}
		for y in pool.imap(train_submat, nargs):
			subrec, i = y
			rec[i] = subrec
			sys.stdout.write('have finished training for %d/%d local groups\r' % (i+1,q))
		pool.close()
		pool.join()

		return rec        

	def pred_weight(self,rec,data_train,data_test):
		user_pred = {}
		user_dropped = 0 # drop cold-start user
		a,b={},{}
		for i in self.submat:
			a[i] = np.sum(self.submat[i]!=0,axis=1) # 每个用户打分物品个数求和
			b[i] = np.sum(self.submat[i]!=0,axis=0) # 每个物品的用户个数求和

		c = np.sum(self.mat!=0,axis=1) # pd.series
		d = np.sum(self.mat!=0,axis=0)


		for u in self.user_ratings_test:
			if u not in self.user_ratings:
				user_dropped += 1
				continue	
			vote = np.zeros(self.n)
			for i in rec:	
				if u not in rec[i]:
					continue
				wa = log((a[i][u]+1)/c[u] + 1)
				for rank,v in enumerate(rec[i][u]):
					if b[i][v]==0:
						wb = 0
					else:
						wb = log((b[i][v]+1)/d[v] + 1)
					w = wa*wb
					vote[v] += w*1/log(rank+2)
			
			user_pred[u] = np.argsort(vote)[::-1][:self.topK]

		#print("the num of dropped user is",user_dropped,len(user_pred))
			
		return user_pred



	#--------------------------
	def get_submat(self,args):
		"""
		工具方法：抽取子矩阵
		"""
		anchor_neighuser,anchor_neighitem,i = args
		submat = self.mat.iloc[anchor_neighuser,anchor_neighitem]
	
		return submat,i

	def fill_pred_dict(self,dict_data,pred,test,len_q,q):
		"""
		工具方法：将预测的值填入字典
		"""
		for i in range(len(test)):
			dict_data.setdefault(test[i][0],{})
			dict_data[test[i][0]].setdefault(test[i][1],np.zeros(len_q))
			dict_data[test[i][0]][test[i][1]][q]=pred[i]

	def get_datadic(self,data):
		"""
		工具方法：构建满足需求的字典 
		"""
		true_dict={}
		for i in range(len(data)):
			uid = data[i][0]
			mid = data[i][1]
			rate = data[i][2]
			true_dict.setdefault(uid, {})
			true_dict[uid][mid] = rate
		return true_dict

	def drop(self,pred):
		for u in self.user_ratings_test:
			if len(self.user_ratings_test[u])<self.topK:
				pred.pop(u)
		print(len(self.user_ratings_test),len(pred))
		return pred

	def rec_reduce(self,rec,subtopK):
		rec_new = {}
		for i in rec:
			rec_new.setdefault(i,{})
			for u in rec[i]:
				if len(rec[i][u])>subtopK:
					rec_new[i][u] = rec[i][u][:subtopK]
				else:
					rec_new[i][u] = rec[i][u]
		return rec_new



def train_submat(args):
	"""
	工具方法：服务于并行训练
	"""
	i,submat,test_items,user_ratings_test,topK = args

	"""得到测试的数据集在该子矩阵中的映射ID"""
	subitems = np.array(submat.columns) # local id -> global id
	item_map_dict = {} # global id -> local id
	subitems_test_map = set() # test item list with local id 
	for v_map,v in enumerate(subitems):   
		item_map_dict[v] = v_map
		if v in test_items:
			subitems_test_map.add(v_map) # 得到该子矩阵中测试物品

	"""得到该子矩阵中的用户"""
	subusers = np.array(submat.index) # local id -> global id
	subuser_ratings_test = {} # user local id - [test item local id,...]
	for u_map,u in enumerate(subusers): 
		if u in user_ratings_test: #如果这个用户需要测试
			subuser_ratings_test.setdefault(u_map,set())
			for v in user_ratings_test[u]: #对于每个测试物品
				if v in subitems: #如果它在该子矩阵里面
					subuser_ratings_test[u_map].add(item_map_dict[v]) #就把他在子矩阵的ID放进测试用户的子矩阵ID中

	"""选择采样器"""
	sample_negative_items_empirically = True
	sampler = UniformUserUniformItem(sample_negative_items_empirically)
	bpr = BPR(submat.values,topK,subitems_test_map,subuser_ratings_test) #输入的是映射后的np.array
	rec_map = bpr.train(sampler,steps = 22)

	"""得到映射回大矩阵的推荐列表"""
	rec = {}
	for u_map in rec_map:
		u = subusers[u_map]
		rec[u] = subitems[rec_map[u_map]]

	return (rec,i)

class ML100k(Template):

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


if __name__=='__main__':
	p = 0.7 
	q = 50 #论文里说是50
	fold = 1
	topk =10
	
	subtopK = topK
	#instance = ML100k(m=943,n=1682,p=p,name='ml-100k',topK=topK)
	instance = Template(m=6040,n=3952,p=p,name='ml-1m',topK=topK)
	print("the dataset running is",instance.name,";the fold is",fold,";top",topK)
	

	data_train,data_test = instance.load_data(fold)	
	# 构建同构图,并存储
	user_isomorphic,item_isomorphic = instance.construct_isomorphic()

	#特征提取,得到状态转移矩阵
	TransU,TransV = instance.preprocess(user_isomorphic,item_isomorphic)
	#选择锚点
	anchors = instance.anchor_select(data_train,q,TransU,TransV)		
	#随机游走寻找邻域
	anchorM = instance.random_walk(TransU,TransV,anchors)
	#得到以每个锚点为中心子矩阵所包含的训练集和测试集
	anchor_neighuser,anchor_neighitem = instance.submatrix_const(anchorM,q)

	# 训练
	rec = instance.local_train(q) #每个子矩阵经过训练，生成推荐列表

	#将推荐列表的长度缩减成预备长度
	rec = instance.rec_reduce(rec,subtopK) # local results



	###########
	# fusion
	pred = instance.pred_weight(rec,data_train,data_test) 


	###########
	ET = EstimatorTopK()
	hr = ET.hr(pred,instance.user_ratings_test,instance.topK)	
	map_ = ET.map(pred,instance.user_ratings_test,instance.topK)
	ndcg = ET.ndcg(pred,instance.user_ratings_test,instance.test_mat.values,instance.topK)
	print("------------------map_:{:.4f},ndcg:{:.4f},hr:{:.4f}--------------------".format(map_,ndcg,hr))