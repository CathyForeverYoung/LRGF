import numpy as np
import abc
import random
from sklearn import preprocessing

class AnchorSelector(object):
	def __init__(self, m,n):
		self.m = m
		self.n = n

	__metaclass__ = abc.ABCMeta
	@abc.abstractmethod
	def anchor_select(self):
		pass

class IsomorphicRWalkAnchorSelector(AnchorSelector):
	def anchor_select(self,q,TransU,TransV):
		print('begin to find anchor points',flush=True)
		#初始节点分布向量
		probU,probV = np.ones((self.m,1)),np.ones((self.n,1))	
		probU[:],probV[:] = 1/self.m,1/self.n
		
		alpha=0.8
		while True:
			probU_t = alpha*np.dot(TransU,probU) + (1-alpha)/self.m
			residual = np.sum(abs(probU-probU_t))
			probU = probU_t
			if abs(residual)<1e-8:
				break  

		while True:
			probV_t = alpha*np.dot(TransV,probV) + (1-alpha)/self.n
			residual = np.sum(abs(probV-probV_t))
			probV = probV_t
			if abs(residual)<1e-8:
				break

		uanchor = np.argsort(probU.flatten())[::-1][:q]
		vanchor = np.argsort(probV.flatten())[::-1][:q]
	
		random.shuffle(uanchor)
		random.shuffle(vanchor)
		anchors = []
		for m,n in zip(uanchor,vanchor): 
			anchors.append((m,n))

		return anchors
	
		


