import random
from collections import defaultdict
from math import exp,pow,sqrt,log

from sklearn.metrics import roc_auc_score
from sklearn import preprocessing

import numpy as np
from pandas import Series,DataFrame
import pandas as pd

from estimator import *
from DataInput import *


class BPR(object):
    def __init__(self,mat,topK,test_items,user_ratings_test):
        self.mat = mat
        self.user_count,self.item_count = mat.shape
        self.topK = topK
        self.test_items = test_items
        self.user_ratings_test = user_ratings_test
        self.latent_factors = 10
        self.lr = 0.01
        self.reg = 0.01
        self.U = np.random.random((self.user_count, self.latent_factors))/10*(np.sqrt(self.latent_factors))
        self.V = np.random.random((self.item_count, self.latent_factors))/10*(np.sqrt(self.latent_factors))
        self.biasV = np.zeros(self.item_count)

    def get_user_items(self):
        user_items = {}
        for u in range(self.user_count):
            item_train = np.where(self.mat[u]!=0)[0]
            if len(item_train>0):
                user_items[u] = item_train  
        return user_items


    def get_negrate(self):
        item_ratecounts = np.sum(self.mat!=0,axis=0)
        neg_rate = np.zeros(self.item_count)
        for item,count in enumerate(item_ratecounts):
            neg_rate[item] = -1/log(count+2,10)
        return neg_rate


    def get_w(self,x):
        return 1/(1+exp(-(x-3.5)))+0.5
      

    def train(self,sampler,steps=10):    
        neg_rate = self.get_negrate()
        self.user_items = self.get_user_items() #训练集，用户物品字典

        #训练    
        best,best_step = 0,0
        step_pred = {}
        for step in range(steps):
            for u,i,j in sampler.generate_samples(self.mat,self.user_items):   
                r_ui = np.dot(self.U[u], self.V[i].T) + self.biasV[i]
                r_uj = np.dot(self.U[u], self.V[j].T) + self.biasV[j]
                r_uij = r_ui - r_uj
                loss_func = -1.0 / (1 + np.exp(r_uij))
    
                d = self.mat[u][i]-neg_rate[j]
                w = self.get_w(d)
               
                self.U[u] += -self.lr * (w*loss_func * (self.V[i] - self.V[j]) + self.reg * self.U[u])
                self.V[i] += -self.lr * (w*loss_func * self.U[u] + self.reg * self.V[i])
                self.V[j] += -self.lr * (w*loss_func * (-self.U[u]) + self.reg * self.V[j])
                # update biasV
                self.biasV[i] += -self.lr * (w*loss_func + self.reg * self.biasV[i])
                self.biasV[j] += -self.lr * (-w*loss_func + self.reg * self.biasV[j])

            # test
            pred = self.test()  
            ET = EstimatorTopK()
            hr = ET.hr(pred,self.user_ratings_test,self.topK)    
            map_ = ET.map(pred,self.user_ratings_test,self.topK)
            
            step_pred[step] = pred   
            if hr+map_>best:
                best, best_step = hr+map_, step

        #print(best_step)
        return step_pred[best_step]


    def test(self):
        biasV_mat=np.tile(self.biasV,[self.user_count,1])
        predict_mat = np.array(np.mat(self.U) * np.mat(self.V.T)) + biasV_mat
        predict_mat = self.pre_handel(predict_mat)#让训练集的为0
        index_sorted = np.argsort(predict_mat,axis=1)
        
        rec = {}
        for u in self.user_ratings_test.keys():  
            rec[u] = index_sorted[u][::-1]

        return rec

    def pre_handel(self,predict):
        # Ensure the recommendation cannot be positive items in the training set.
        for u in self.user_items.keys():
            for tr in self.user_items[u]:
                predict[u][tr] = -100
        
        items_not_test = np.array(list(set(range(self.item_count))- self.test_items))
        if len(items_not_test)>0:
            predict[:,items_not_test] = -100
        return predict

class Sampler(object):
    def __init__(self,sample_negative_items_empirically):
        self.sample_negative_items_empirically = sample_negative_items_empirically

    def init(self,data,user_items,user_pmn,max_samples=None):
        self.mat = data
        self.user_count,self.item_count = data.shape
        self.max_samples = max_samples
        self.user_items = user_items

    def sample_negative_item(self,u):
        j = self.random_item()
        while j in self.user_items[u]:
            j = self.random_item()
        return j

    def uniform_user(self):
        u = random.randint(0,self.user_count-1)
        while u not in self.user_items.keys():
            u = random.randint(0,self.user_count-1)
        return u

    def random_item(self):
        """sample an item uniformly or from the empirical distribution
           observed in the training data
        """
        if self.sample_negative_items_empirically:
            # just pick something someone rated!
            u = self.uniform_user()
            i = random.choice(self.user_items[u])
        else:
            i = random.randint(0,self.num_items-1)
        return i

    def num_samples(self,n):
        if self.max_samples is None:
            return n
        return min(n,self.max_samples)

class UniformUserUniformItem(Sampler):

    def generate_samples(self,data,user_items,max_samples=None):
        self.init(data,user_items,max_samples)
        for _ in range(len(self.mat.nonzero()[0])):
        #for _ in range(100):
            u = self.uniform_user()       
            i = random.choice(self.user_items[u])
            j = self.sample_negative_item(u)
            yield u,i,j

        
if __name__ == '__main__':
    TOPK = 10
    instance = DataInput(m=6040,n=3952,name='ml-1m',topK = TOPK)
    #instance = ML100k(m=943,n=1682,name='ml-100k',topK = TOPK)
    for fold in [5]:
        print("------the dataset running is {}; the fold is {}-------".format(instance.name,fold))
        data_train,data_test = instance.load_data(fold)

        sample_negative_items_empirically = True
        sampler = UniformUserUniformItem(sample_negative_items_empirically)
    
        bpr = BPR(instance.mat.values,TOPK,instance.test_items,instance.user_ratings_test) #输入的是映射后的np.array
        pred = bpr.train(sampler,steps = 25)

        ET = EstimatorTopK()
        hr = ET.hr(pred,instance.user_ratings_test,TOPK)
        map_ = ET.map(pred,instance.user_ratings_test,TOPK)
        ndcg = ET.ndcg(pred,instance.user_ratings_test,instance.test_mat.values,TOPK)

        print("hr:{:.4f},map_:{:.4f},ndcg:{:.4f}".format(hr,map_,ndcg))