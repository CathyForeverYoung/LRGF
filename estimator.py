from sklearn.metrics import mean_squared_error,mean_absolute_error
from math import sqrt,log
import numpy as np


class EstimatorTopK(object):
    def pn_recall(self,u_rec,u_true,k):
        """
        urec:dict,每个测试集用户的推荐列表集合
        u_test:dict,每个测试集用户真实的列表集合
        """    
        hit = recall = pn = 0 
        for u in u_rec:
            hit += len(set(u_rec[u][:k]) & u_true[u])
            recall += len(u_true[u])
            pn += k
        pn = hit/pn
        recall = hit/recall
        return pn,recall

    def Micro_pn_recall(self,u_rec,u_true,k):
        """
        urec:dict,每个测试集用户的推荐列表集合
        u_test:dict,每个测试集用户真实的列表集合
        """    
        hits = recalls = pns = 0 

        for u in u_rec:
            hit = len(set(u_rec[u][:k]) & u_true[u])
            recall = hit/len(u_true[u])
            pn = hit/k
            hits+=hit
            recalls+=recall
            pns+=pn
            
        pn = round(pns/len(u_rec),4)
        recall = round(recalls/len(u_rec),4)
        return pn,recall

    def hr(self,u_rec,u_true,k):
        hitu = 0
        num_u = 0
        for u in u_rec:
            if len(set(u_rec[u][:k])&u_true[u])>0:
                hitu += 1
            num_u += 1
        return hitu/num_u
    
    def map(self,u_rec,u_true,k):
        mean_ap = 0
        a = []
        for u in u_rec:
            hit = 0
            ap = 0
            for kk in range(k):
                if u_rec[u][kk] in u_true[u]:
                    hit+=1
                    ap+=hit/(kk+1)
            if hit!=0:
                ap/=hit
            else:
                ap=0
            mean_ap += ap
            a.append(ap)
        mean_ap = mean_ap/len(u_rec)
        return mean_ap

    def DCG(self,u_rec,u_true,k):
        dcg = 0
        for u in u_rec:
            u_dcg = 0
            for kk in range(k):
                if u_rec[u][kk] in u_true[u]:
                    u_dcg += 1/np.log2(kk+2)
            dcg += u_dcg
        dcg = dcg / len(u_rec)
        return dcg

    def ndcg(self,u_rec,u_true,mat,k):
        ndcg = 0
        for u in u_rec:
            u_dcg = 0
            true_list = []
            
            #DCG
            for kk in range(k):
                true_list.append((u_rec[u][kk],mat[u][u_rec[u][kk]]))
                u_dcg += (pow(2,mat[u][u_rec[u][kk]])-1)/np.log2(kk+2)
            
            #IDCG
            idcg = 0
            true_list = sorted(true_list,key=lambda x:x[1],reverse=True)
            for kk in range(k):
                idcg += (pow(2,true_list[kk][1])-1)/np.log2(kk+2)
            
            #NDCG
            if u_dcg>0:
                u_ndcg = u_dcg/idcg
                ndcg += u_ndcg

        ndcg = ndcg / len(u_rec)
        return ndcg