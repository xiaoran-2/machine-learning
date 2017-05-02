
# coding: utf-8

# 朴素贝叶斯的分类实现

# In[2]:

import numpy as np
import scipy as sp


# In[5]:

class NaiveBayes():
    classDict = {} #每一类的个数
    featureDict = {} #在每一类中的所有特征的次数，使用拉普拉斯平滑，<keys,values> == <classID, featureVector>
    
    proClassDict = {} #计算概率时，使用拉普拉斯平滑
    profeatureDict = {} 
    
    def __init__(self):
        pass
     
    def fit(self, x, y):
        """
        计算先验概率和条件概率
        """
        for i in range(len(y)):
            if y[i] not in self.classDict.keys():
                self.classDict[y[i]] = 0;
            self.classDict[y[i]] += 1;
        
        for index in self.classDict.keys(): #循环多，效率低
            counti = {} #每一个位置i设置一个字典，记录该位置上的所有可能的情况出现的次数
            for i in range(len(y)):
                if y[i] == index:
                    for j in range(len(x[i])):
                        count = {}# 初始化所有的特征次数
                        if x[i][j] not in count.keys():
                            count[x[i][j]] = 0
                            
                        count[x[i][j]] += 1 #记录每个特值的个数
                
            
            self.featureDict[index] = count
            
            procount = {}
            for feature in count.keys():
                procount[feature] = (count[feature] + 1) / (classDict[index] + len(x[i]))
                
            
            #利用极大似然计算概率
            self.proClassDict[index] = (classDict[index] + 1) / len(y)
            self.profeatureDict[index] = procount
            
    
    def __calProbility(self,x):
        
        
    
    
    def predict(self, x):
        xx = []
        if type(x[0]) is not list:
            xx.append(x)
            x = np.array(xx)
        else:
            x = np.array(x)
        ll = x.shape[0]
        result = []
        
        for j in range(ll):
            

    
    
    


# In[ ]:



