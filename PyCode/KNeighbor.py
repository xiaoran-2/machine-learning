
# coding: utf-8

# K近邻算法，（物以类聚，人以群分），对于给定的实例x，选出最相近的K的实例，找出类别最多的类C作为x的类别y

# In[82]:

import numpy as np
import scipy as sp

import operator


# In[174]:

class KNeighbor():
    
    trainSet = np.nan
    labels = np.nan
    
    def __init__(self):
        pass
    
    def fit(self, dataSet, labels):
        self.trainSet = np.array(dataSet)
        self.labels = np.array(labels)
        
    def L1(self,x, y):
        '''
        L1范式，哈曼顿距离
        '''
        x = np.array(x)
        y = np.array(y)
        
        return np.sum(np.abs(x-y))
        
    
    def L2(x, y):
        '''
        欧几里德距离，L2范式
        '''
        x = np.array(x)
        y = np.array(y)
        
        return np.sum((x-y)**2)**0.5
        
    def Lmax(self,x, y):
        '''
        Lmax,L无穷范式
        '''
        x = np.array(x)
        y = np.array(y)
        
        return np.max(x-y)
    
    def predict(self, x, k, distance = L2):
        '''
        预测X的类别,k
        '''
        xx = []
        if type(x[0]) is not list:
            xx.append(x)
            x = np.array(xx)
        else:
            x = np.array(x)
        ll = x.shape[0]
        result = []
#         print(x)
        for j in range(ll): #遍历所有的单个实例
        
            d = []
            length = self.trainSet.shape[0]

            for i in range(length):
                d.append(distance(self.trainSet[i], x[j]))

            dis = dict()
            for i in range(len(self.labels)):
                dis[d[i]] = self.labels[i]

#             print(d)
#             print(dis)

            sortedDis = sorted(dis.items(),
                                     key=operator.itemgetter(0), reverse=False)
#             print(sortedDis)


            kd = min(k, len(sortedDis))

            label_Dict = {}
            for i in range(kd):
                if sortedDis[i][1] not in label_Dict.keys():
                    label_Dict[sortedDis[i][1]] = 0

                label_Dict[sortedDis[i][1]] += 1

#             print("label_Dict",label_Dict)    
            labelRank = sorted(label_Dict.items(),key=operator.itemgetter(1), reverse=True)
#             print(labelRank)
        
            result.append(labelRank[0][0])

        return  result
    


# In[175]:

kNN = KNeighbor()


# In[176]:

kNN.fit([[1,1],[1,2],[2,1],[4,4],[4,5],[5,6]],[+1, +1, +1, -1, -1, -1])


# In[179]:

kNN.predict([[2,5],[1,2]],4)


# In[ ]:



