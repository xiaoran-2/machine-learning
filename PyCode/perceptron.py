
# coding: utf-8

# In[4]:

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


# In[71]:

class Perceptron:
    trainSet = np.nan
    labels = np.nan
    
    w = 0
    b = 0

    
    def __init__(self):
       pass
    
    def plot(self):
        '''
        显示每一次的直线的位置
        '''
        
        for i in range(self.trainSet.shape[0]):
            if self.labels[i] == 1:
                plt.scatter(self.trainSet[i][0],self.trainSet[i][1],marker = 'o', s=200, c='red')
            else:
                plt.scatter(self.trainSet[i][0],self.trainSet[i][1],marker = 'x', s=200, c='black')
        
        x = np.arange(0, 5)
        y = -(x * self.w[0] + self.b) / self.w[1]
        plt.plot(x,y,c = 'blue')
        
        plt.show() # 显示训练的过程
    
    
    def plotAll(self):
        w = np.zeros(self.trainSet.shape[1])
        b = 0
        
        learnRate = 1
        flag = True
        length = self.trainSet.shape[0]
        
        i = 0
        while flag:
            flag = False
            for i in range(length):
                t = self.labels[i] * (np.sum(self.trainSet[i] * w) + b)
#                 print(t)
                if t <= 0: #错误分类的点，修改权重
                    w = w + learnRate * self.labels[i] * self.trainSet[i]
                    b = b + learnRate * self.labels[i]
                    flag = True
            print(w,b)
            
            i += 1
        # 修改类的值
            self.w = w
            self.b = b
             # plot
            for i in range(self.trainSet.shape[0]):
                if self.labels[i] == 1:
                    plt.scatter(self.trainSet[i][0],self.trainSet[i][1],marker = 'o', s=200, c='red')
                else:
                    plt.scatter(self.trainSet[i][0],self.trainSet[i][1],marker = 'x', s=200, c='black')

            x = np.arange(0, 5)
            y = -(x * self.w[0] + self.b) / self.w[1]
            plt.plot(x,y,c = 'blue')
            
        plt.show()
    
    
    def loadDataSet(self,trainData,labels):
        '''
        格式化训练数据
        '''
        return np.array(trainData), np.array(labels)



    def fit(self,trainData,label,learnRate=1):
        '''
        原始训练数据和labels，以及学习率(0 < learnRate <= 1)
        '''
        self.trainSet = np.array(trainData)
        self.labels = np.array(label)
        
        
        w = np.zeros(self.trainSet.shape[1])
        b = 0

        flag = True
        length = self.trainSet.shape[0]

        while flag:
            flag = False
            for i in range(length):
                t = self.labels[i] * (np.sum(self.trainSet[i] * w) + b)
#                 print(t)
                if t <= 0: #错误分类的点，修改权重
                    w = w + learnRate * self.labels[i] * self.trainSet[i]
                    b = b + learnRate * self.labels[i]
                    flag = True
#             print(w,b)
        # 修改类的值
            self.w = w
            self.b = b
            
#             self.plot()
        return w, b


    def predict(self,testSet):
        '''
        测试数据
        '''
        data = []
        if type(testSet[0]) is not list:
            data.append(testSet)
            
            testSet = np.array(data)
        else:
            testSet = np.array(testSet)
            
#         print(testSet, testSet.shape[0])
        result = []
        for i in range(testSet.shape[0]):
            a = np.sign(np.sum(self.w * testSet[i]) + self.b)
#             print(a,np.sum(self.w * testSet[i]) + self.b)
            result.append(a)
        
        return result


# In[72]:

perceptron =  Perceptron()
# [[1,1],[1,2],[2,1],[4,4],[4,5],[5,6]],[+1, +1, +1, -1, -1, -1]


# In[73]:

perceptron.fit([[1,1],[1,2],[2,1],[4,4],[4,5],[5,6]],[+1, +1, +1, -1, -1, -1])


# In[74]:

# perceptron.plotAll()


# In[76]:

perceptron.predict([[2,2],[1,2],[4,3]])


# In[ ]:




# In[ ]:



