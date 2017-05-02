
# coding: utf-8

# In[101]:

import os
from numpy import *


# In[336]:

# 创建Bayes类
class Bayes:
    trainSet = None 
    labels = None
    allWordsSet = None # 训练集中所有出现的单词
    
    # 先验情况下得到的概率
    p0 = None
    p1 = None
    pClass1 = None
    
    def __inint__():
        pass

    # 格式化读取文件
    def textParse(self,bigString):    #input is big string, #output is word list
        import re
        listOfTokens = re.split(r'\W*', bigString)
        return [tok.lower() for tok in listOfTokens if len(tok) > 2] 

    # 得到所有的单词
    def createVocabList(self,dataSet):
        allWordSet = set([])  #create empty set
        for document in dataSet:
#             print(document)
            allWordSet = allWordSet | set(document) #union of the two sets
        return list(allWordSet)
    
    
    def loadDataSet(self):
        trainSet= [] # 前面是训练数据，后面是lable
        labels = []
        #读取所有的ham文件,文件名字已经格式化为i.ham.txt
        for i in range(1,3014):
            if os.path.isfile('ham/'+str(i)+'.ham.txt') == False: #没有这个路径
                continue
            try:
                wordList = textParse(open('ham/'+str(i)+'.ham.txt').read())
                trainSet.append(wordList)
                labels.append(1)
    #         print(i)
            except UnicodeEncodeError:
                pass

        #读取所有的spam文件,文件名字已经格式化为i.spam.txt
        for i in range(1,1127):
            if os.path.isfile('spam/'+str(i)+'.spam.txt') == False: #没有这个路径
                continue
            try:

                wordList = textParse(open('spam/'+str(i)+'.spam.txt').read())
                trainSet.append(wordList)
                labels.append(0)
    #             print(i)
            except UnicodeDecodeError:
    #             os.remove('spam/'+str(i)+'.spam.txt')
                pass
            
        # 得到训练集合和所有单词
        self.trainSet = trainSet
        self.labels = labels
        
        self.allWordsSet = self.createVocabList(trainSet)
        
        return trainSet,labels
    
    # 词袋模型
    def bagOfWords2Vec(self,inputSet):
        returnVec = [0 for i in range(0,len(self.allWordsSet))]
        
        for word in inputSet:
            if word in self.allWordsSet:
                returnVec[self.allWordsSet.index(word)] += 1

        return returnVec

    # 训练分类器
    def fit(self,trainData,trainCategory):
        trainMatrix = []
        #这儿最费时，
        for doc in trainData:
            trainMatrix.append(self.bagOfWords2Vec(doc))

        numTrainDocs = len(trainMatrix)
        numWords = len(trainMatrix[0])
        
        pAb = sum(trainCategory) / float(numTrainDocs)
        p0 = ones(numWords); p1 = ones(numWords)      #所有次初始化1，
        p0Denom = 2.0; p1Denom = 2.0                        #分母初始化2
 
        for i in range(numTrainDocs):
            if trainCategory[i] == 1:
                p1 += trainMatrix[i]
                p1Denom += sum(trainMatrix[i])
            else:
                p0 += trainMatrix[i]
                p0Denom += sum(trainMatrix[i])
        p1V = log(p1 / p1Denom)          #change to log()
        p0V = log(p0 / p0Denom)          #change to log()
        
        self.p0 = p0V
        self.p1 = p1V
        self.pClass1 = pAb
        
        return p0V,p1V,pAb
    
    # 预测X,X是个Numpy列表，或者是个文件名
    def predict(self,X):
        if type(X) != list and os.path.isfile(X): #X是个文件名,转化为numpy列表
            X = textParse(open(X).read())
        
#         print(X)
        # 转化为概率
        px = self.bagOfWords2Vec(X)
#         print(px)
        p1 = sum(px * self.p1) + log(self.pClass1)    #element-wise mult
        p0 = sum(px * self.p0) + log(1.0 - self.pClass1)
        if p1 > p0:
            return 1
        else: 
            return 0
    
    
    def predictall(self, testSet):
        res = []
        for x in testSet:
            res.append(self.predict(x))
            
        return res
        
    
    # k折交叉得到的数据，
    def kfold(self,k,seed=0):
        '''
        k分数据，seed是一个小于k的值，得到的第seed份数据进行测试，其余作为训练
        '''
        from random import shuffle # 用来打乱数据
        length = len(self.trainSet)
        sign = length // k

        trainSetk = []
        labelsk = []
        index = [i for i in range(length)] #得到所欲数据的下标，打乱下标，之后讲数据等分成k分
        shuffle(index)
        vis = [0 for i in range(k)]
#         print(vis)
        vis[0] = 1
        tmp1 = []       
        tmp2 = []

        for i in range(length-length%k):
    #         print(i // sign)
            if vis[i // sign] == 0:
                vis[i // sign] = 1
                trainSetk.append(tmp1)
                labelsk.append(tmp2)
                tmp1 = []
                tmp2 = []

            tmp1.append(self.trainSet[index[i]])
            tmp2.append(self.labels[index[i]])
            
        #得到最后的数据
        trainSetk.append(tmp1)
        labelsk.append(tmp2)
        
#         for i in range(k):
#             print(len(trainSetk[i]))
        
        newtrinSet = []
        newtrainSet_labels = []
        
        for i in range(k):
            if i != seed:
                newtrinSet.extend(trainSetk[i])
                newtrainSet_labels.extend(labelsk[i])
        
        test = trainSetk[seed]
        testlabels = labelsk[seed]
        
        return newtrinSet, newtrainSet_labels, test, testlabels
    
    
    def culprecise(self,predict_res, test_labels):
        '''
        计算正确率
        ''' 
        return sum(array(predict_res) == array(test_labels)) / len(test_labels)
        
        
    def culrecall(self,predict_res, test_labels):
        '''
        计算召回率
        '''
        k = 0
        for i in predict_res:
            if i in test_labels:
                k += 1
        
        return k / len(test_labels)
        
    


# In[337]:

bayes = Bayes()


# In[338]:

trainSet,labels = bayes.loadDataSet()
train,train_labels, test, test_labels = bayes.kfold(5)


# In[340]:

bayes.fit(train[:],train_labels[:]) #训练时间费时


# In[345]:

bayes.predict('spam/1.spam.txt') 


# In[342]:

ans = bayes.predictall(test)


# In[343]:

preciserate = bayes.culprecise(ans,test_labels)
racallrate = bayes.culrecall(ans,test_labels)


# In[344]:

print('preciserate=',preciserate)
print('recallrate=',racallrate)


# In[102]:



