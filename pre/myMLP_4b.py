# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 17:20:09 2020

@author: sshaf
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 12:44:49 2020

@author: sshaf
"""
import numpy as np
import matplotlib.pyplot as plt

def tanh(x):
    t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    return t

def d_tanh(x):
    t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    dt=1-t**2
    return dt

def sigmoid(x):
    return 1/(1+np.exp(-x))
#    r=np.zeros(x.shape)
#    for i in range(x.shape[1]):
#        r[0,i]= 1 / (1 + math.exp(-x[0,i]))
#    return r

def d_sigmoid(x):
    return np.exp(-x)/((1+np.exp(-x))**2)
#    r=np.zeros(x.shape)
#    for i in range(x.shape[1]):
#        r[0,i]= math.exp(-x[0,i])/((1+math.exp(-x[0,i]))**2)
#    return r

def sigmoid_derivative(p):
    return p * (1 - p)

class NeuralNetwork:
    def __init__(self, x, y,hiddenNode,epoch):
        
        self.input      = x
        self.weights1   =np.random.rand(self.input.shape[1],hiddenNode) 
        self.weights2   =np.random.rand(hiddenNode,1)                 
        self.y          = y
        self.output     = np.zeros(self.y.shape)
        self.alpha      = 0.01
        self.epoch      = epoch
       


    def feedforward(self):
        self.z2=np.dot(self.input, self.weights1)
        self.y_hat2 = tanh(self.z2) #sigmoid(self.z2)
        self.z3=np.dot(self.y_hat2, self.weights2)
        self.y_hat3 = tanh(self.z3)#sigmoid(self.z3)
        self.error=1/self.input.shape[0] * sum(0.5 *(self.y-self.y_hat3)**2)
        #print(self.y) 
        plt.figure()
#        plt.plot(self.z3,'green')
#        plt.plot(abs(self.y-self.y_hat3),'red')
#        plt.plot(self.y,'blue')
        #print(self.y-self.y_hat3)
                
    def backproppagation(self):
        
        #d_weights2 = np.dot(self.y_hat2.T, 2*self.alpha*(self.y -self.y_hat3)*sigmoid_derivative(self.y_hat3))
        #d_weights1 = np.dot(self.input.T, np.dot(self.alpha*2*(self.y -self.y_hat3)*sigmoid_derivative(self.y_hat3), self.weights2.T)*sigmoid_derivative(self.y_hat2))

#        d_weights2 = np.dot(self.y_hat2.T, (self.alpha*(self.y - self.y_hat3) * d_sigmoid(self.z3)))
#        d_weights1 = np.dot(self.input.T,  np.dot(self.alpha*(self.y - self.y_hat3) * d_sigmoid(self.z3), self.weights2.T) * d_sigmoid(self.z2))
        d_weights2 = np.dot(self.y_hat2.T, (self.alpha*(self.y - self.y_hat3) * d_tanh(self.z3)))
        d_weights1 = np.dot(self.input.T,  np.dot(self.alpha*(self.y - self.y_hat3) * d_tanh(self.z3), self.weights2.T) * d_tanh(self.z2))
        
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def fit(self):
        err= np.zeros(self.epoch)
        for k in range(self.epoch):
            self.feedforward()
            self.backproppagation()
            err[k]=self.error
        real_out=np.zeros(len(self.y_hat3))
        for i in range(len(self.y_hat3)):
             if self.y_hat3[i]>0.5:
                 real_out[i]=1
#        print("&&&&&&&&&&&&&&&&&&")
#        print(real_out)
        plt.figure()
        plt.plot(err,'b')
        return err    
        
    def test(self,test_data,test_output):
         z2=np.dot(test_data, self.weights1)
         y_hat2 = sigmoid(z2)
         z3=np.dot(y_hat2, self.weights2)
         y_hat3 = sigmoid(z3)
         error=1/test_data.shape[0] * sum(0.5 *(test_output-y_hat3)**2)
         for i in range(len(y_hat3)):
             if y_hat3[i]<0.5:
                 y_hat3[i]=0
             else:
                 y_hat3[i]=1
                 
         #print(y_hat3)
         plt.figure()
         for i in range(len(y_hat3)) :
             if abs(test_output[i]-y_hat3[i])<0.5:
                 plt.scatter(test_data[i,0],test_data[i,1],c='green')
                 
             else:
                 plt.scatter(test_data[i,0],test_data[i,1],c='red')                 
         return error
     
        
        
def dataGeneration1(mu, sigma, num ):
    n=0
    x=np.zeros((num,2))
    while n<num:
        r1=np.random.normal(mu, sigma , 1)
        r2=np.random.normal(mu, sigma , 1)
        if (r1**2 + r2**2)<1 and r1>=0 and r2>=0 :
            x[n,0]=r1
            x[n,1]=r2
            n+=1
    return x         
    
def dataGeneration2(mu, sigma, num ):
    n=0
    x=np.zeros((num,2))
    while n<num:
        r1=np.random.normal(mu, sigma , 1)
        r2=np.random.normal(mu, sigma , 1)
        if (r1**2 + r2**2)>=1 and 0<=r1<=1 and 0<=r2<=1:
            x[n,0]=r1
            x[n,1]=r2
            n+=1
    return x     
    
#data generation
x1=dataGeneration1(0.5, 0.08**0.5, 100)
shp=(200,2)
d=np.zeros(shp)
shp=(200,1)
target=np.zeros(shp)
plt.scatter(x1[:,0],x1[:,1],c='red')
for i in range (100):
    d[i,0]=x1[i,0]
    d[i,1]=x1[i,1]
    target[i,0]=0


x2=dataGeneration2(0.75, 0.08**0.5, 100)
plt.scatter(x2[:,0],x2[:,1],c='blue')
for i in range (100):
    d[i+100,0]=x2[i,0]
    d[i+100,1]=x2[i,1]
    target[i,0]=1

   #devide data to test and train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(d, target, test_size=0.2)

   # learn the model:
plt.figure()
for i in range(len(y_test)) :
    if y_test[i]==0:
        plt.scatter(X_test[i,0],X_test[i,1],c='green')
    else:
        plt.scatter(X_test[i,0],X_test[i,1],c='blue')
                 


    