#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 09:37:08 2018

@author: patrickorourke
"""

# Assignment for the dataset "Auto MPG"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr   
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

learning_rate = 0.003
EPOCH = 750

# update_count used for the number of iterations in each epoch before gradients 
# applied to weights for Stocahstic Gradient descent
update_count = 2000

# STEP 1 - GATHERING DATA

file = "/Users/patrickorourke/Documents/letter-recognition/letter-recognition.data.txt"
# Label the columsn of the Pandas DataFrame
columns = ['Letter', 'X-Box', 'Y-Box', 'Width', 'High', 'onpix', 'x-bar', 'y-bar', 'x2bar','y2bar','xybar','x2ybr','xy2br',
               'x-ege','xegvy','y-ege','yegvx']

# Function to read textfile dataset and load it as a Pandas DataFrame
def loadData(file,columns):
    df = pd.read_table(file, sep=',')
    df.columns = columns
    return df

def correlation(data):
    correlation = []
    for i in range(0,7):
        j = pearsonr(data.iloc[:,i],data.iloc[:,9])
        correlation.append(j)
    return correlation

def charToOneHot(s):
        oneHot = [0 for _ in range(len(labels))]
        oneHot[labels.index(s.lower())] = 1
        return oneHot

class NNSoftClassif:
    
    def __init__(self, ins, hids, outs ):
        self.input_size = ins
        self.hidden_size = hids
        self.output_size = outs  
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.W1_grads = np.zeros(shape=(self.input_size, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) 
        self.W2_grads = np.zeros(shape=(self.hidden_size, self.output_size))
        # Biases
        self.IH_bias = np.zeros((1, self.hidden_size))    # Input -> Hidden
        self.HO_bias = np.zeros((1, self.output_size))   # Hidden -> Output
        # Biases gradients computed during backprop
        self.IH_b_gradients = np.zeros_like(self.IH_bias)
        self.HO_b_gradients = np.zeros_like(self.HO_bias)
    
    # Sigmoid activation function
    def sigmoid(self, z):
        return 1/(1+np.exp(-z) + 1e-10)
    
    # Softmax activation function
    def softmax(self, z):
        exps = np.exp(z - np.max(z))
        return exps / (np.sum(exps) + 1e-10)
    
    # derivative of sigmoid
    def sigmoidPrime(self, s):
        return s * (1 - s)

    # derivative of softmax
    def softmaxPrime(self, s):
        return 1
    
    def crossEntropyPrime(self, y,outputs):
        return y - outputs

    # forward propagation through our network
    def forward(self, X):
        
        # dot product of X (input) and first set of weights
        self.inputs = X.reshape(1, self.input_size)
        z = np.dot(self.inputs, self.W1) + self.IH_bias
        # activation function
        self.hidden = self.sigmoid(z)
        # dot product of hidden layer (forw1) and second set of 4x1 weights
        z3 = np.dot(self.hidden, self.W2) + self.HO_bias
        # final activation function
        self.outputs = self.softmax(z3)
        self.prediction = [0] * 26
        self.prediction[np.argmax(self.outputs[0])] = 1
        self.prediction = [self.prediction]
        return self.prediction

    # backward propagate through the network
    def backward(self, y):
        self.y = np.array(y).reshape(1, self.output_size)
        # error in output
        # Cross-entropy Error
        outputs_error = self.loss(y)
        # applying derivative of sigmoid to error
        outputs_delta = self.crossEntropyPrime(self.y,self.outputs) * self.softmaxPrime(self.outputs) 
    
        # adjusting first set (input --> hidden) weights
    
        # z2 error: how much our hidden layer weights contributed to output error
        hidden_error = outputs_delta.dot(self.W2.T) 
        # applying derivative of sigmoid to z2 error
        hidden_delta = hidden_error * self.sigmoidPrime(self.hidden)
        

        
        # adjusting first set (input --> hidden) weights
        # W1 += X.T.dot(forw1_delta) 
        self.W1_grads += self.inputs.T.dot(hidden_delta)
        # adjusting second set (hidden --> output) weights
        # W2 += forw1.T.dot(forw2_delta)
        self.W2_grads += self.hidden.T.dot(outputs_delta)
        
        self.IH_b_gradients += hidden_delta
        self.HO_b_gradients += outputs_delta
        
        return outputs_error

    def loss(self, y):
        return - np.sum([q * np.log(p + 1e-10) for p, q in zip(self.outputs, self.y)])
    

    def update_weights(self, learning_rate):
        
        self.W1 += self.W1_grads * learning_rate
        self.W2 += self.W2_grads * learning_rate
    
        self.W1_grads = np.zeros(shape=(self.input_size, self.hidden_size))
        self.W2_grads = np.zeros(shape=(self.hidden_size, self.output_size))
        
        self.IH_bias += learning_rate * self.IH_b_gradients
        self.HO_bias += learning_rate * self.HO_b_gradients
        
        self.IH_b_gradients = np.zeros_like(self.IH_bias)
        self.HO_b_gradients = np.zeros_like(self.HO_bias)
        
    def accuracy(self):
        return int(np.argmax(self.prediction[0]) == np.argmax(self.y[0]))
    
    
#if __name__ == "__main__":

    
# STEP 2 - PREPARING THE DATA
    
# Examine the dataset

data = loadData(file,columns)
    
train, test = train_test_split(data, test_size=0.2)
# Both ys_train and ys_test are arrays of letters as strings
ys_train = np.array(train.iloc[:,0].values)
    
ys_test = np.array(test.iloc[:,0].values)

train = train.iloc[:,1:17]
    
test = test.iloc[:,1:17]
    
labels = 'abcdefghijklmnopqrstuvwxyz'
    
train_losses, test_losses = [], []

# Using 50 Hidden Nodes
model = NNSoftClassif(16,50,26)
 
for e in range(EPOCH):
    
    epoch_train_losses, epoch_accs =  [], []
    epoch_acc = 0
    for i in range(train.shape[0]):
            
        x = np.array(train.iloc[i].values)
        x = x/15
        s = ys_train[i]
        y = charToOneHot(s)
    
        model.forward(x)
        
        loss = model.backward(y)
        epoch_train_losses.append(loss)
    
        accuracy = model.accuracy()
        epoch_acc += accuracy                

        if i % update_count == 0:       
            model.update_weights(learning_rate)
                        
    epoch_loss = np.mean(epoch_train_losses)
    train_losses.append(epoch_loss)
    epoch_acc /= (i + 1)
    epoch_accs.append(epoch_acc * 100)   

    print(e, loss, epoch_acc * 100)        

test_acc = 0     
for i in range(test.shape[0]):
            
    x = np.array(test.iloc[i].values)
    x = x/15
    s = ys_test[i]
    y = charToOneHot(s)
    
    model.forward(x)
    loss = model.loss(y)
    acc = model.accuracy()
        
    test_losses.append(loss)
    test_acc += acc
    
test_loss = np.mean(test_losses)
accuracy = test_acc/test.shape[0]

print(test_loss, accuracy)
            
    
plt.plot(train_losses, label='Train Loss')
plt.plot(epoch_accs, label='Train Accuracy')
#plt.plot(test_losses, label='test')
plt.legend()        
    
  
    
    
            
       
    
    
    
    
    
    
        
        
        
        
    
        
        
        
       
        
        
    
    
    
    
    
    
    
        
        
    
        

        
        
    
        
        
    
    
   
        
    
    
    