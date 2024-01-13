# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 15:20:21 2022

@author: SATYA MAHA LAKSHMI
"""

import pandas as pd
#import numpy as np

data=pd.read_csv(r"C:\Users\SATYA MAHA LAKSHMI\OneDrive\Desktop\aiml\iris.csv")

data.shape #no.of rows and columns
data.size  #total no.of elements
data.head()
data.info()
data.describe()   #to find range of values in each column

##splitting data into dependent and independent variables

#iloc=integer location
## values------used to convert dataframe into matrix.  since algorithm only accepts matrix format

x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

##splitting data for training and testing purpose
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=9)  #test_size denotes 20% of data splitted for testing and remaining for training


##importing kNN algorithm named KNeighborsClassifier from neighbors module
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=1)
model.fit(xtrain,ytrain)

ypred=model.predict(xtest) #prediction values

#comparing prediction values given by algorithm and testing values and checking for accuracy

from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,ypred)*100) 

## giving input and checking output
print(model.predict([[7.3,5.5,4.3,1.9]]))