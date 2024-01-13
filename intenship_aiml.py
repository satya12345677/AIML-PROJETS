# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 14:41:18 2022

@author: SATYA MAHA LAKSHMI
"""
##--->KNN
import pandas as pd
a=pd.read_csv(r'C:\Users\SATYA MAHA LAKSHMI\OneDrive\Desktop\aiml\bank-additional-full - Copy.csv')
a.shape #shape of the data frame
a.columns##tells how many numbers of coloums,rows 
a.index
a.describe()
a.info() #tells is there any null coloums or rows

l=["job","marital","education","default","housing","loan","contact","month","day_of_week","poutcome"]
for i in l:
    print(a[i].value_counts())




from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in l:
    a[i]=le.fit_transform(a[i])##convert srings into intergers
#data['o']=le.fit_transform(data['o'])

x=a.iloc[:,:-1].values
y=a.iloc[:,-1].values

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

'-------------------------------------------------------------------------------------------------'

##--->DECISION_THREE
import pandas as pd
a=pd.read_csv(r'C:\Users\SATYA MAHA LAKSHMI\OneDrive\Desktop\aiml\bank-additional-full - Copy.csv')
a.shape#shape of the data frame
a.columns##tells how many numbers of coloums,rows 
a.index
a.describe()
a.info()

l=["job","marital","education","default","housing","loan","contact","month","day_of_week","poutcome"]
for i in l:
    print(a[i].value_counts())


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in l:
    a[i]=le.fit_transform(a[i])##convert srings into intergers
#data['o']=le.fit_transform(data['o'])

x=a.iloc[:,:-1].values
y=a.iloc[:,-1].values

##splitting data for training and testing purpose
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=9) 
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(criterion="gini")
model.fit(xtrain,ytrain)


ypred=model.predict(xtest)

from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,ypred)*100)

'-------------------------------------------------------------------------------------------------'

##--->RANDOMFORSEST
import pandas as pd
a=pd.read_csv(r'C:\Users\SATYA MAHA LAKSHMI\OneDrive\Desktop\aiml\bank-additional-full - Copy.csv')
a.shape#shape of the data frame
a.columns##tells how many numbers of coloums,rows 
a.index
a.describe()
a.info()

l=["job","marital","education","default","housing","loan","contact","month","day_of_week","poutcome"]
for i in l:
    print(a[i].value_counts())


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in l:
    a[i]=le.fit_transform(a[i])##convert srings into intergers
#data['o']=le.fit_transform(data['o'])

x=a.iloc[:,:-1].values
y=a.iloc[:,-1].values

##splitting data for training and testing purpose
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=9) 
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(xtrain,ytrain)


ypred=model.predict(xtest)

from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,ypred)*100)

'------------------------------------------------------------------------------------------------------------------'

##--->>NAIVEBASE
import pandas as pd
a=pd.read_csv(r'C:\Users\SATYA MAHA LAKSHMI\OneDrive\Desktop\aiml\bank-additional-full - Copy.csv')
a.shape#shape of the data frame
a.columns##tells how many numbers of coloums,rows 
a.index
a.describe()
a.info()

l=["job","marital","education","default","housing","loan","contact","month","day_of_week","poutcome"]
for i in l:
    print(a[i].value_counts())


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in l:
    a[i]=le.fit_transform(a[i])##convert srings into intergers
#data['o']=le.fit_transform(data['o'])

x=a.iloc[:,:-1].values
y=a.iloc[:,-1].values

##splitting data for training and testing purpose
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=9)
from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(xtrain,ytrain)


ypred=model.predict(xtest)

from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,ypred)*100)


'------------------------------------------------------------------------------------------------------'

##--->>LOGESTICREHGRESSION
import pandas as pd
a=pd.read_csv(r'C:\Users\SATYA MAHA LAKSHMI\OneDrive\Desktop\aiml\bank-additional-full - Copy.csv')
a.shape#shape of the data frame
a.columns##tells how many numbers of coloums,rows 
a.index
a.describe()
a.info()

l=["job","marital","education","default","housing","loan","contact","month","day_of_week","poutcome"]
for i in l:
    print(a[i].value_counts())


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in l:
    a[i]=le.fit_transform(a[i])##convert srings into intergers
#data['o']=le.fit_transform(data['o'])

x=a.iloc[:,:-1].values
y=a.iloc[:,-1].values

##splitting data for training and testing purpose
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=9)
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
xtrain = sc_x.fit_transform(xtrain) 
xtest = sc_x.transform(xtest)


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(xtrain, ytrain)
y_pred = classifier.predict(xtest)


from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,y_pred)*100)





