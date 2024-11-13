# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages
2. Import the dataset to operate on
3. Split the dataset.
4. Predict the required output

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Thirumalai V
RegisterNumber: 212223040229
*/
```
```
import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extractiaon.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:
## Data Head:
![image](https://github.com/user-attachments/assets/e385e182-1339-4348-9516-e72a5117a281)

## Data Info:
![image](https://github.com/user-attachments/assets/d3a3242c-bfea-44b1-8aea-fdc1bd3fc4cf)

## Data isnull():
![image](https://github.com/user-attachments/assets/f5bffc81-7e81-4f4c-b10c-9a40d3673f31)

## y_pred:
![image](https://github.com/user-attachments/assets/48865336-1543-4504-be1b-eee3ea09daaf)

## Accuracy:
![image](https://github.com/user-attachments/assets/66ba834e-48ef-467e-98fd-4ff948f84be0)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
