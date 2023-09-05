# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array. 
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown values

## Program:


Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

Developed by: S.VINOD KUMAR

RegisterNumber:  212222240116
```python 
import pandas as pd
data=pd.read_csv('/content/Placement_Data (1).csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size =0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver ="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy


from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:

## 1.Placement Data
![the Logistic Regression Model to Predict the Placement Status of Student](/s1.png)

## 2.Salary Data

![the Logistic Regression Model to Predict the Placement Status of Student](/s2.png)

## 3. Checking the null function()
![the Logistic Regression Model to Predict the Placement Status of Student](/s3.png)

## 4.Data Duplicate
![the Logistic Regression Model to Predict the Placement Status of Student](/s4.png)

## 5.Print Data
![the Logistic Regression Model to Predict the Placement Status of Student](/s5.png)

## 6.Data Status
![the Logistic Regression Model to Predict the Placement Status of Student](/s7.png)

## 7.y_prediction array
![the Logistic Regression Model to Predict the Placement Status of Student](/s12.png)

## 8.Accuracy value

![the Logistic Regression Model to Predict the Placement Status of Student](/s8.png)

## 9.Confusion matrix
![the Logistic Regression Model to Predict the Placement Status of Student](/s9.png)

## 10.Classification Report
![the Logistic Regression Model to Predict the Placement Status of Student](/s10.png)

## 11.Prediction of LR
![the Logistic Regression Model to Predict the Placement Status of Student](/s11.png)




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
