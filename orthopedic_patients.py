

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

#%%

df = pd.read_csv("column_3C_weka.csv")

print(df.isna().sum())
print("Total null values: ",df.isnull().sum().sum())
df.info()

df.nunique()


#%%
df["class"] = [0 if each == "Hernia" else 1 if each == "Spondylolisthesis" else 2 for each in df["class"]] 

#%%

x_data = df.drop(["class"],axis = 1)
y = df["class"].values
#%%

x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

#%%
from sklearn.model_selection import train_test_split

x_train,x_test, y_train,y_test = train_test_split(x, y,test_size=0.2,random_state=42)

#%%
# Logistic regression 
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(x_train.T,y_train.T)

print("Score: ",lr.score(x_test.T,y_test.T))

#%%

# Confusion Matrix For Logistic Regression
y_pred = lr.predict(x_test.T)
cm = confusion_matrix(y_test.T, y_pred)

f,ax = plt.subplots(figsize = (5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Y Predict")
plt.ylabel("Y True")
plt.show()


#%%
#Finding score range(1,15) neighbors

from sklearn.neighbors import KNeighborsClassifier



score_list = []

for i in range(1,15):
    knn2 = KNeighborsClassifier(n_neighbors=i)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))


#%%
# Score visualization
plt.plot(range(1,15),score_list)
plt.xlabel("n_neighbors")
plt.ylabel("Accuracy")
plt.show()



#%%

best_k = 0

for i in range(len(score_list)):
    
    if score_list[i] > score_list[best_k]:
        best_k = i

best_k = best_k + 1 # Because score_list[0] k=1 ,score_list[1] k = 2,..., score_list[n] k = n+1
print(score_list)
print(best_k)

#%%

# KNN Model
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(x_train,y_train)
print("{} nn Score: {} ".format(best_k,knn.score(x_test,y_test)))



#%%

# Confusion Matrix For Knn Model
y_pred = knn.predict(x_test)
cm = confusion_matrix(y_test,y_pred)

f, ax = plt.subplots(figsize=(5,5))

sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Y Predict")
plt.ylabel("Y True")
plt.show()


#%%

# Support Vector Machine
from sklearn.svm import SVC

svm = SVC()

svm.fit(x_train,y_train)

print("Support Vector Machine Accuracy: ",svm.score(x_test,y_test))

#%%

# Confusion Matrix For Svm Model
y_pred = svm.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Y Predict")
plt.ylabel("Y True")



#%%
# Navie Bayes
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)

print("Naive Bayes Accuracy",nb.score(x_test,y_test))


#%%
# Confusion Matrix For Naive Bayes Model

y_pred = nb.predict(x_test)
cm = confusion_matrix(y_test, y_pred)

f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Y Predict")
plt.ylabel("Y True")
plt.show()


#%%

# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(x_train,y_train)

print("Decision Tree Accuracy: ",dt.score(x_test, y_test))

#%%

# Confusion Matrix For Decision Tree

y_pred = dt.predict(x_test)
cm = confusion_matrix(y_test, y_pred)


f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Y Predict")
plt.ylabel("Y True")
plt.show()




#%%
# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=50)
rf.fit(x_train,y_train)

print("Random Forest Classifier Accuracy: ",rf.score(x_test,y_test))

#%%

#Confusion Matrix For Random Forest Classifier

y_pred = rf.predict(x_test)
cm = confusion_matrix(y_test, y_pred)

f, ax = plt.subplots(figsize = (5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Y Predict")
plt.ylabel("Y True")
plt.show()



    












