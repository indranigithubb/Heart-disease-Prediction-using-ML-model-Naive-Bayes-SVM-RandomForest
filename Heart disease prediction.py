#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing the dependencies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas


# In[3]:


# data collection and processing
# loading csv data to pandas dataframe

heart_data = pd.read_csv("heart.csv")


# In[4]:


# print first 5 rows of the dataset

heart_data.head()


# In[5]:


# print last 5 rows of the dataset

heart_data.tail()


# In[6]:


heart_data.columns


# In[7]:


# number of rows and columns in the dataset

heart_data.shape


# In[8]:


# getting information about the data

heart_data.info()


# In[9]:


# check age distribution of dataset

sns.distplot(heart_data['age'],bins=20)
plt.show()


# In[10]:


# check sex distribution of dataset

sns.distplot(heart_data['sex'],bins=20)
plt.show()


# In[11]:


# check cp distribution of dataset

sns.distplot(heart_data['cp'],bins=20)
plt.show()


# In[12]:


# check trestbps distribution of dataset

sns.distplot(heart_data['trestbps'],bins=20)
plt.show()


# In[13]:


# check chol distribution of dataset

sns.distplot(heart_data['chol'],bins=20)
plt.show()


# In[14]:


# check fbs distribution of dataset

sns.distplot(heart_data['fbs'],bins=20)
plt.show()


# In[15]:


# check restecg distribution of dataset

sns.distplot(heart_data['restecg'],bins=20)
plt.show()


# In[16]:


# check thalach distribution of dataset

sns.distplot(heart_data['thalach'],bins=20)
plt.show()


# In[17]:


# check exang distribution of dataset

sns.distplot(heart_data['exang'],bins=20)
plt.show()


# In[18]:


# check oldpeak distribution of dataset

sns.distplot(heart_data['oldpeak'],bins=20)
plt.show()


# In[19]:


# check slope distribution of dataset

sns.distplot(heart_data['slope'],bins=20)
plt.show()


# In[20]:


# check ca distribution of dataset

sns.distplot(heart_data['ca'],bins=20)
plt.show()


# In[21]:


# check thal distribution of dataset

sns.distplot(heart_data['thal'],bins=20)
plt.show()


# In[22]:


# check target distribution of dataset

sns.distplot(heart_data['target'],bins=20)
plt.show()


# In[23]:


# checking for missing values

heart_data.isnull().sum()


# In[25]:


# check for duplicate values (TRUE)

data_duplicate=heart_data.duplicated().any()
print(data_duplicate)


# In[26]:


# check for duplicate values (FALSE)

heart_data=heart_data.drop_duplicates()
data_duplicate=heart_data.duplicated().any()
print(data_duplicate)


# In[27]:


# Preprocessing

def onehot_encode(df, column_dict):
    df = df.copy()
    for column, prefix in column_dict.items():
        dummies = pd.get_dummies(df[column], prefix=prefix)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(column, axis=1)
    return df


# In[28]:


#get correlations of each features in dataset

corrmat = heart_data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))

#plot heat map
g=sns.heatmap(heart_data[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[29]:


# statistical measures of the data

heart_data.describe()


# In[30]:


# data without target column

p=heart_data.drop(["target"],axis=1)
q=heart_data["target"] # data of death event
p


# In[31]:


# checking distribution of the target variable

heart_data ['target'].value_counts()


# In[32]:


# evaluate target, finding whether data is imbalance or not
# 'cols' is list of colours for each column
# plotting of death and non death occured

cols = ["Orange","Blue"]
sns.countplot (x=heart_data["target"], palette=cols)
plt.show()


# In[33]:


# plot heatmap

sns.heatmap(heart_data.isnull(),yticklabels=False,cmap="viridis")
plt.show()


# In[34]:


# plotting histogram

heart_data.hist()

plt.show()


# In[35]:


plt.scatter(x=heart_data.age[heart_data.target==1], y=heart_data.thalach[(heart_data.target==1)], c="red")
plt.scatter(x=heart_data.age[heart_data.target==0], y=heart_data.thalach[(heart_data.target==0)])
plt.legend(["Disease", "Not Disease"])
plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate")
plt.show()


# In[36]:


# Boxplots for each numeric variable 

numeric_features = ['age', 'sex', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']

eda_df = heart_data.loc[:, numeric_features].copy()


# In[37]:


plt.figure(figsize=(16, 10))

for i in range(len(eda_df.columns)):
    plt.subplot(2, 4, i + 1)
    sns.boxplot(eda_df[eda_df.columns[i]])

plt.show()


# In[38]:


# Filtering data by POSITIVE Heart Disease patient
pos_data = heart_data[heart_data['target']==1]
pos_data.describe()


# In[39]:


# Filtering data by NEGATIVE Heart Disease patient
neg_data = heart_data[heart_data['target']==0]
neg_data.describe()


# In[40]:


print("(Positive Patients ST depression): " + str(pos_data['oldpeak'].mean()))
print("(Negative Patients ST depression): " + str(neg_data['oldpeak'].mean()))


# In[41]:


print("(Positive Patients thalach): " + str(pos_data['thalach'].mean()))
print("(Negative Patients thalach): " + str(neg_data['thalach'].mean()))


# In[42]:


# categorical and continuous attributes

categorical_value=[]
continuous_value=[]

for column in heart_data.columns:
    if heart_data[column].nunique()<=10:
        categorical_value.append(column)
    else:
        continuous_value.append(column)


# In[43]:


# Encoding categorical data

categorical_value


# In[44]:


heart_data['cp'].unique()


# In[45]:


# caterogical attributes all values plotting

heart_data.hist(categorical_value,figsize=(15,6))
plt.tight_layout()
plt.show()


# In[46]:


continuous_value


# In[47]:


# continuous attributes all values plotting

heart_data.hist(continuous_value,figsize=(15,6))
plt.tight_layout()
plt.show()


# In[48]:


# extracting the x and y from the dataset
x = heart_data.iloc[:,:-1].values
y = heart_data.iloc[:,-1].values
print(x)
print(y)


# In[49]:


# feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)
print(x)


# In[50]:


# Feature Scaling

st=StandardScaler()
heart_data[continuous_value]=st.fit_transform(heart_data[continuous_value])
heart_data.head()


# In[51]:


# splitting data into training and test data

x=heart_data.drop('target',axis=1)
y=heart_data['target']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print(x.shape,x_train.shape,x_test.shape)


# In[52]:


x_train


# In[53]:


x_test


# In[54]:


y_train


# In[55]:


y_test


# In[56]:


countNoDisease = len(heart_data[heart_data.target == 0])
countHaveDisease = len(heart_data[heart_data.target == 1])
print("Percentage of patients dont have heart disease: {:.2f}%".format((countNoDisease/(len(heart_data.target)))*100))
print("Percentage of patients have heart disease: {:.2f}%".format((countHaveDisease/(len(heart_data.target)))*100))


# In[57]:


countFemale= len(heart_data[heart_data.sex == 0])
countMale = len(heart_data[heart_data.sex == 1])
print("% of Female Patients: {:.2f}%".format((countFemale/(len(heart_data.sex))*100)))
print("% of male Patients: {:.2f}%".format((countMale/(len(heart_data.sex))*100)))


# In[58]:


# Logistic Regression

from sklearn.linear_model import LogisticRegression

log=LogisticRegression()
log.fit(x_train,y_train)

y_prediction1=log.predict(x_test)
accuracy_score(y_test,y_prediction1)


# In[59]:


# Classification

from sklearn.metrics import classification_report
def modeldevelopment(algorithm,x_train,y_train,x_test,y_test):
    model=algorithm()
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    print(classification_report(y_test,y_pred))
    


# In[60]:


modeldevelopment(LogisticRegression,x_train,y_train,x_test,y_test)


# In[61]:


# SVM

from sklearn import svm
svm=svm.SVC()
svm.fit(x_train,y_train)

y_prediction2=svm.predict(x_test)
accuracy_score(y_test,y_prediction2)


# In[62]:


# KNN

knn=KNeighborsClassifier()
knn.fit(x_train,y_train)
y_prediction3=knn.predict(x_test)
accuracy_score(y_test,y_prediction3)


# In[63]:


#from sklearn.model_selection import cross_val_score
knn_scores = []

for k in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    y_prediction=knn.predict(x_test)
    knn_scores.append(accuracy_score(y_test,y_prediction))
    


# In[64]:


knn_scores


# In[65]:


# Non-Linear ML Algorithms

heart_data=heart_data.drop_duplicates()
heart_data.shape


# In[66]:


x=heart_data.drop('target',axis=1)
y=heart_data['target']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[67]:


# Decision Tree Classifier

dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
y_prediction4=dt.predict(x_test)
accuracy_score(y_test,y_prediction4)


# In[68]:


# Random Forest Classifier

rf=RandomForestClassifier()
rf.fit(x_train,y_train)
y_prediction5=rf.predict(x_test)
accuracy_score(y_test,y_prediction5)


# In[69]:


# Gradient Boosting Classifier

gbc=GradientBoostingClassifier()
gbc.fit(x_train,y_train)
y_prediction6=gbc.predict(x_test)
accuracy_score(y_test,y_prediction6)


# In[70]:


# Naive bayes

nb = GaussianNB()
nb.fit(x_train, y_train)
y_prediction7=nb.predict(x_test)
accuracy_score(y_test,y_prediction7)


# In[71]:


# Neural network

from sklearn.neural_network import MLPClassifier
nn_model = MLPClassifier()
nn_model.fit(x_train, y_train)

y_prediction8=nn_model.predict(x_test)
accuracy_score(y_test,y_prediction8)


# In[72]:


import pandas as pd
final_data=({'Models':['LR','SVM','KNN','DT','RF','GB','NB','NN'],
                         'ACC':[accuracy_score(y_test,y_prediction1),
                                accuracy_score(y_test,y_prediction2),
                                accuracy_score(y_test,y_prediction3),
                                accuracy_score(y_test,y_prediction4),
                                accuracy_score(y_test,y_prediction5),
                                accuracy_score(y_test,y_prediction6),
                                accuracy_score(y_test,y_prediction7),
                                accuracy_score(y_test,y_prediction8)]})

df = pd.DataFrame(final_data)
print(df)


# In[73]:


# Plotting bar graph 

sns.barplot(final_data['Models'],final_data['ACC'])
plt.show()


# In[74]:


# Create confusion matrix and creating heatmap and getting accuracy score

cm=confusion_matrix(y_test,y_prediction)
ac=accuracy_score(y_test,y_prediction)
print('Accuracy is: ',ac*100)


# In[75]:


# Plot the graph

sns.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('True Value')
plt.show()


# In[76]:


# Building a predictive system to predict the target '0' or '1' i.e whether the heart is defective or not based on the feature values like "age" etc

input_data=(61,1,0,148,203,0,1,161,0,0,2,1,3)

# change input data to a numpy array

input_data_as_numpy_array=np.asarray(input_data)

# reshape numpy array as we are predicting for only one instance

input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

prediction=log.predict(input_data_reshaped)
print(prediction)

if(prediction[0]==0):
    print("The person does not have heart disease")
else:
    print("The person have heart disease")

