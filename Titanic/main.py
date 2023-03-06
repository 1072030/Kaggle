#Type of machine learning system to build:
    #1. Supervised Learning.
    #2. Batch Learning (also called "offline learning").
    #3. Model-based learning.

#Python Libraries

import numpy as np # linear algebra
import pandas as pd # data processing
import os
import seaborn as sns
import matplotlib.pyplot as plt # data visualization
from pandas.plotting import scatter_matrix # data visualization

from sklearn.model_selection import train_test_split # Machine Learning - split dataset (train/test)
from sklearn.model_selection import StratifiedShuffleSplit # Machine Learning - stratified shuffle split (train/test)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
#讀取data
training_data = pd.read_csv(os.getcwd()+"/train.csv")
#,index_col='Survived'
df_train = pd.read_csv(os.getcwd()+"/train.csv")
df_test = pd.read_csv(os.getcwd()+"/test.csv")
total_data = pd.DataFrame()
total_data = total_data.append(df_train)
total_data = total_data.append(df_test)
print(total_data.head())
# drop(inplace=True) 對原始值進行修改

# df_y = pd.read_csv(os.getcwd()+"/train.csv",usecols=["Survived"])
print(df_train.head())
print(df_train.isnull().sum()) # ckeck null values


# 填補空值 
# df_train["Age"].fillna(df_train["Age"].median(),inplace=True)
# 一開始使用中位數來填補空值 但結果發現準確率太低了，可能是中間沒有分析完全
total_data.loc[df_train["Age"]<=16,"Age"]=1
total_data.loc[df_train["Age"]!=1,"Age"]=2
print("Age value",total_data["Age"].unique()) #確認值
sns.barplot(data=df_train,x="Embarked",y="Survived")
# plt.show()
total_data["Embarked"].fillna("S",inplace=True)

# clear data
# Cabin 缺失值太多 PassengerId,Ticket,Fare 跟存活率較無關
total_data.drop(["Cabin","PassengerId","Ticket","Fare","Name"],axis=1,inplace=True)
lable_1 = LabelEncoder()
total_data["Embarked"] = lable_1.fit_transform(total_data["Embarked"])

# 處理 SibSp(兄弟姊妹)
sns.countplot(df_train,x=df_train["SibSp"]) #show data
#　0占大多數 代表大多數沒有兄弟姊妹
# plt.show()
#比對 SibSp 特徵存活率 1和2生存率較高分為一組
sns.barplot(data=df_train,x="SibSp",y="Survived") #show data
# plt.show()

total_data.loc[(total_data["SibSp"]==1) | (total_data["SibSp"]==2),"SibSp"]=1
total_data.loc[total_data["SibSp"]>2,"SibSp"] = 2
total_data.loc[total_data["SibSp"]==0,"SibSp"] = 0
print(total_data["SibSp"].value_counts())

# 處理 Parch(父母或孩子的數量) 
sns.countplot(df_train,x=df_train["Parch"]) #show data
# plt.show() # 0 占多數 1和2少量

sns.barplot(data=df_train,x="Parch",y="Survived") #show data
plt.show()
# 分類 0 -> 0  1.2 -> 1 超過2 -> 3
total_data.loc[(total_data["Parch"]==1) | (total_data["Parch"]==2),"Parch"]=1
total_data.loc[total_data["Parch"]>2,"Parch"] = 2
total_data.loc[total_data["Parch"]==0,"Parch"] = 0
print(total_data["Parch"].value_counts())

# test_data有418 所以Survived is null 有418 由此可知空值都處理完畢
print(total_data.isnull().sum())



