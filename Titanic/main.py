#Type of machine learning system to build:
    #1. Supervised Learning.
    #2. Batch Learning (also called "offline learning").
    #3. Model-based learning.

#Python Libraries

import pandas as pd # data processing
import os
import seaborn as sns
import matplotlib.pyplot as plt # data visualization

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import neighbors
#讀取data
training_data = pd.read_csv(os.getcwd()+"/train.csv")
df_train = pd.read_csv(os.getcwd()+"/train.csv")
df_test = pd.read_csv(os.getcwd()+"/test.csv")
total_data = pd.DataFrame()
total_data = total_data.append(df_train)
total_data = total_data.append(df_test)
print(total_data.head()) # check data

print(df_train.head()) # check data
print(df_train.isnull().sum()) # ckeck null values


# 填補空值 
# df_train["Age"].fillna(df_train["Age"].median(),inplace=True)
# 一開始使用中位數來填補空值 但結果發現準確率太低了，可能是中間沒有分析完全
total_data.loc[df_train["Age"]<=16,"Age"]=1
total_data.loc[df_train["Age"]!=1,"Age"]=2
print("Age value",total_data["Age"].unique()) #確認值
sns.barplot(data=df_train,x="Embarked",y="Survived")
plt.show()

# clear data
# Cabin 缺失值太多 PassengerId,Ticket,Fare 跟存活率較無關
total_data.drop(["Cabin","PassengerId","Ticket","Fare","Name"],axis=1,inplace=True)
total_data["Embarked"].fillna("S",inplace=True)
lable_1 = LabelEncoder()
total_data["Embarked"] = lable_1.fit_transform(total_data["Embarked"])
total_data.loc[total_data["Sex"]=="male","Sex"] = 1
total_data.loc[total_data["Sex"]!=1,"Sex"] = 2


# 處理 SibSp(兄弟姊妹)
sns.countplot(df_train,x=df_train["SibSp"]) #show data
#　0占大多數 代表大多數沒有兄弟姊妹
plt.show()
#比對 SibSp 特徵存活率 1和2生存率較高分為一組
sns.barplot(data=df_train,x="SibSp",y="Survived") #show data
plt.show()

total_data.loc[(total_data["SibSp"]==1) | (total_data["SibSp"]==2),"SibSp"]=1
total_data.loc[total_data["SibSp"]>2,"SibSp"] = 2
total_data.loc[total_data["SibSp"]==0,"SibSp"] = 0
print(f"""SibSp values:\n{total_data["SibSp"].value_counts()}""")

# 處理 Parch(父母或孩子的數量) 
sns.countplot(df_train,x=df_train["Parch"]) #show data
plt.show() # 0 占多數 1和2少量

sns.barplot(data=df_train,x="Parch",y="Survived") #show data
plt.show()
# 分類 0 -> 0  1.2 -> 1 超過2 -> 3
total_data.loc[(total_data["Parch"]==1) | (total_data["Parch"]==2),"Parch"]=1
total_data.loc[total_data["Parch"]>2,"Parch"] = 2
total_data.loc[total_data["Parch"]==0,"Parch"] = 0
print(f"""Parch values:\n{total_data["Parch"].value_counts()}""")

# test_data有418 所以Survived is null 有418 由此可知空值都處理完畢
print(total_data.head()) #check data
print(total_data.isnull().sum()) # check null values
# 製作train set and test set
x = []
y = "Survived"
for j in total_data.keys():
    if j != y:
        x.append(j)
df_x = total_data[total_data[y].notnull()][x].values
df_y = total_data[total_data[y].notnull()][y].values
test_x = total_data[total_data[y].isnull()][x].values
# 確認大小
print(f"shape:[{df_x.shape},{df_y.shape},{test_x.shape}]")

# 分類
X_train, X_test, Y_train, Y_test = train_test_split(df_x,df_y,test_size=0.2,random_state=60,shuffle=True)
print(f"測試大小:[{X_train.shape},{Y_train.shape},{X_test.shape},{Y_test.shape}]")
# classifier 
# classifier = MLPClassifier(activation='relu',solver='adam',alpha=0.0001)

# KNN classifier
# n_neighbors=3 0.737 n_neighbors=4 0.754 n_neighbors=5 0.782 n_neighbors=7 0.776
classifier = neighbors.KNeighborsClassifier(n_neighbors=5,algorithm="brute")

classifier.fit(X_train,Y_train)
print("KNeighbors Classifier:")
# print("MPL Classifier:")
print("訓練分數: ",classifier.score(X_train,Y_train))#訓練分數
Y_pred = classifier.predict(X_test)
print("測試分數: ",accuracy_score(Y_test,Y_pred))#測試分數

final_pred = classifier.predict(test_x)
# 轉成整數格式
final_pred = final_pred.astype(int)

final_set = pd.DataFrame({"PassengerId":df_test["PassengerId"],"Survived":final_pred})
final_set.to_csv("submission.csv",index=False)