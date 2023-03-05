import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
from sklearn.neural_network import MLPClassifier
#-------------------------------------讀取檔案
df_X = pd.read_csv('train.csv',index_col='Attribute17')
df_Y = pd.read_csv('train.csv',usecols=['Attribute17'])
df_X['Attribute1'] = pd.to_datetime(df_X['Attribute1'])
df_X = df_X.drop('Attribute1',axis=1) #將日期去除
#-------------------------------------SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
#處理空值 Strategy 使用最常出現的值(因為有文字)
X = imputer.fit_transform(df_X.values)
Y = imputer.fit_transform(df_Y.values)
#-------------------------------------進行文字處理
#因為有幾條項目中存在中文，用labelEncoder轉為數字
lable_1 = LabelEncoder()
X[:,6] = lable_1.fit_transform(X[:,6]) 
X[:,8] = lable_1.fit_transform(X[:,8])
X[:,-1] = lable_1.fit_transform(X[:,-1])
Y = lable_1.fit_transform(Y)
#-------------------------------------資料測試分類
[X_train, X_test, Y_train, Y_test] = train_test_split(X,Y,test_size=0.2,random_state=60,shuffle=True)
#-------------------------------------Model
# MLPClassifier是一個監督學習演算法，MLP又名多層感知機，也叫人工神經網路
classifier = MLPClassifier(activation='relu',solver='adam',alpha=0.0001)
classifier.fit(X_train,Y_train)
# MLP參數說明
# 1. hidden_layer_sizes ：例如hidden_layer_sizes=（50， 50），表示有兩層隱藏層，第一層隱藏層有50個神經元，第二層也有50個神經元。 
# 2. activation ：啟動函數，{'identity'， 'logistic'， 'tanh'， 'relu'}， 預設relu 
# 3. solver： {'lbfgs'， 'sgd'， 'adam'}， 預設adam，用來優化權重 
# 4. alpha ：float，可選的，預設0.0001，正則化項參數 
# 5. batch_size ： int ， 可選的，預設『auto』，隨機優化的minibatches的大小batch_size=min（200，n_samples），如果solver是『lbfgs』，分類器將不使用minibatch 
# 6. learning_rate ：學習率，用於權重更新，只有當solver為'sgd'時使用，{'constant'，'invscaling'， 'adaptive'}，預設constant 
# 7. power_t： double， 可選， default 0.5，只有solver='sgd'時使用，是逆擴展學習率的指數. 當learning_rate='invscaling'，用來更新有效學習率。 
# 8. max_iter： int，可選，預設200，最大反覆運算次數。 
# 9. random_state：int 或RandomState，可選，預設None，隨機數生成器的狀態或種子。 
# 10. shuffle： bool，可選，預設True，只有當solver='sgd'或者'adam'時使用，判斷是否在每次迭代時對樣本進行清洗。 
#..........
#-------------------------------------分數
print("訓練分數")
print(classifier.score(X_train,Y_train)) #訓練分數
Y_pred = classifier.predict(X_test)
joblib.dump(classifier, 'model.pkl') # 將model存入檔案內
print("測試分數")
print(accuracy_score(Y_test,Y_pred)) #測試分數
#-------------------------------------用test.csv 來進行測試
df_test = pd.read_csv('test.csv') #
df_test = df_test.drop('Attribute1',axis=1)
df_answer = pd.read_csv('ex_submit.csv',usecols=['ans'])
#-------------------------------------將空值補滿
Z = imputer.fit_transform(df_test.values)
#-------------------------------------處理文字
lable_2 = LabelEncoder()
Z[:,6] = lable_2.fit_transform(Z[:,6]) 
Z[:,8] = lable_2.fit_transform(Z[:,8])
Z[:,-1] = lable_2.fit_transform(Z[:,-1])
Z_model = joblib.load('model.pkl') #將前面的model用來預測
Z_pred = Z_model.predict(Z)
#-------------------------------------把預測csv檔案匯出
submission = pd.read_csv('ex_submit.csv')
submission["ans"] = Z_pred
submission.to_csv('ex_submit_result.csv',index=False)
#-------------------------------------測試csv檔案分數
print("用test.csv的測試分數")
print(accuracy_score(df_answer.values,Z_pred))