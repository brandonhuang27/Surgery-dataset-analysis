from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics

df = pd.read_csv("nsqip_outcomes_v2.csv")

df.loc[df["age"] == '90+', "age"] = '90'

# assign numbers to categorical variables
df["sex"] = df["sex"].astype('category')
df["sex"] = df["sex"].cat.codes

df["race_new"] = df["race_new"].astype('category')
df["race_new"] = df["race_new"].cat.codes

df["ethnicity_hispanic"] = df["ethnicity_hispanic"].astype('category')
df["ethnicity_hispanic"] = df["ethnicity_hispanic"].cat.codes

df["prncptx"] = df["prncptx"].astype('category')
df["prncptx"] = df["prncptx"].cat.codes

df["inout"] = df["inout"].astype('category')
df["inout"] = df["inout"].cat.codes

df["transt"] = df["transt"].astype('category')
df["transt"] = df["transt"].cat.codes

df["anesthes"] = df["anesthes"].astype('category')
df["anesthes"] = df["anesthes"].cat.codes

df["surgspec"] = df["surgspec"].astype('category')
df["surgspec"] = df["surgspec"].cat.codes

df["electsurg"] = df["electsurg"].astype('category')
df["electsurg"] = df["electsurg"].cat.codes

df["diabetes"] = df["diabetes"].astype('category')
df["diabetes"] = df["diabetes"].cat.codes

df["smoke"] = df["smoke"].astype('category')
df["smoke"] = df["smoke"].cat.codes

df["dyspnea"] = df["dyspnea"].astype('category')
df["dyspnea"] = df["dyspnea"].cat.codes

df["fnstatus2"] = df["fnstatus2"].astype('category')
df["fnstatus2"] = df["fnstatus2"].cat.codes

df["ventilat"] = df["ventilat"].astype('category')
df["ventilat"] = df["ventilat"].cat.codes

df["hxcopd"] = df["hxcopd"].astype('category')
df["hxcopd"] = df["hxcopd"].cat.codes

df["ascites"] = df["ascites"].astype('category')
df["ascites"] = df["ascites"].cat.codes

df["hxchf"] = df["hxchf"].astype('category')
df["hxchf"] = df["hxchf"].cat.codes

df["hypermed"] = df["hypermed"].astype('category')
df["hypermed"] = df["hypermed"].cat.codes

df["renafail"] = df["renafail"].astype('category')
df["renafail"] = df["renafail"].cat.codes

df["dialysis"] = df["dialysis"].astype('category')
df["dialysis"] = df["dialysis"].cat.codes

df["discancr"] = df["discancr"].astype('category')
df["discancr"] = df["discancr"].cat.codes

df["wndinf"] = df["wndinf"].astype('category')
df["wndinf"] = df["wndinf"].cat.codes

df["steroid"] = df["steroid"].astype('category')
df["steroid"] = df["steroid"].cat.codes

df["wtloss"] = df["wtloss"].astype('category')
df["wtloss"] = df["wtloss"].cat.codes

df["bleeddis"] = df["bleeddis"].astype('category')
df["bleeddis"] = df["bleeddis"].cat.codes

df["transfus"] = df["transfus"].astype('category')
df["transfus"] = df["transfus"].cat.codes

df["prsepis"] = df["prsepis"].astype('category')
df["prsepis"] = df["prsepis"].cat.codes

df["emergncy"] = df["emergncy"].astype('category')
df["emergncy"] = df["emergncy"].cat.codes

df["wndclas"] = df["wndclas"].astype('category')
df["wndclas"] = df["wndclas"].cat.codes

df["podiagtx10"] = df["podiagtx10"].astype('category')
df["podiagtx10"] = df["podiagtx10"].cat.codes

df["anesthes_other"] = df["anesthes_other"].astype('category')
df["anesthes_other"] = df["anesthes_other"].cat.codes

df["sodium"] = df["sodium"].astype('category')
df["sodium"] = df["sodium"].cat.codes

df["creatinine"] = df["creatinine"].astype('category')
df["creatinine"] = df["creatinine"].cat.codes

df["albumin"] = df["albumin"].astype('category')
df["albumin"] = df["albumin"].cat.codes

df["wbc"] = df["wbc"].astype('category')
df["wbc"] = df["wbc"].cat.codes

df["hct"] = df["hct"].astype('category')
df["hct"] = df["hct"].cat.codes

df["plt"] = df["plt"].astype('category')
df["plt"] = df["plt"].cat.codes

df["inr"] = df["inr"].astype('category')
df["inr"] = df["inr"].cat.codes

df["asa"] = df["asa"].astype('category')
df["asa"] = df["asa"].cat.codes

# decision trees

from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree

X = df[['sex', 'race_new', 'ethnicity_hispanic', 'prncptx', 'workrvu',
       'inout', 'transt', 'age', 'anesthes', 'surgspec', 'electsurg', 'height',
       'weight', 'diabetes', 'smoke', 'dyspnea', 'fnstatus2', 'ventilat',
       'hxcopd', 'ascites', 'hxchf', 'hypermed', 'renafail', 'dialysis',
       'discancr', 'wndinf', 'steroid', 'wtloss', 'bleeddis', 'transfus',
       'prsepis', 'emergncy', 'wndclas', 'optime', 'podiagtx10',
       'anesthes_other', 'sodium', 'creatinine', 'albumin', 'wbc', 'hct',
       'plt', 'inr', 'asa']].values
y = df["outcome"]

# Decision Tree

from sklearn.model_selection import train_test_split
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.2, random_state=3)

from sklearn.preprocessing import StandardScaler # for feature scaling
sc = StandardScaler()
X_trainset = sc.fit_transform(X_trainset)
X_testset = sc.transform(X_testset)

DT = DecisionTreeClassifier(criterion="entropy", max_depth=13)
DT.fit(X_trainset,y_trainset)

# Logistic Regression

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

from sklearn.preprocessing import StandardScaler # for feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(C=0.01, penalty='l2', solver='liblinear').fit(X_train,y_train)

# Random Forest

from sklearn.model_selection import train_test_split
X_training, X_testing, y_training, y_testing = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler # for feature scaling
sc = StandardScaler()
X_training = sc.fit_transform(X_training)
X_testing = sc.transform(X_testing)

from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators=20, criterion='entropy', bootstrap=False, random_state=0)
RF.fit(X_training, y_training)

# Neural Network

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

from sklearn.model_selection import train_test_split
X_trainingdata, X_testingdata, y_trainingdata, y_testingdata = train_test_split(X, y, test_size=0.2)

from sklearn.preprocessing import StandardScaler # for feature scaling
sc = StandardScaler()
X_trainingdata = sc.fit_transform(X_trainingdata)
X_testingdata = sc.transform(X_testingdata)

X_trainingdata = np.array(X_trainingdata)
y_trainingdata = np.array(y_trainingdata)
X_testingdata = np.array(X_testingdata)
y_testingdata = np.array(y_testingdata)

NN = Sequential()
NN.add(Dense(20, input_shape=(44,))) # 20 nodes, 44 input features
NN.add(Dense(20, activation='relu'))
NN.add(Dense(20, activation='relu'))
NN.add(Dense(20, activation='relu'))
NN.add(Dense(1, activation='sigmoid'))

NN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = NN.fit(X_train, y_train, epochs=5, validation_split=0.2)

# AUC plot

from sklearn.metrics import roc_curve
yhat_prob1 = DT.predict_proba(X_testset)[:,1]
fpr1 , tpr1, thresholds1 = roc_curve(y_testset, yhat_prob1)

yhat_prob2 = LR.predict_proba(X_test)[:,1]
fpr2 , tpr2, thresholds2 = roc_curve(y_test, yhat_prob2)

yhat_prob3 = RF.predict_proba(X_testing)[:,1]
fpr3 , tpr3, thresholds3 = roc_curve(y_testing, yhat_prob3)

yhat_prob4 = NN.predict(X_testingdata)
yhat_classes = (NN.predict(X_testingdata) > 0.5).astype("int32")
fpr4 , tpr4, thresholds4 = roc_curve(y_testingdata, yhat_prob4)

roc_auc1 = metrics.auc(fpr1, tpr1)
roc_auc2 = metrics.auc(fpr2, tpr2)
roc_auc3 = metrics.auc(fpr3, tpr3)
roc_auc4 = metrics.auc(fpr4, tpr4)
plt.plot([0,1],[0,1], 'k--')
plt.plot(fpr1, tpr1, 'b', label= "Decision Trees AUC = %0.2f" % roc_auc1)
plt.plot(fpr2, tpr2, 'g', label= "Logistic Regression AUC = %0.2f" % roc_auc2)
plt.plot(fpr3, tpr3, 'r', label= "Random Forest AUC = %0.2f" % roc_auc3)
plt.plot(fpr4, tpr4, 'y', label= "Neural Network AUC = %0.2f" % roc_auc4)
plt.legend(loc = 'lower right')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title('AUC plot')
plt.show()
