import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
import matplotlib.pyplot as plt
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

X = df[['sex', 'race_new', 'ethnicity_hispanic', 'prncptx', 'workrvu',
       'inout', 'transt', 'age', 'anesthes', 'surgspec', 'electsurg', 'height',
       'weight', 'diabetes', 'smoke', 'dyspnea', 'fnstatus2', 'ventilat',
       'hxcopd', 'ascites', 'hxchf', 'hypermed', 'renafail', 'dialysis',
       'discancr', 'wndinf', 'steroid', 'wtloss', 'bleeddis', 'transfus',
       'prsepis', 'emergncy', 'wndclas', 'optime', 'podiagtx10',
       'anesthes_other', 'sodium', 'creatinine', 'albumin', 'wbc', 'hct',
       'plt', 'inr', 'asa']].values
y = df['outcome']

from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X) # normalizes the dataset

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(C=0.01, penalty='l2', solver='liblinear').fit(X_train,y_train)

yhat = LR.predict(X_test) # predicted y values
yhat_prob = LR.predict_proba(X_test) # first col is prob of predicting 0, second col is prob of predicting 1
yhat_prob1 = yhat_prob[:,1]

#from sklearn.metrics import jaccard_score
#accuracy = jaccard_score(y_test, yhat,pos_label=0) # accuracy
#print('Accuracy: ', accuracy)

#from sklearn.metrics import f1_score
#f1score = f1_score(y_test, yhat, average='weighted') # f1 score
#print('F1 score: ', f1score)

#auc = metrics.roc_auc_score(y_test, yhat_prob1)
#print('auc: ',auc)

#from sklearn.metrics import confusion_matrix
#cm = metrics.confusion_matrix(y_test, yhat, labels=[0,1])

#sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
#print('Sensitivity: ', sensitivity)

#specificity = cm[1,1]/(cm[1,0]+cm[1,1])
#print('Specificity: ', specificity)

# SHAP plot

#import shap
#explainer = shap.Explainer(LR, X_train, feature_names=X.dtype.names)
#shap_values = explainer(X_test)
#shap.plots.beeswarm(shap_values)

# k-fold cross-validation

#from sklearn.model_selection import KFold
#kf = KFold(n_splits=5)
#kf.get_n_splits(X)

#for train_index, test_index in kf.split(X):
    #X_train, X_test = X[train_index], X[test_index]
    #y_train, y_test = y[train_index], y[test_index]

#from sklearn.model_selection import cross_val_score
#accuracies = cross_val_score(LR, X, y, cv=10)
#print(f'Scores for each fold: {accuracies}')
#print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
#print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

#from sklearn.model_selection import GridSearchCV
#grid_param = {"C":[0.001,0.01,0.1,1], "penalty":["l1","l2"], "solver":["liblinear"]}
#grid = GridSearchCV(estimator=LR, param_grid=grid_param, scoring='accuracy', cv=5, refit=True, n_jobs=1)
#grid.fit(X_train, y_train)
#print(grid.best_params_)
#print(grid.best_score_)

#clfLR = grid.best_estimator_
#print('Test accuracy: %.3f' % clfLR.score(X_test, y_test))

# AUC plot

#fpr, tpr, threshold = metrics.roc_curve(y_test, yhat_prob1)
#roc_auc = metrics.auc(fpr, tpr)
#plt.title('AUC plot')
#plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
#plt.legend(loc = 'lower right')
#plt.plot([0, 1], [0, 1],'r--')
#plt.xlim([0, 1])
#plt.ylim([0, 1])
#plt.ylabel('True Positive Rate')
#plt.xlabel('False Positive Rate')
#plt.show()
