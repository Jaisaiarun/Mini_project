# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 15:22:08 2021

@author: jaisa
"""

#%%
import pandas as pd
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.naive_bayes import GaussianNB #Naive Bayes
from sklearn.ensemble import RandomForestClassifier, VotingClassifier #VOTE
from sklearn import svm
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score),  
           'recall' : make_scorer(recall_score), 
           'f1_score' : make_scorer(f1_score)}
data = pd.read_csv("processedTestset.csv")

#%%
SVM = svm.SVC(kernel='linear', C=1)
gnb = GaussianNB()
logreg = LogisticRegression(solver='liblinear')
clf1 = LogisticRegression(solver='liblinear')
clf2 = GaussianNB()
vote = VotingClassifier(estimators=[('lr', clf1),  ('gnb', clf2)], voting='hard')

models={'Naive Bayes':gnb,
        'Logistic Regression':logreg,'Vote':vote,
        'SVM':SVM}
#%%
results_df = pd.DataFrame(columns=['Model', 'Accuracy mean %', 
                                   'Precision mean %','Recall mean %','F1 Score mean %','Features'])
#%%  
def kfold_scores(clf,X,y):
    cv = StratifiedKFold(n_splits=10, shuffle=True)
    scores = cross_validate(clf, X, y, cv=cv,scoring=scoring)
    return scores
#%%
y=data['num']
X=data.drop('num',1)
features=list(X.columns)
#print("Features ARE : ",features)
#%%
significant_features=['sex', 'cp', 
                       'fbs', 'restecg', 
                      'exang', 'oldpeak', 
                      'slope', 'ca', 'thal']
found_significant_features=['sex', 'cp', 
                       'fbs', 'restecg', 
                      'exang', 'oldpeak', 
                      'thalach', 'ca', 'thal']
significant_features_list=[significant_features,found_significant_features,features]
for j in list(models.keys()):
    for i in significant_features_list:
         print(str(i)+'BEGIN:')
         X=data[i]
         scores=kfold_scores(models[j],X,y)
         keys=list(scores.keys())
         results_df_2 = pd.DataFrame(data=[[j, scores[keys[2]].mean()*100, scores[keys[3]].mean()*100,scores[keys[4]].mean()*100,scores[keys[5]].mean()*100,i]], 
                          columns=['Model', 'Accuracy mean %', 
                                   'Precision mean %','Recall mean %','F1 Score mean %','Features'])
         results_df = results_df.append(results_df_2, ignore_index=True)    
#%%
#print(list(models.keys()))
results_df.to_csv('Testset_result.csv')            
#%%
#acc = accuracy_score(y_test, lr_clf.predict(X_test)) * 100
#prec = precision_score(y_test, lr_clf.predict(X_test)) * 100
#rec = recall_score(y_test, lr_clf.predict(X_test)) * 100
#f1 = f1_score(y_test, lr_clf.predict(X_test)) * 100
#
#print(acc,prec,rec,f1)