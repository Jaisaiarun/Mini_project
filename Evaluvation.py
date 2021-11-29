#%%
import pandas as pd
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import GaussianNB #Naive Bayes
from sklearn.ensemble import RandomForestClassifier, VotingClassifier #VOTE
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,precision_score,f1_score,recall_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate #for setting multiple metrics
#%%
data = pd.read_csv("processedTestset.csv")

#%%
print(data.columns)
#%%

SVM = svm.SVC(kernel='linear', C=1)
gnb = GaussianNB()
logreg = LogisticRegression(solver='liblinear')
clf1 = LogisticRegression(solver='liblinear')
clf2 = GaussianNB()
#vote = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
vote = VotingClassifier(estimators=[('lr', clf1),  ('gnb', clf2)], voting='hard')
models={'Naive Bayes':gnb,
        'Logistic Regression':logreg,'Vote':vote,
        'SVM':SVM,}
#%%
#results_df = pd.DataFrame(columns=['Model', 'Accuracy mean %', 
#                                   'Precision mean %','Recall mean %','F1 Score mean %','Features'])
results_df = pd.DataFrame(columns=['Model', 'Accuracy mean %','Features'])
#%%  
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score),  
           'recall' : make_scorer(recall_score), 
           'f1_score' : make_scorer(f1_score)}
def model_build(m,X,y):
#    m.fit()
#    y_pred = cross_val_predict(m, X, y, cv=10)
#    conf_mat = confusion_matrix(y, y_pred)
    cv = StratifiedKFold(n_splits=10, shuffle=True)
    scores = cross_validate(m, X, y, cv=cv,scoring=scoring)
#    return (((conf_mat[0][0]+conf_mat[1][1])/270)*100)
#    models[m].fit(X_train, y_train)
#    acc = accuracy_score(y_test, models[m].predict(X_test)) * 100
#    prec = precision_score(y_test, models[m].predict(X_test)) * 100
#    rec = recall_score(y_test, models[m].predict(X_test)) * 100
#    f1 = f1_score(y_test, models[m].predict(X_test)) * 100
    return scores
#%%
y=data['num']
X=data.drop('num',1)
features=list(X.columns)
#print("Features ARE : ",features)
    
#sex, cp, Fbs, Restecg, Exang, Oldpeak, Slope, ca and Thal 
significant_features=['sex', 'cp', 
                       'fbs', 'restecg', 
                      'exang', 'oldpeak', 
                      'slope', 'ca', 'thal']

#sex, cp, Fbs, Restecg, Thalach, Exang, Oldpeak, ca and Thal 
found_significant_features=['sex', 'cp', 
                       'fbs', 'restecg', 
                      'exang', 'oldpeak', 
                      'thalach', 'ca', 'thal']
significant_features_list=[significant_features,found_significant_features,features]
for j in list(models.keys()):
    for i in significant_features_list:
        X=data[i]
#        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#        acc,prec,rec,f1=model_build(j,X_train, X_test, y_train, y_test)
        scores=model_build(models[j],X,y)
        keys=list(scores.keys())
        results_df_2 = pd.DataFrame(data=[[j, scores[keys[2]].mean()*100,i]], 
                          columns=['Model', 'Accuracy mean %', 'Features'])
#        scores=model_build(models[j],X,y)
#        results_df_2 = pd.DataFrame(data=[[j,scores,i]], 
#                          columns=['Model', 'Accuracy mean %', 'Features'])
        results_df = results_df.append(results_df_2, ignore_index=True)    
#%%
#print(list(models.keys()))
results_df.to_csv('Validation_result.csv')            
#%%
#acc = accuracy_score(y_test, lr_clf.predict(X_test)) * 100
#prec = precision_score(y_test, lr_clf.predict(X_test)) * 100
#rec = recall_score(y_test, lr_clf.predict(X_test)) * 100
#f1 = f1_score(y_test, lr_clf.predict(X_test)) * 100
#
#print(acc,prec,rec,f1)
#%%
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

y=data['num']
X=data.drop('num',1)

y_pred = cross_val_predict(vote, X, y, cv=10)
conf_mat = confusion_matrix(y, y_pred)
#%%
#print((conf_mat[0][0]+conf_mat[1][1])/len(data)*100)
print(len(data))
print(conf_mat)