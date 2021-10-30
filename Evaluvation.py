#%%
import pandas as pd
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import GaussianNB #Naive Bayes
from sklearn.ensemble import RandomForestClassifier, VotingClassifier #VOTE
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,precision_score,f1_score,recall_score
from sklearn.model_selection import train_test_split

data = pd.read_csv("heart.csv")

#%%
SVM = svm.SVC(kernel='linear', C=1, random_state=1)
gnb = GaussianNB()
logreg = LogisticRegression(solver='liblinear',random_state=1)
clf1 = LogisticRegression(solver='liblinear',random_state=1)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GaussianNB()
#vote = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
vote = VotingClassifier(estimators=[('lr', clf1),  ('gnb', clf3)], voting='hard')
models={'Naive Bayes':gnb,
        'Logistic Regression':logreg,'Vote':vote,
        'SVM':SVM,}
#%%
results_df = pd.DataFrame(columns=['Model', 'Accuracy mean %', 
                                   'Precision mean %','Recall mean %','F1 Score mean %','Features'])
#%%  
def model_build(m,X_train, X_test, y_train, y_test):
    models[m].fit(X_train, y_train)
    acc = accuracy_score(y_test, models[m].predict(X_test)) * 100
    prec = precision_score(y_test, models[m].predict(X_test)) * 100
    rec = recall_score(y_test, models[m].predict(X_test)) * 100
    f1 = f1_score(y_test, models[m].predict(X_test)) * 100
    return acc,prec,rec,f1
#%%
y=data['target']
X=data.drop('target',1)
features=list(X.columns)
#print("Features ARE : ",features)

significant_features=['sex', 'cp', 
                       'fbs', 'restecg', 
                      'exang', 'oldpeak', 
                      'slope', 'ca', 'thal']
found_significant_features=['sex', 'cp', 
                        'restecg', 
                      'oldpeak', 
                      'slope', 'ca', 'thal']
significant_features_list=[significant_features,found_significant_features,features]
for j in list(models.keys()):
    for i in significant_features_list:
        X=data[i]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        acc,prec,rec,f1=model_build(j,X_train, X_test, y_train, y_test)
        results_df_2 = pd.DataFrame(data=[[j, acc, prec,rec,f1,i]], 
                          columns=['Model', 'Accuracy mean %', 
                                   'Precision mean %','Recall mean %','F1 Score mean %','Features'])    
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