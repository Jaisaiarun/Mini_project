#%%
import pandas as pd
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import GaussianNB #Naive Bayes
from sklearn.ensemble import VotingClassifier #VOTE
from sklearn import svm
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate #for setting multiple metrics
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score),  
           'recall' : make_scorer(recall_score), 
           'f1_score' : make_scorer(f1_score)}
#%%  
def kfold_scores(m,X,y):
    cv = StratifiedKFold(n_splits=10, shuffle=True)
    scores = cross_validate(m, X, y, cv=cv,scoring=scoring)
    return scores
#%%
data = pd.read_csv("processedTestset.csv")
y=data['num']
X=data.drop('num',1)
features=list(X.columns)
#print("Features ARE : ",features)
#%%
SVM = svm.SVC(kernel='linear', C=1)
gnb = GaussianNB()
logreg = LogisticRegression(solver='liblinear')
clf1 = LogisticRegression(solver='liblinear')
clf2 = GaussianNB()

vote = VotingClassifier(estimators=[('lr', clf1),  ('gnb', clf2)], voting='hard')
models={'Naive Bayes':gnb,
        'Logistic Regression':logreg,'Vote':vote,
        'SVM':SVM,}

#%%
results_df = pd.DataFrame(columns=['Model', 'Accuracy mean %','Features'])
    
significant_features=['sex', 'cp', 
                       'fbs', 'restecg', 
                      'exang', 'oldpeak', 
                      'thalach', 'ca', 'thal']
significant_features_list=[significant_features,features]
for j in list(models.keys()):
    for i in significant_features_list:
        X=data[i]
        scores=kfold_scores(models[j],X,y)
        keys=list(scores.keys())
        results_df_2 = pd.DataFrame(data=[[j, scores[keys[2]].mean()*100,i]], 
                          columns=['Model', 'Accuracy mean %', 'Features'])

        results_df = results_df.append(results_df_2, ignore_index=True)    
        
#print(list(models.keys()))
results_df.to_csv('Validation_result.csv')            
