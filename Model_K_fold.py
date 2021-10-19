import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate #for setting multiple metrics
from sklearn.naive_bayes import GaussianNB #Naive Bayes
from sklearn.linear_model import LogisticRegression 
from sklearn.neural_network import MLPClassifier 
from sklearn.ensemble import RandomForestClassifier, VotingClassifier #VOTE

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score), 
           'f1_score' : make_scorer(f1_score)}
#%%
data = pd.read_csv("heart.csv")
#%%
col=list(data.columns)
print(col)
#%%
y=data['target']
X=data.drop('target',1)
#%%
def kfold_scores(clf,X,y):
    cv = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    scores = cross_validate(clf, X, y, cv=cv,scoring=scoring)
    return scores

#%%
#MODELS
SVM = svm.SVC(kernel='linear', C=1, random_state=1)
knn = KNeighborsClassifier(n_neighbors=21)
tree=DecisionTreeClassifier(max_depth=3)
gnb = GaussianNB()
logreg = LogisticRegression(solver='liblinear',random_state=1)
neural_net = MLPClassifier()
clf1 = LogisticRegression(solver='liblinear',random_state=1)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GaussianNB()
#vote = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
vote = VotingClassifier(estimators=[('lr', clf1),  ('gnb', clf3)], voting='hard')

models={'KNN':knn,'Naive Bayes':gnb,'Decision Tree':tree,
        'Logistic Regression':logreg,'Vote':vote,
        'Support Vector Machine (SVM)':SVM,'Neural Network':neural_net}
#%%
scores=kfold_scores(vote,X,y)

keys=list(scores.keys())
#for i in range(len(scores)):
for i in range(len(scores)):
    print(keys[i]+" mean : "+str(scores[keys[i]].mean()))
#%%
results_df = pd.DataFrame(columns=['Model', 'Accuracy mean %', 
                                   'Precision mean %','Recall mean %','F1 Score mean %'])
for i in models.keys():
    print(i+'BEGIN:')
    scores=kfold_scores(models[i],X,y)
    keys=list(scores.keys())
    results_df_2 = pd.DataFrame(data=[[i, scores[keys[2]].mean()*100, scores[keys[3]].mean()*100,scores[keys[4]].mean()*100,scores[keys[5]].mean()*100]], 
                          columns=['Model', 'Accuracy mean %', 
                                   'Precision mean %','Recall mean %','F1 Score mean %'])
    results_df = results_df.append(results_df_2, ignore_index=True)
#%%
results_df.to_csv('heart_result.csv')
#%%
results_df = pd.DataFrame(columns=['Model', 'Accuracy mean %', 
                                   'Precision mean %','Recall mean %','F1 Score mean %'])
#results_df.head()
#%%
#CHOOSING K VALUE
accuracy_rate = []
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    cv = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    score=cross_val_score(knn,X,y,cv=cv)
    accuracy_rate.append(score.mean())

plt.figure(figsize=(10,6))
plt.plot(range(1,40),accuracy_rate,color='green', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Accuracy Rate vs. K Value')
plt.xticks(np.arange(1,40,1))
plt.xlabel('K')
plt.ylabel('Accuracy Rate')
#%%
#CHOOSING depth VALUE for Random tree
accuracy_rate = []
for i in range(1,40):
    tree=DecisionTreeClassifier(max_depth=i)
    cv = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    score=cross_val_score(tree,X,y,cv=cv)
    accuracy_rate.append(score.mean())

plt.figure(figsize=(10,6))
plt.plot(range(1,40),accuracy_rate,color='green', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Accuracy Rate vs. Depth')
plt.xticks(np.arange(1,40,1))
plt.xlabel('Depth')
plt.ylabel('Accuracy Rate')
