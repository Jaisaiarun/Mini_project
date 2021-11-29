import pandas as pd
#%%
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate #for setting multiple metrics
from sklearn.naive_bayes import GaussianNB #Naive Bayes
from sklearn.linear_model import LogisticRegression 
from sklearn.neural_network import MLPClassifier 
from sklearn.ensemble import VotingClassifier #VOTE
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

#%%
import itertools
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score),  
           'recall' : make_scorer(recall_score), 
           'f1_score' : make_scorer(f1_score)}

#%%
def column_combination(a_list):
    all_combinations=[]
    for r in range(len(a_list) + 1):
        combinations_object = itertools.combinations(a_list, r)
        combinations_list = list(combinations_object)
        for i in combinations_list:
            #        print(list(i))
            i=list(i)
            if len(i)>=3:
                all_combinations += [i]
#    print(all_combinations)
    return all_combinations

#%%
data = pd.read_csv("heart.csv")
#%%
y=data['target']
X=data.drop('target',1)
features=list(X.columns)
print("Features ARE : ",features)
#%%
features_combination=column_combination(features)
#for i in features_combination:
#    print(i)

print(features_combination)
print("Length Of combinations : ",len(features_combination))    
#%%
#X_features=data[features_combination[3]]
#print(features_combination[3])
#print(X_features.head())
#%%
def kfold_scores(clf,X,y):
    cv = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    scores = cross_validate(clf, X, y, cv=cv,scoring=scoring)
    return scores

#%%
#scores=kfold_scores(gnb,X,y)
#keys=list(scores.keys())
#print(scores[keys[2]].mean()*100)
#%%
#MODELS
SVM = svm.SVC(kernel='linear', C=1, random_state=1)
knn = KNeighborsClassifier(n_neighbors=21)
tree = DecisionTreeClassifier(max_depth=3)
gnb = GaussianNB()
logreg = LogisticRegression(solver='liblinear',random_state=1)
neural_net = MLPClassifier()
clf1 = LogisticRegression(solver='liblinear',random_state=1)
clf2 = GaussianNB()
#vote = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
vote = VotingClassifier(estimators=[('lr', clf1),  ('gnb', clf2)], voting='hard')

models={'KNN':knn,'Naive Bayes':gnb,'Decision Tree':tree,
        'Logistic Regression':logreg,'Vote':vote,
        'Support Vector Machine (SVM)':SVM,'Neural Network':neural_net}

results_df = pd.DataFrame(columns=['Model', 'Accuracy mean %', 
                                   'Precision mean %','Recall mean %','F1 Score mean %','Features'])
#%%
##TESTING CODE
scores=kfold_scores(logreg,X,y)

keys=list(scores.keys())
#for i in range(len(scores)):
for i in range(len(scores)):
    print(keys[i]+" mean : "+str(scores[keys[i]].mean()))
#%%MAIN CODE
for i in features_combination:
    print(str(i)+'BEGIN:')
    X=data[i]
    scores=kfold_scores(logreg,X,y)
    keys=list(scores.keys())
    results_df_2 = pd.DataFrame(data=[["Logestic Regression", scores[keys[2]].mean()*100, scores[keys[3]].mean()*100,scores[keys[4]].mean()*100,scores[keys[5]].mean()*100,i]], 
                          columns=['Model', 'Accuracy mean %', 
                                   'Precision mean %','Recall mean %','F1 Score mean %','Features'])
    results_df = results_df.append(results_df_2, ignore_index=True)    
#%%
results_df.to_csv('Logestic Regression.csv')
#%%
#%%

results_df = pd.DataFrame(columns=['Model', 'Accuracy mean %', 
                                   'Precision mean %','Recall mean %','F1 Score mean %','Features'])
#%%
significant_features=['sex', 'cp', 
                       'fbs', 'restecg', 
                      'exang', 'oldpeak', 
                      'slope', 'ca', 'thal']
found_significant_features=['sex', 'cp', 
                        'restecg', 
                      'oldpeak', 
                      'slope', 'ca', 'thal']
#print(features)
y=data['target']
X=data[features]

significant_features_list=[significant_features,found_significant_features,features]
#%%

for i in significant_features_list:
    print(str(i)+'BEGIN:')
    X=data[i]
    scores=kfold_scores(vote,X,y)
    keys=list(scores.keys())
    results_df_2 = pd.DataFrame(data=[["Vote", scores[keys[2]].mean()*100, scores[keys[3]].mean()*100,scores[keys[4]].mean()*100,scores[keys[5]].mean()*100,i]], 
                          columns=['Model', 'Accuracy mean %', 
                                   'Precision mean %','Recall mean %','F1 Score mean %','Features'])
    results_df = results_df.append(results_df_2, ignore_index=True)    
#%%
results_df = pd.DataFrame(columns=['Model', 'Accuracy mean %', 
                                   'Precision mean %','Recall mean %','F1 Score mean %','Features'])

dt = pd.read_csv("Decision Tree.csv",index_col=0)
dt=dt.dropna()
results_df = results_df.append(dt, ignore_index=True)  

nb = pd.read_csv("Naive Bayes.csv",index_col=0)
nb=nb.dropna()
results_df = results_df.append(nb, ignore_index=True)    

sv = pd.read_csv("SVM.csv",index_col=0)
sv=sv.dropna()
results_df = results_df.append(sv, ignore_index=True)    

KN = pd.read_csv("KNN.csv",index_col=0)
KK=KN.dropna()
results_df = results_df.append(KN, ignore_index=True)    

lgr = pd.read_csv("Logestic Regression.csv",index_col=0)
lgr=lgr.dropna()
results_df = results_df.append(lgr, ignore_index=True)    

nn = pd.read_csv("Neural Network.csv",index_col=0)
nn=nn.dropna()
results_df = results_df.append(nn, ignore_index=True)    

v = pd.read_csv("Vote.csv",index_col=0)
v=v.dropna()
results_df = results_df.append(v, ignore_index=True)    

results_df.to_csv('Final_result.csv')
#%%------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#%%
for i in models.keys():
    print(i+'BEGIN:')
    scores=kfold_scores(models[i],X,y)
    keys=list(scores.keys())
    results_df_2 = pd.DataFrame(data=[["Logistic Regression", scores[keys[2]].mean()*100, scores[keys[3]].mean()*100,scores[keys[4]].mean()*100,scores[keys[5]].mean()*100]], 
                          columns=['Model', 'Accuracy mean %', 
                                   'Precision mean %','Recall mean %','F1 Score mean %'])
    results_df = results_df.append(results_df_2, ignore_index=True)
#%%
results_df.to_csv('heart_result.csv')
#%%
results_df = pd.DataFrame(columns=['Model', 'Accuracy mean %', 
                                   'Precision mean %','Recall mean %','F1 Score mean %'])
results_df.head()
#%%
#CHOOSING K VALUE
#accuracy_rate = []
#for i in range(1,40):
#    
#    knn = KNeighborsClassifier(n_neighbors=i)
#    cv = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
#    score=cross_val_score(knn,X,y,cv=cv)
#    accuracy_rate.append(score.mean())
#
#plt.figure(figsize=(10,6))
#plt.plot(range(1,40),accuracy_rate,color='green', linestyle='dashed', marker='o',
#         markerfacecolor='blue', markersize=10)
#plt.title('Accuracy Rate vs. K Value')
#plt.xticks(np.arange(1,40,1))
#plt.xlabel('K')
#plt.ylabel('Accuracy Rate')
##%%
##CHOOSING depth VALUE for Random tree
#accuracy_rate = []
#for i in range(1,40):
#    tree=DecisionTreeClassifier(max_depth=i)
#    cv = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
#    score=cross_val_score(tree,X,y,cv=cv)
#    accuracy_rate.append(score.mean())
#
#plt.figure(figsize=(10,6))
#plt.plot(range(1,40),accuracy_rate,color='green', linestyle='dashed', marker='o',
#         markerfacecolor='blue', markersize=10)
#plt.title('Accuracy Rate vs. Depth')
#plt.xticks(np.arange(1,40,1))
#plt.xlabel('Depth')
#plt.ylabel('Accuracy Rate')
#%%
data = pd.read_csv("processedDataset.csv")
y=data['num']
X=data.drop('num',1)

features=list(X.columns)

print("Features ARE : ",features)
#%%
from sklearn.model_selection import GridSearchCV
model = svm.SVC()
model.fit(X,y)
model.score(X,y)
#%%
params = {'C' : [0.01,0.1,0.25,0.5,0.75,1,10,100],
         'gamma' : ['auto','scale'],
         'kernel': ['rbf','poly','linear']}
gridsearch = GridSearchCV(model,params,refit=True)
gridsearch.fit(X,y)
print(gridsearch.best_params_)