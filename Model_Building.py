#%%
import pandas as pd
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
import itertools
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score),  
           'recall' : make_scorer(recall_score), 
           'f1_score' : make_scorer(f1_score)}

#%% FUNCTIONS
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

def kfold_scores(clf,X,y):
    cv = StratifiedKFold(n_splits=10, shuffle=True)
    scores = cross_validate(clf, X, y, cv=cv,scoring=scoring)
    return scores
#%% GETTING DATASET AND EXTRACTING FEATURES AND TARGET VALUES

data = pd.read_csv("processedDataset.csv")
y=data['num']
X=data.drop('num',1)

features=list(X.columns)

print("Features ARE : ",features)
#%%FINDING ALL THE COMBINATIONS
features_combination=column_combination(features)
print("Combinations of features :\n",features_combination)
print("Length Of combinations : ",len(features_combination))        
#%% Building Models
SVM = svm.SVC(kernel='linear', C=1)
knn = KNeighborsClassifier(n_neighbors=21)
tree = DecisionTreeClassifier(max_depth=3)
gnb = GaussianNB()
logreg = LogisticRegression(solver='liblinear')
neural_net = MLPClassifier()
clf1 = LogisticRegression(solver='liblinear')
clf2 = GaussianNB()
vote = VotingClassifier(estimators=[('lr', clf1),  ('gnb', clf2)], voting='hard')

models={'KNN':knn,'Naive Bayes':gnb,'Decision Tree':tree,
        'Logistic Regression':logreg,'Vote':vote,
        'SVM':SVM,'Neural Network':neural_net}
#%% MAIN CODE

results_df = pd.DataFrame(columns=['Model', 'Accuracy mean %', 
                                   'Precision mean %','Recall mean %','F1 Score mean %','Features'])

model_name='Neural Network' #Change Model name to implement other models
for i in features_combination:
    print(str(i)+'BEGIN:')
    X=data[i]
    scores=kfold_scores(models[model_name],X,y)
    keys=list(scores.keys())
    results_df_2 = pd.DataFrame(data=[[model_name, scores[keys[2]].mean()*100, scores[keys[3]].mean()*100,scores[keys[4]].mean()*100,scores[keys[5]].mean()*100,i]], 
                          columns=['Model', 'Accuracy mean %', 
                                   'Precision mean %','Recall mean %','F1 Score mean %','Features'])
    results_df = results_df.append(results_df_2, ignore_index=True)    
#%%    
results_df.to_csv(str(model_name)+'.csv')
#%% TRAIL
#%%TESTING EACH MODEL
#X=data.drop('num',1)
#['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
f=['sex', 'restecg', 'exang']
for i in range(1,40):
    tree = DecisionTreeClassifier(max_depth=i)
    X=data[f]
    scores=kfold_scores(tree,X,y)
    keys=list(scores.keys())
    if scores[keys[3]].mean()*100>80:    
        for i in range(len(scores)):
            print(keys[i]+" mean : "+str(scores[keys[i]].mean()*100))