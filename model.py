import pandas as pd
import numpy as np
from sklearn import metrics, cross_validation, preprocessing

SEED = 42

def cross_validation_function(X, y, model, N):
    total_auc = 0.0
    for i in range(N):
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=i*SEED)
        model.fit(X_train, y_train)
        predict = model.predict_proba(X_test)[:,1]
        auc = metrics.roc_auc_score(y_test, predict)
        print ("AUC (fold %d/%d): %f" % (i + 1, N, auc))
        total_auc += auc
    return total_auc / N

train = pd.read_csv('data/train.csv', index_col=None, na_values=['NA'])
test = pd.read_csv('data/test.csv', index_col=None, na_values=['NA'])

# X = train[['RESOURCE', 'MGR_ID', 'ROLE_ROLLUP_1', 'ROLE_ROLLUP_2', 'ROLE_DEPTNAME', 'ROLE_TITLE', 'ROLE_FAMILY_DESC', 'ROLE_FAMILY', 'ROLE_CODE']]
# y = train['ACTION']
# data_test = test[['RESOURCE', 'MGR_ID', 'ROLE_ROLLUP_1', 'ROLE_ROLLUP_2', 'ROLE_DEPTNAME', 'ROLE_TITLE', 'ROLE_FAMILY_DESC', 'ROLE_FAMILY', 'ROLE_CODE']]

X = train[['RESOURCE', 'MGR_ID', 'ROLE_ROLLUP_1', 'ROLE_ROLLUP_2', 'ROLE_DEPTNAME', 'ROLE_TITLE', 'ROLE_FAMILY_DESC', 'ROLE_FAMILY']]
y = train['ACTION']
data_test = test[['RESOURCE', 'MGR_ID', 'ROLE_ROLLUP_1', 'ROLE_ROLLUP_2', 'ROLE_DEPTNAME', 'ROLE_TITLE', 'ROLE_FAMILY_DESC', 'ROLE_FAMILY']]

encoder = preprocessing.OneHotEncoder()
encoder.fit(np.vstack((X, data_test)))
X = encoder.transform(X)
data_test = encoder.transform(data_test)

#################### Model ####################
print ('--------------------')

print ('LogisticRegression')
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1.5)
print (cross_validation_function(X, y, lr, 10))
print ('--------------------')

# print ('SVC')
# from sklearn.svm import SVC
# svc = SVC()
# print (cross_validation_function(X, y, svc, 10))
# print ('--------------------')

print ('RandomForestClassifier')
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
print (cross_validation_function(X, y, rf, 10))
# for name, importance in zip(X.columns, rf.feature_importances_):
#     print (name, importance)
print ('--------------------')

print ('GradientBoostingClassifier')
from sklearn.ensemble import GradientBoostingClassifier
gdbt = GradientBoostingClassifier()
print (cross_validation_function(X, y, gdbt, 10))
print ('--------------------')

print ('DecisionTreeClassifier')
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
print (cross_validation_function(X, y, decision_tree, 10))
# for name, importance in zip(X.columns, decision_tree.feature_importances_):
#     print (name, importance)
print ('--------------------')

# print ('neighbors')
# from sklearn import neighbors
# knn = neighbors.KNeighborsClassifier()
# print (cross_validation_function(X, y, knn, 10))
# print ('--------------------')

print ('xgb')
import xgboost as xgb
xgbo = xgb.XGBClassifier()
print (cross_validation_function(X, y, xgbo, 10))
print ('--------------------')

# print ('VotingClassifier')
# from sklearn.ensemble import VotingClassifier  
# voting = VotingClassifier(estimators=[('lr', lr), ('svc', svc), ('rf', rf), ('gdbt', gdbt), ('decision_tree', decision_tree), ('knn', knn), ('xgbo', xgbo)])
# print (cross_validation_function(X, y, voting, 10))
# print ('--------------------')
#################### Model ####################

index = []
for i in range(1, 58922):
    index.append(i)

model = lr
model.fit(X, y)
predictions = model.predict_proba(data_test)[:,1]
result = pd.DataFrame({'Id': index, 'Action': predictions})
result.to_csv('result.csv', index=False)
