import matplotlib
matplotlib.use('Agg')
from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot
import pandas as pd

dataset = pd.read_csv('data/train.csv', index_col=None, na_values=['NA'])

X = dataset[['RESOURCE', 'MGR_ID', 'ROLE_ROLLUP_1', 'ROLE_ROLLUP_2', 'ROLE_DEPTNAME', 'ROLE_TITLE', 'ROLE_FAMILY_DESC', 'ROLE_FAMILY', 'ROLE_CODE']]
# X = dataset[['ACTION', 'RESOURCE', 'MGR_ID', 'ROLE_ROLLUP_1', 'ROLE_ROLLUP_2', 'ROLE_DEPTNAME', 'ROLE_TITLE', 'ROLE_FAMILY_DESC', 'ROLE_FAMILY', 'ROLE_CODE']]
y = dataset['ACTION']

model = XGBClassifier()
model.fit(X, y)

plot_importance(model)

pyplot.tight_layout()
pyplot.savefig('feature_importance.png')
