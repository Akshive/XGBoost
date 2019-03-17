import pandas as pd
import numpy as np
import xgboost as xgb

'''
XGBoost Only works with numerical values, so in your dataset change categorical values to numerica using pandas get_dummies fn.
'''


df = pd.read_csv('../input/mushrooms.csv')                 #data set path(change to your own)
df.head()

#creating train/val split.
train = np.array(df.loc[:, 'cap-shape_b':])
x_train = train[:int(train.shape[0]*0.8), :]
x_val = train[int(train.shape[0]*0.8):, :]
labels = np.array(df.loc[:, 'class_e':'class_p'])
y_train = labels[:int(train.shape[0]*0.8), :]
y_val = labels[int(train.shape[0]*0.8):, :]


dtrain = xgb.DMatrix(x_train, label = y_train)
dval = xgb.DMatrix(x_val, label = y_val)

params = {
    'objective':'binary:logistic',
    'max_depth':2,
    'silent':1,
    'eta':1
}

num_rounds = 5

watchlist  = [(dval,'val'), (dtrain,'train')]
bst = xgb.train(params, dtrain, num_rounds, watchlist)
