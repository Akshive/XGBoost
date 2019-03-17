import pandas as pd
import numpy as np
import xgboost as xgb

'''
XGBoost Only works with numerical values, so in your dataset change categorical values to numerica using pandas get_dummies fn.
'''


df = pd.read_csv('../input/mushrooms.csv')                 #data set path(change to your own)
df.head()

