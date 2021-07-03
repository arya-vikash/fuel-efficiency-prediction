import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder



cols = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
data=pd.read_csv(r'C:\Users\vikash arya\datascience\datascience_krishnaik\data_sets\auto-mpg.data', names=cols,na_values = "?",comment = '\t',sep= " ",skipinitialspace=True)


split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(data,data['Cylinders']):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]

df_feature=strat_train_set.drop('MPG',axis=1)
df_target=strat_train_set['MPG']


###fully automated process

def process_origin_col(df):
    df['Origin']=df['Origin'].map({1: "India", 2: "USA", 3: "Germany"})
    return df

cyl,hp,acc=0,2,4

class CustomAttrAdder(BaseEstimator, TransformerMixin):
    def __init__(self, acc_on_power=True): # no *args or **kargs
        self.acc_on_power = acc_on_power
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        acc_on_cyl = X[:, acc] / X[:, cyl]
        if self.acc_on_power:
            acc_on_power = X[:, acc] / X[:, hp]
            return np.c_[X, acc_on_power, acc_on_cyl]
        
        return np.c_[X, acc_on_cyl]

def num_pipeline_transformer(data):
    numeric=['int64','float64']
    num_attrs=data.select_dtypes(include=numeric)
    num_pipeline=Pipeline([('imputer',SimpleImputer(strategy='median')),
                       ('attr_adder',CustomAttrAdder()),
                           ('scaler',StandardScaler())])
    return num_attrs,num_pipeline

def full_pipeline_transformer(data):
    cat_attrs=['Origin']
    # access num_pipeline by calling the respective function
    num_attrs,num_pipeline=num_pipeline_transformer(data)
    full_pipeline=ColumnTransformer([('num',num_pipeline,list(num_attrs)),('cat',OneHotEncoder(),cat_attrs)])
    prepared_data=full_pipeline.fit_transform(data)
    return prepared_data


# -----preprocessing-------
preprocessed_data=process_origin_col(df_feature)
prepared_data=full_pipeline_transformer(preprocessed_data)


#grid search cv for random forest


param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid,
                           scoring='neg_mean_squared_error',
                           return_train_score=True,
                           cv=10,
                          )

grid_search.fit(prepared_data, df_target)

#best params
print(grid_search.best_params_)

final_model=grid_search.best_estimator_
import pickle
pickle.dump(final_model,open('auto_mpg.pkl','wb'))