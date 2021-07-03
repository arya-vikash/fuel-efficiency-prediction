import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

def process_origin_col(df):
    df['Origin']=df['Origin'].map({1: "India", 2: "USA", 3: "Germany"})
    return df

cyl,hp,acc=0,2,4

class CustomAttrAdder(BaseEstimator, TransformerMixin):
    def __init__(self, acc_on_power=True): 
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
                       ('attr_adder',CustomAttrAdder())])
    return num_attrs,num_pipeline

def full_pipeline_transformer(data):
    cat_attrs=['Origin']
    # access num_pipeline by calling the respective function
    num_attrs,num_pipeline=num_pipeline_transformer(data)
    full_pipeline=ColumnTransformer([('num',num_pipeline,list(num_attrs))])
    full_pipeline.fit_transform(data)
    return full_pipeline



#Standard scaler
def standard_scaler(data):
    df=pd.read_csv('pre_scaler_num_data.csv')
    scaler=StandardScaler()
    scaler.fit(df)
    scaled_data=scaler.transform(data)
    return scaled_data

  
## One hot encoder
def onehot_encoder(df):
    data=df['Origin'][0]
    if data=='Germany':
        return np.array([1., 0., 0.])
    elif data=='India':
        return np.array([[0., 1., 0.]])
    else:
        return np.array([[0., 0., 1.]])

    
def predict_mpg(config,model):
    if type(config)==dict:
        df=pd.DataFrame(config)
    else:
        df=config
    
    preproc_df=process_origin_col(df)
    
    pipeline=full_pipeline_transformer(df)
    #returns numerical data after imputation and add_attr
    pre_scaled_num_data=pipeline.transform(preproc_df)
    
    # give this data to standard scaler ...returns scaled data
    scaled_data=standard_scaler(pre_scaled_num_data)
    
    ## call one hot encoder for categorical feature encoding
    cat_data=onehot_encoder(df)
    
    ## now combine the numercal scaled data and categorical encoded data
    combined_data=np.append(scaled_data,cat_data)
    
    ## reshape combine data from 1d array to ndarray
    prepared_data=combined_data.reshape(1,-1)
    
    #predict output
    y_pred=model.predict(prepared_data)

    return y_pred

