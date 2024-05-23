import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, FunctionTransformer
import warnings
warnings.filterwarnings(action='ignore')
class rescale():
    """
    Rescale data to normal or range distribution

    Inputs:
    -------
    data_train: pandas.DataFrame
        Data for training model after cleaning
    data_test: pandas.DataFrame
        Data for external validation after cleaning
    acitivity_col: str
        Name of acitivity columns (pIC50, pChEMBL Value)
    scaler_method: string ('MinMaxScaler', 'StandardScaler', 'RobustScaler')
        Scaling method

    Returns: Rescaled Data
    -------
    data_train: pandas.DataFrame
        Data train afer rescaling
    data_test: pandas.DataFrame
        Data test afer rescaling
    """
    def __init__(self, data_train, data_test, activity_col, scaler_method ='MinMaxScaler'):
        self.activity_col    = activity_col
        self.scaler_method   = scaler_method
        
        self.data_train_0 = data_train
        self.data_test_0 = data_test
        
        self.Data_train = data_train.copy()
        self.Data_test = data_test.copy()
        
        self.scl1 = MinMaxScaler() 
        self.scl2 = StandardScaler()  
        self.scl3 = RobustScaler()   
        self.scl4 = FunctionTransformer(lambda x: x)
    
    def fit(self):
        self.data_train = self.data_train_0.copy()
        self.data_test = self.data_test_0.copy()
        self.activity_col = self.activity_col
        df_train_int = self.data_train.drop([self.activity_col], axis = 1).select_dtypes("int64")
        df_train_int = df_train_int.reset_index(drop = True)
        print("*"*75)
        print("Scaling method:", self.scaler_method)
        if df_train_int.shape[1] == (self.data_train.shape[1]-1):
            return None
        else:
            if self.scaler_method == 'MinMaxScaler':
                self.scl =self.scl1
            elif self.scaler_method == 'StandardScaler':
                self.scl =self.scl2
            elif self.scaler_method == 'RobustScaler':
                self.scl =self.scl3
            else:
                self.scl =self.scl4

            #Train
            
            y_train = self.data_train[self.activity_col].values
            X_train = self.data_train.drop([self.activity_col], axis = 1).select_dtypes("float64").values

            self.scl.fit(X_train)
            X_train_trans = self.scl.transform(X_train)
            idx = self.data_train.drop([self.activity_col], axis = 1).select_dtypes("float64").T.index

            df_X_train = pd.DataFrame(X_train_trans, columns = idx)
            df_y_train = pd.DataFrame(y_train, columns = [self.activity_col])
            Data_train_float = pd.concat([df_y_train, df_X_train], axis = 1)

            self.data_train = pd.concat([Data_train_float , df_train_int], axis = 1)

                #test
            df_test_int = self.data_test.drop([self.activity_col], axis = 1).select_dtypes("int64")
            df_test_int = df_test_int.reset_index(drop = True)

            y_test = self.data_test[self.activity_col].values
            X_test = self.data_test.drop([self.activity_col], axis = 1).select_dtypes("float64").values


            X_test_trans = self.scl.transform(X_test)

            df_X_test = pd.DataFrame(X_test_trans, columns = idx)
            df_y_test = pd.DataFrame(y_test, columns = [self.activity_col])
            Data_test_float = pd.concat([df_y_test, df_X_test], axis = 1)

            self.data_test = pd.concat([Data_test_float , df_test_int], axis = 1)
            
           
class rescale_scaffold():
    """
    Rescale data to normal or range distribution

    Inputs:
    -------
    data_train: pandas.DataFrame
        Data for training model after cleaning
    data_test: pandas.DataFrame
        Data for external validation after cleaning
    acitivity_col: str
        Name of acitivity columns (pIC50, pChEMBL Value)
    scaler_method: string ('MinMaxScaler', 'StandardScaler', 'RobustScaler')
        Scaling method

    Returns: Rescaled Data
    -------
    data_train: pandas.DataFrame
        Data train afer rescaling
    data_test: pandas.DataFrame
        Data test afer rescaling
    """
    def __init__(self, data_train, data_test, activity_col,ID, smile_col, scaler_method ='MinMaxScaler'):
        self.activity_col    = activity_col
        self.scaler_method   = scaler_method
        self.ID = ID
        self.smile_col = smile_col
        self.data_train_0 = data_train
        self.data_test_0 = data_test
        self.data_train = data_train.drop([self.ID, self.smile_col],axis = 1)
        self.data_test = data_test.drop([self.ID, self.smile_col],axis = 1)
        self.id_smile_train = data_train[[self.ID, self.smile_col]].reset_index(drop=True)
        self.id_smile_test = data_test[[self.ID, self.smile_col]].reset_index(drop=True)
        self.Data_train = data_train.copy()
        self.Data_test = data_test.copy()
        self.scl1 = MinMaxScaler() 
        self.scl2 = StandardScaler()  
        self.scl3 = RobustScaler()   
        self.scl4 = FunctionTransformer(lambda x: x)
    
    def fit(self):
        #self.data_train = self.data_train_0.copy()
        #self.data_test = self.data_test_0.copy()
        self.activity_col = self.activity_col
        df_train_int = self.data_train.drop([self.activity_col], axis = 1).select_dtypes("int64")
        df_train_int = df_train_int.reset_index(drop = True)
        print("*"*75)
        print("Scaling method:", self.scaler_method)
        if df_train_int.shape[1] == (self.data_train.shape[1]-1):
            self.data_train_final = self.data_train_0
            self.data_test_final = self.data_test_0
        else:
            if self.scaler_method == 'MinMaxScaler':
                self.scl =self.scl1
            elif self.scaler_method == 'StandardScaler':
                self.scl =self.scl2
            elif self.scaler_method == 'RobustScaler':
                self.scl =self.scl3
            else:
                self.scl =self.scl4

            #Train
            
            y_train = self.data_train[self.activity_col].values
            X_train = self.data_train.drop([self.activity_col], axis = 1).select_dtypes("float64").values

            self.scl.fit(X_train)
            X_train_trans = self.scl.transform(X_train)
            idx = self.data_train.drop([self.activity_col], axis = 1).select_dtypes("float64").T.index

            df_X_train = pd.DataFrame(X_train_trans, columns = idx)
            df_y_train = pd.DataFrame(y_train, columns = [self.activity_col])
            Data_train_float = pd.concat([df_y_train, df_X_train], axis = 1)

            self.data_train_final = pd.concat([self.id_smile_train, Data_train_float , df_train_int], axis = 1)

                #test
            df_test_int = self.data_test.drop([self.activity_col], axis = 1).select_dtypes("int64")
            df_test_int = df_test_int.reset_index(drop = True)

            y_test = self.data_test[self.activity_col].values
            X_test = self.data_test.drop([self.activity_col], axis = 1).select_dtypes("float64").values


            X_test_trans = self.scl.transform(X_test)

            df_X_test = pd.DataFrame(X_test_trans, columns = idx)
            df_y_test = pd.DataFrame(y_test, columns = [self.activity_col])
            Data_test_float = pd.concat([df_y_test, df_X_test], axis = 1)

            self.data_test_final = pd.concat([self.id_smile_test, Data_test_float , df_test_int], axis = 1)