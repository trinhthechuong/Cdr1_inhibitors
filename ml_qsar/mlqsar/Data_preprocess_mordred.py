import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

# from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor


class Data_preprocess():
    """
    Preprocess data include:
     - Clean duplicated data (columns and rows)
     - Find percentage of missing value, choose suitable method to make imputation
     - Check variance threshold
     - Convert nomial to integer.
    
    Input:
    -----
    data_train: pandas.DataFrame
        Data for training model.
    data_test: pandas.DataFrame
        Data for external validation.
    acitivity_col: str
        Name of acitivity columns (pIC50, pChEMBL Value)
    var_thresh: float
        Variance threshold to remove features

    Returns:
    --------
    Data_train and Data_test: pandas.DataFrame
        Data after preprocessing
    """
    def __init__(self, data_train, data_test, activity_col, var_thresh=0.05):
        self.data_train= data_train.copy()
        self.data_test= data_test.copy()
        self.activity_col = activity_col
        self.var_thresh = var_thresh
    
    # 1. Remove duplicate data - rows and columns
    def Duplicate_data(self):
        
        # 1.1. Duplicate rows: xóa hàng bị trùng 
        # train
        dup_rows = self.data_train.duplicated()
        print(f"Total duplicated rows-train: {(dup_rows == True).sum()}")
        print("Data train before drop duplicates:", self.data_train.shape[0])
        self.data_train.drop_duplicates(inplace = True)
        self.data_train.reset_index(drop = True, inplace = True)
        print("Data train after drop duplicates:", self.data_train.shape[0])
        print(75*"*")
        self.train_dup = (dup_rows == True).sum()
        
        # test
        dup_rows = self.data_test.duplicated()
        print(f"Total duplicated rows-test: {(dup_rows == True).sum()}")
        print("Data test before drop duplicates:", self.data_test.shape[0])
        self.data_test.drop_duplicates(inplace = True)
        self.data_test.reset_index(drop = True, inplace = True)
        print("Data test after drop duplicates:", self.data_test.shape[0])
        print(75*"*")
        self.test_dup = (dup_rows == True).sum()
       
        
        # # 1.2. Duplicate columns: xóa cột bị trùng
          
        # #matrix transpose
        # dup = self.data_train.T[self.data_train.T.duplicated()]
        # idx = dup.index
        # print(idx)
        # print(f"Total similar columns: {dup.shape[0]}")
        # self.dup_col = dup.shape[0]
        # #train
        # print("Data train before drop duplicates:", self.data_train.shape)
        # self.data_train.drop(idx, axis = 1, inplace = True) 
        # print("Data after drop duplicates:", self.data_train.shape)
        # print(75*"*")
        
        # #test
        # print("Data test before drop duplicates:", self.data_test.shape)
        # self.data_test.drop(idx, axis = 1, inplace = True) 
        # print("Data test after drop duplicates:", self.data_test.shape)
        # print(75*"*")
        
    # 2. Check Variance Threshold: 
    def Variance_Threshold(self):
        y = self.data_train[self.activity_col]
        X = self.data_train.drop([self.activity_col], axis =1)
        print(X.shape, y.shape)
        while True:
            try:
               # Define thresholds to check
                thresholds = np.arange(0.0, 1, 0.05)
                # Apply transform with each threshold
                results = list()
                for t in thresholds:
                # define the transform
                    transform = VarianceThreshold(threshold=t)
                # transform the input data
                    X_sel = transform.fit_transform(X)
                # determine the number of input features
                    n_features = X_sel.shape[1]
                    print('>Threshold=%.2f, Features=%d' % (t, n_features)) 
                # store the result
                    results.append(n_features)
                break
            except:
                break
        # plot the threshold vs the number of selected features
        sns.set('notebook')
         # plot the threshold vs the number of selected features
        plt.figure(figsize=(14,8))
        plt.title("Variance analysis", fontsize = 24, weight = 'semibold')
        plt.xlabel("Variance threshold", fontsize = 16)
        plt.ylabel("Number of features", fontsize = 16)
       
        plt.plot(thresholds[:len(results)], results)
        #plt.savefig(self.SAVE_PREFIX +"var.png", dpi = 600)
        plt.show()

    # 3. Remove variance Threshold:  
    def remove_low_variance(self, thresh):
        selector = VarianceThreshold(thresh)
        selector.fit(self.data_train)
        features = selector.get_support(indices = False)
        features[0]=True # target can not be removed
        self.data_train = self.data_train.loc[:, features]
        #lst = self.data_train.columns
        self.data_test = self.data_test.loc[:, features]
        print(75*"*")
        self.var = features.sum()
    

###########################################################################################
# MISSING VALUES
    # 4. Find Missing Percentage: 
    
    # def find_missing_percent(self, data):
    #     """
    #     Returns dataframe containing the total missing values and percentage of total
    #     missing values of a column.
    #     """
    #     miss_df = pd.DataFrame({'ColumnName':[],'TotalMissingVals':[],'PercentMissing':[]})
    #     for col in data.columns:
    #         sum_miss_val = data[col].isnull().sum()
    #         percent_miss_val = round((sum_miss_val/data.shape[0])*100,2)
    #         missinginfo = {"ColumnName" : col, "TotalMissingVals" : sum_miss_val, "PercentMissing" : percent_miss_val}
    #         miss_df = miss_df.append(missinginfo, ignore_index = True)

    #     miss_df = miss_df[miss_df["PercentMissing"] > 0.0]
    #     miss_df = miss_df.reset_index(drop = True)
    #     return miss_df
    
    # 5. Handle missing values
    def Missing_value_cleaning(self):
        miss_df = self.find_missing_percent(self.data_train)
        print(miss_df)
        # Remove columns with high missing percentage
        miss_thresh= 50 # modify
        
        drop_cols = miss_df[miss_df['PercentMissing'] > miss_thresh].ColumnName.tolist()
        print("Drop_cols",  drop_cols)
        
        
            
        self.data_train.drop(drop_cols, axis =1, inplace = True)
        self.data_test.drop(drop_cols, axis =1, inplace = True)
         

        
        print("Total missing value-train", self.data_train.isnull().sum().sum())
        print("Total missing value-test", self.data_test.isnull().sum().sum())
        
        null_data_train = self.data_train[self.data_train.isnull().any(axis=1)]
        display(null_data_train)
        print("Total row-train with missing value", null_data_train.shape[0])
        
        null_data_test = self.data_test[self.data_test.isnull().any(axis=1)]
        display(null_data_test)
        print("Total row-test with missing value", null_data_test.shape[0])
        
        if null_data_train.shape[0] == 0:
            self.data_test = self.data_test.dropna(inplace = False)
            self.data_test.reset_index(drop = True, inplace = True)
        else:
            #S = (input("Do you want to remove rows have missing values:").title())
            S = 'N'
            if S == 'Y':
                Data_train = self.data_train.dropna(inplace = False)
                Data_train.reset_index(drop = True, inplace = True)

                Data_test = self.data_test.dropna(inplace = False)
                Data_test.reset_index(drop = True, inplace = True)
            else:
                while True:
                    try:
                        #M = int(input("Please select type of Imputation:\n\t1:Mean\n\t2:Median\n\t3:Mode\n\t4:KNNImputer\n\t5:MICE\n"))
                        M = 4
                        break
                    except:
                        print("Error value!")
                if M == 1:
                    imp=SimpleImputer(missing_values=np.NaN, strategy='mean')
                elif M == 2:
                    imp=SimpleImputer(missing_vlues=np.NaN, strategy='median')
                elif M == 3:
                    imp=SimpleImputer(missing_values=np.NaN, strategy='mode')
                elif M == 4:
                    while True:
                        try:
                            #n_neighbors = int(input("Please input n_neighbors:"))
                            n_neighbors = 5
                            break
                        except:
                            print("Error value!")
                    imp=KNNImputer(n_neighbors=n_neighbors) 
                else:
                    A = int(input("Please select Algorithm of Imputation:\n\t1:BayesianRidge\n\t2:RandomForest\n"))
                    if A == 1:
                        estimator = BayesianRidge()
                    else:
                        estimator = GradientBoostingRegressor(random_state = 42)
                    imp= IterativeImputer(random_state=42, missing_values=np.NaN, estimator= estimator)
                
                imp.fit(self.data_train)
                
               
                # train
                Data_train=pd.DataFrame(imp.transform(self.data_train))
                Data_train.columns=self.data_train.columns
                Data_train.index=self.data_train.index
                display(Data_train.shape)
                # test
                Data_test=pd.DataFrame(imp.transform(self.data_test))
                Data_test.columns=self.data_test.columns
                Data_test.index=self.data_test.index
                display(Data_test.shape)
                print(75*"*")
            self.data_train = Data_train
            self.data_test = Data_test
      
    
    # 6. Convert low unique columns to integer
 
    def Nomial(self):
        data = self.data_train.loc[:, (self.data_train.nunique() <10).values & (self.data_train.max() <10).values]  #feature with unique < 10 and max value <10 will set to be int64
        col = data.columns #select columns with int64
        #set all  col_olumn to int64
        #print("DINH TINH:", col)
        #savefile("Nomial_Col.txt", col)
        self.data_train[col]=self.data_train[col].astype('int64')  
        self.data_test[col]=self.data_test[col].astype('int64')    
        display(self.data_train.head(5))
    
    
        
    # 7. Activate 
    def fit(self):
        self.Duplicate_data()
        self.Missing_value_cleaning() # missing values must be handled before feature selection
        self.Variance_Threshold()
        self.remove_low_variance(thresh= self.var_thresh)
        self.Nomial()
       


class Data_preprocess_Scaffold():
    """
    Preprocess data include:
     - Clean duplicated data (columns and rows)
     - Find percentage of missing value, choose suitable method to make imputation
     - Check variance threshold
     - Convert nomial to integer.
    
    Input:
    -----
    data_train: pandas.DataFrame
        Data for training model.
    data_test: pandas.DataFrame
        Data for external validation.
    acitivity_col: str
        Name of acitivity columns (pIC50, pChEMBL Value)
    var_thresh: float
        Variance threshold to remove features

    Returns:
    --------
    Data_train and Data_test: pandas.DataFrame
        Data after preprocessing
    """
    def __init__(self, data_train, data_test, activity_col, ID, smile_col,  var_thresh=0.05):
        self.data_train= data_train.copy()
        self.data_test= data_test.copy()
        self.activity_col = activity_col
        self.var_thresh = var_thresh
        self.ID = ID
        self.smile_col = smile_col
    
    # 1. Remove duplicate data - rows and columns
    def Duplicate_data(self):
        
        # 1.1. Duplicate rows: xóa hàng bị trùng 
        # train
        dup_rows = self.data_train.duplicated()
        print(f"Total duplicated rows-train: {(dup_rows == True).sum()}")
        print("Data train before drop duplicates:", self.data_train.shape[0])
        self.data_train.drop_duplicates(inplace = True)
        self.data_train.reset_index(drop = True, inplace = True)
        print("Data train after drop duplicates:", self.data_train.shape[0])
        print(75*"*")
        self.train_dup = (dup_rows == True).sum()
        
        # test
        dup_rows = self.data_test.duplicated()
        print(f"Total duplicated rows-test: {(dup_rows == True).sum()}")
        print("Data test before drop duplicates:", self.data_test.shape[0])
        self.data_test.drop_duplicates(inplace = True)
        self.data_test.reset_index(drop = True, inplace = True)
        print("Data test after drop duplicates:", self.data_test.shape[0])
        print(75*"*")
        self.test_dup = (dup_rows == True).sum()
       
        
    # 2. Check Variance Threshold: 
    def Variance_Threshold(self):
        y = self.data_train[self.activity_col]
        X = self.data_train.drop([self.activity_col], axis =1)
        print(X.shape, y.shape)
        while True:
            try:
               # Define thresholds to check
                thresholds = np.arange(0.0, 1, 0.05)
                # Apply transform with each threshold
                results = list()
                for t in thresholds:
                # define the transform
                    transform = VarianceThreshold(threshold=t)
                # transform the input data
                    X_sel = transform.fit_transform(X)
                # determine the number of input features
                    n_features = X_sel.shape[1]
                    print('>Threshold=%.2f, Features=%d' % (t, n_features)) 
                # store the result
                    results.append(n_features)
                break
            except:
                break
        # plot the threshold vs the number of selected features
        sns.set('notebook')
         # plot the threshold vs the number of selected features
        plt.figure(figsize=(14,8))
        plt.title("Variance analysis", fontsize = 24, weight = 'semibold')
        plt.xlabel("Variance threshold", fontsize = 16)
        plt.ylabel("Number of features", fontsize = 16)
       
        plt.plot(thresholds[:len(results)], results)
        #plt.savefig(self.SAVE_PREFIX +"var.png", dpi = 600)
        plt.show()

    # 3. Remove variance Threshold:  
    def remove_low_variance(self, thresh):
        selector = VarianceThreshold(thresh)
        selector.fit(self.data_train)
        features = selector.get_support(indices = False)
        features[0]=True # target can not be removed
        self.data_train = self.data_train.loc[:, features]
        #lst = self.data_train.columns
        self.data_test = self.data_test.loc[:, features]
        print(75*"*")
        self.var = features.sum()
    

###########################################################################################
# MISSING VALUES
    # 4. Find Missing Percentage: 
    
    # def find_missing_percent(self, data):
    #     """
    #     Returns dataframe containing the total missing values and percentage of total
    #     missing values of a column.
    #     """
    #     miss_df = pd.DataFrame({'ColumnName':[],'TotalMissingVals':[],'PercentMissing':[]})
    #     for col in data.columns:
    #         sum_miss_val = data[col].isnull().sum()
    #         percent_miss_val = round((sum_miss_val/data.shape[0])*100,2)
    #         missinginfo = {"ColumnName" : col, "TotalMissingVals" : sum_miss_val, "PercentMissing" : percent_miss_val}
    #         miss_df = miss_df.append(missinginfo, ignore_index = True)

    #     miss_df = miss_df[miss_df["PercentMissing"] > 0.0]
    #     miss_df = miss_df.reset_index(drop = True)
    #     return miss_df
    def find_missing_percent(self,data):
        """
        Returns DataFrame containing the total missing values and percentage of total
        missing values of a column.
        """
        missing_info = []
        for col in data.columns:
            sum_miss_val = data[col].isnull().sum()
            percent_miss_val = round((sum_miss_val / data.shape[0]) * 100, 2)
            if percent_miss_val > 0.0:
                missing_info.append({
                    "ColumnName": col,
                    "TotalMissingVals": sum_miss_val,
                    "PercentMissing": percent_miss_val
                })

        miss_df = pd.DataFrame(missing_info)
        return miss_df
    
    # 5. Handle missing values
    def Missing_value_cleaning(self):
        miss_df = self.find_missing_percent(self.data_train)
        print(miss_df)
        # Remove columns with high missing percentage
        miss_thresh= 50 # modify
        
        drop_cols = miss_df[miss_df['PercentMissing'] > miss_thresh].ColumnName.tolist()
        print("Drop_cols",  drop_cols)
        
        
            
        self.data_train.drop(drop_cols, axis =1, inplace = True)
        self.data_test.drop(drop_cols, axis =1, inplace = True)
         

        
        print("Total missing value-train", self.data_train.isnull().sum().sum())
        print("Total missing value-test", self.data_test.isnull().sum().sum())
        
        null_data_train = self.data_train[self.data_train.isnull().any(axis=1)]
        display(null_data_train)
        print("Total row-train with missing value", null_data_train.shape[0])
        
        null_data_test = self.data_test[self.data_test.isnull().any(axis=1)]
        display(null_data_test)
        print("Total row-test with missing value", null_data_test.shape[0])
        
        if null_data_train.shape[0] == 0:
            self.data_test = self.data_test.dropna(inplace = False)
            self.data_test.reset_index(drop = True, inplace = True)
        else:
            #S = (input("Do you want to remove rows have missing values:").title())
            S = 'N'
            if S == 'Y':
                Data_train = self.data_train.dropna(inplace = False)
                Data_train.reset_index(drop = True, inplace = True)

                Data_test = self.data_test.dropna(inplace = False)
                Data_test.reset_index(drop = True, inplace = True)
            else:
                while True:
                    try:
                        #M = int(input("Please select type of Imputation:\n\t1:Mean\n\t2:Median\n\t3:Mode\n\t4:KNNImputer\n\t5:MICE\n"))
                        M = 4
                        break
                    except:
                        print("Error value!")
                if M == 1:
                    imp=SimpleImputer(missing_values=np.NaN, strategy='mean')
                elif M == 2:
                    imp=SimpleImputer(missing_vlues=np.NaN, strategy='median')
                elif M == 3:
                    imp=SimpleImputer(missing_values=np.NaN, strategy='mode')
                elif M == 4:
                    while True:
                        try:
                            #n_neighbors = int(input("Please input n_neighbors:"))
                            n_neighbors = 5
                            break
                        except:
                            print("Error value!")
                    imp=KNNImputer(n_neighbors=n_neighbors) 
                else:
                    A = int(input("Please select Algorithm of Imputation:\n\t1:BayesianRidge\n\t2:RandomForest\n"))
                    if A == 1:
                        estimator = BayesianRidge()
                    else:
                        estimator = GradientBoostingRegressor(random_state = 42)
                    imp= IterativeImputer(random_state=42, missing_values=np.NaN, estimator= estimator)
                
                imp.fit(self.data_train)
                
               
                # train
                Data_train=pd.DataFrame(imp.transform(self.data_train))
                Data_train.columns=self.data_train.columns
                Data_train.index=self.data_train.index
                display(Data_train.shape)
                # test
                Data_test=pd.DataFrame(imp.transform(self.data_test))
                Data_test.columns=self.data_test.columns
                Data_test.index=self.data_test.index
                display(Data_test.shape)
                print(75*"*")
            self.data_train = Data_train
            self.data_test = Data_test
      
    
    # 6. Convert low unique columns to integer
 
    def Nomial(self):
        data = self.data_train.loc[:, (self.data_train.nunique() <10).values & (self.data_train.max() <10).values]  #feature with unique < 10 and max value <10 will set to be int64
        col = data.columns #select columns with int64
        #set all  col_olumn to int64
        #print("DINH TINH:", col)
        #savefile("Nomial_Col.txt", col)
        self.data_train[col]=self.data_train[col].astype('int64')  
        self.data_test[col]=self.data_test[col].astype('int64')    
        display(self.data_train.head(5))
    
    
        
    # 7. Activate 
    def fit(self):
        self.Duplicate_data()        
        id_smile_train = self.data_train[[self.ID, self.smile_col]]
        id_smile_test = self.data_test[[self.ID, self.smile_col]]
        self.data_train = self.data_train.drop([self.ID, self.smile_col], axis=1)
        self.data_test = self.data_test.drop([self.ID, self.smile_col], axis=1)
        self.Missing_value_cleaning() # missing values must be handled before feature selection

        self.Variance_Threshold()
        self.remove_low_variance(thresh= self.var_thresh)
        self.Nomial()
        self.data_train_final = pd.concat([id_smile_train, self.data_train], axis =1)
        self.data_test_final = pd.concat([id_smile_test, self.data_test], axis =1)