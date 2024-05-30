# import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, KBinsDiscretizer
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from .Data_preprocess import Data_preprocess, Data_preprocess_Scaffold #
import warnings
warnings.filterwarnings(action='ignore')
sns.set('notebook')

class Univariate_Outliers(Data_preprocess):
    """
    - Check quality features
    - Remove univariate outliers: using Interquartile Range (IQR)
    - Find the suitable data transform  method to minimize the amount of outliers to be deleted.
      . Imputation
      . Winsorzation
      . Transformation
    - KBIN Discretizer

    Input:
    ------
    Data_train: pandas.DataFrame
        Data for training model after preprocessing.
    Data_test: pandas.DataFrame
        Data for external validation after preprocessing
    acitivity_col: str
        Name of acitivity columns (pIC50, pChEMBL Value)
    handling_method: string ('Winsorization', 'Imputation', 'Transformation')
        Handling outliers method
    transform_method: string ('Uniform Transformer', 'Gaussian Transformer', 'Power Transformer')
        Handling outliers method

    Returns:
    --------
    Data_train, Data_test: pandas.DataFrame
        Data after handling outliers
 
    """
    def __init__(self, data_train, data_test, activity_col, handling_method = 'Transformation', transform_method = 'Uniform Transformer', Kbin_handling ='Y', variance_threshold = 'Y'):
        self.activity_col      = activity_col
        self.handling_method   = handling_method
        self.transform_method  = transform_method
        self.Kbin_handling     = Kbin_handling
        self.variance_threshold = variance_threshold
        
        self.data_train_0      = data_train
        self.data_test_0       = data_test
        
        self.data_train        = self.data_train_0.copy()
        self.data_test         = self.data_test_0.copy()
        
        self.scl1 = PowerTransformer()
        self.scl2 = QuantileTransformer(output_distribution = "normal")
        self.scl3 = QuantileTransformer(output_distribution = "uniform")
        
############# 1. Check number of outliers would be removed ####################################    
    def Check_IQR(self):
        self.df_train = self.data_train.copy()
        self.df_test = self.data_test.copy()
         
        for col_name in self.df_train.drop([self.activity_col], axis = 1).select_dtypes('float').columns:
            q1 = self.df_train[col_name].quantile(0.25)
            q3 = self.df_train[col_name].quantile(0.75)
            iqr = q3 - q1
            low = q1-1.5*iqr
            high = q3+1.5*iqr 
            self.df_train = self.df_train[(self.df_train[col_name] <= high) & (self.df_train[col_name] >= low)]
            self.df_test = self.df_test[(self.df_test[col_name] <= high) & (self.df_test[col_name] >= low)]
        print("Total data remove on Train", self.data_train_0.shape[0] -self.df_train.shape[0])
        print("Total data remove on Test", self.data_test_0.shape[0] -self.df_test.shape[0])
        self.data_train_clean = self.df_train
        self.data_test_clean = self.df_test
       
    # Check good or bad features. Good features mean that there would be no outliers  
    def Quality_features(self):
        self.good = []
        self.bad = []
        self.df_train = self.data_train.copy()
        self.df_test = self.data_test.copy()
        for col_name in self.df_train.drop([self.activity_col], axis = 1).select_dtypes("float64").columns:
            q1 = self.df_train[col_name].quantile(0.25)
            q3 = self.df_train[col_name].quantile(0.75)
            iqr = q3-q1     
            remove = self.data_train.shape[0] - (self.df_train[(self.df_train[col_name] <= (q3+1.5*iqr)) & (self.df_train[col_name] >= (q1-1.5*iqr))]).shape[0]
            if remove == 0:
                self.good.append(col_name)
            else:
                self.bad.append(col_name)
        print(f"Number of good features: {len(self.good)}")
        print(f"Number of bad features with data remove > 0: {len(self.bad)}")
        print("*"*75)
    
    def Check_univariate_outliers(self):
        self.Check_IQR()
        self.Quality_features()
        
############# 2. Method Selection ###########################################################
    def Method(self):
        if self.handling_method == 'Winsorization':
            self.Winsorization()
        elif self.handling_method == 'Imputation':
            self.Impute_nan()
        elif self.handling_method == 'Transformation':
            self.Transformation()
        else:
            return None
        
############# 3. Winsorization ###########################################################        
    def Winsorization(self):
        print("Handling with Winsorization method")
        self.df_train = self.data_train.copy()
        self.df_test = self.data_test.copy()
        for col_name in self.df_train.drop([self.activity_col], axis = 1).select_dtypes(include="float64").columns:
            q1 = self.df_train[col_name].quantile(0.25)
            q3 = self.df_train[col_name].quantile(0.75)
            iqr = q3-q1
            self.df_train.loc[(self.df_train[col_name] <= (q1-1.5*iqr)), col_name] = q1-1.5*iqr
            self.df_train.loc[(self.df_train[col_name] >= (q3+1.5*iqr)), col_name] = q3+1.5*iqr
            #for test
            self.df_test.loc[(self.df_test[col_name] <= (q1-1.5*iqr)), col_name] = q1-1.5*iqr
            self.df_test.loc[(self.df_test[col_name] >= (q3+1.5*iqr)), col_name] = q3+1.5*iqr
        self.data_train = self.df_train
        self.data_test = self.df_test
        self.Check_univariate_outliers()
    
############# 4. Imputation - Outlier == NaN ########################################################### 
    
    def Impute_nan(self):
        print("Handling with Imputation method")
        for col_name in self.data_train.drop([self.activity_col], axis = 1).select_dtypes(include="float64").columns:
            q1 = self.data_train[col_name].quantile(0.25)
            q3 = self.data_train[col_name].quantile(0.75)
            iqr = q3-q1
            self.data_train.loc[(self.data_train[col_name] < (q1-1.5*iqr)), col_name] = np.nan
            self.data_train.loc[(self.data_train[col_name] > (q3+1.5*iqr)), col_name] = np.nan
            #for test
            self.data_test.loc[(self.data_test[col_name] < (q1-1.5*iqr)), col_name] = np.nan
            self.data_test.loc[(self.data_test[col_name] > (q3+1.5*iqr)), col_name] = np.nan
        self.Missing_value_cleaning() #Gọi lại hàm ở Class preprocess để chọn phương pháp impute phù hợp
        self.Check_univariate_outliers()
        self.Method()
    
############# 5. Transformation ###########################################################    
 
    def Transformation(self):
       
        if self.transform_method == 'Power Transformer':
            self.scl =self.scl1
            print("Power Transformer technique")
        elif self.transform_method == 'Gaussian Transformer':
            self.scl =self.scl2
            print("Gaussian Transformer technique")
        elif self.transform_method == 'Uniform Transformer':
            self.scl =self.scl3
            print("Uniform Transformer technique")
        else:
            return None

        # Train
        self.data_train_good = self.data_train.drop(self.bad, axis = 1)
        self.data_train_bad = self.data_train[self.bad]
        self.real_bad = self.bad.copy()
        if len(self.real_bad)==0:
            self.data_train = self.data_train
            self.data_test = self.data_test 
        
        else:
            self.scl.fit(self.data_train_bad)

            #with open(SAVE_PREFIX + 'scl.pkl','wb') as f:
                #pickle.dump(self.scl,f)

            self.bad_new = pd.DataFrame(self.scl.transform(self.data_train_bad), columns = self.bad)
            self.data_train = pd.concat([self.data_train_good,self.bad_new], axis = 1)

            #test

            self.data_test_good = self.data_test.drop(self.bad, axis = 1)
            self.data_test_bad = self.data_test[self.bad]

            self.bad_new = pd.DataFrame(self.scl.transform(self.data_test_bad),  columns = self.bad)  
            self.data_test = pd.concat([self.data_test_good,self.bad_new], axis = 1)


            self.Check_univariate_outliers()
            
            if self.Kbin_handling == "Y":
                self.KBin()
            else:
                pass
    
   

    def KBin (self):
        print("Handling with KBin method")
        #Train
        self.data_train_good = self.data_train.drop(self.bad, axis = 1)
        self.data_train_bad = self.data_train[self.bad]
        bad_col_kbin = self.data_train_bad.columns
        
        print("//////", self.bad, len(self.bad))
        if len(self.bad) !=0:
            while True:
                try:
                    #self.n_bins = int(input("Please input number of bins"))
                    self.n_bins = 3
                    #self.encode = input("Please input type of encode")
                    self.encode = 'ordinal'
                    #self.strategy = input("Please input type of strategy")
                    self.strategy = 'quantile'
                    kst = KBinsDiscretizer(n_bins = 3, encode = self.encode, strategy = self.strategy)
                    break
                except:
                    print("Error")
            kst.fit(self.data_train_bad)
            self.bad_new = pd.DataFrame(kst.transform(self.data_train_bad)).astype("int64")
            self.bad_new.columns=["Kbin"+str(i) for i in range(1, len(self.bad_new.columns) + 1)]
            self.data_train_clean = pd.concat([self.data_train_good,self.bad_new], axis = 1)
            self.data_train = self.data_train_clean
        
            #test

            self.data_test_good = self.data_test.drop(self.bad, axis = 1)
            self.data_test_bad = self.data_test[self.bad]
            self.bad_new = pd.DataFrame(kst.transform(self.data_test_bad)).astype("int64")
            self.bad_new.columns=["Kbin"+str(i) for i in range(1, len(self.bad_new.columns) + 1)]

            self.data_test_clean = pd.concat([self.data_test_good,self.bad_new], axis = 1)
            self.data_test = self.data_test_clean

            self.Check_univariate_outliers()
            if self.variance_threshold == 'Y':
                self.remove_low_variance(thresh = 0.05)
            

    def fit(self):
        print('Remove by IQR without handling')
        self.Check_univariate_outliers()
        self.Method()
        
    
class Mutivariate_Outliers():

    """
    Remove multivariate outliers by suitable method 

    Input:
    ------
    data_train: pandas.DataFrame
        Data for training model after univariate outliers handling
    data_test: pandas.DataFrame
        Data for external validation after univariate outliers handling
    method: string 
        Include: 'LocalOutlierFactor', 'IsolationForest','OneClassSVM','Robust Covariance',
        'Emperical Covariance', 'Compare'
    n_jobs: int (-1 for all cpu)
        Number of cpu using for cleaning outliers
    Returns:
    --------
    data_train: pandas.DataFrame
        Data for training model after remove multivariate outliers
    data_test: pandas.DataFrame
        Data for external validation after remove multivariate outliers
    """
    def __init__(self, data_train, data_test, method = 'LocalOutlierFactor', n_jobs =-1):
        self.data_train = data_train
        self.data_test = data_test
        self.method = method
        self.n_jobs = n_jobs
    
    # 1. LOF
    def LOF(self):
        self.data_train_LOF = self.data_train.copy()
        self.data_test_LOF = self.data_test.copy()
        
        LOF = LocalOutlierFactor(n_neighbors = 20, n_jobs =self.n_jobs)
        LOF.fit(self.data_train_LOF)
        self.Outlier_LOF = self.data_train_LOF[LOF.fit_predict(self.data_train_LOF) == -1]
        self.Data_train_LOF = self.data_train_LOF[LOF.fit_predict(self.data_train_LOF) != -1]
        print(f"Total outlier remove by LOF:", self.Outlier_LOF.shape[0])
        #Test
        LOF = LocalOutlierFactor(n_neighbors = 20, novelty = True, n_jobs =self.n_jobs)
        LOF.fit(self.data_train_LOF)   
        self.Data_test_LOF = self.data_test_LOF[LOF.predict(self.data_test_LOF) != -1]
        
        

    # 2. Isolation forest  
    def Ist_for(self):
        self.data_train_Ist_for = self.data_train.copy()
        self.data_test_Ist_for = self.data_test.copy()
        
        Iso_for = IsolationForest(n_estimators=100, contamination = 'auto',random_state=42, n_jobs =self.n_jobs)
        Iso_for.fit(self.data_train_Ist_for)
        self.Outlier_iso = self.data_train_Ist_for[Iso_for.predict(self.data_train_Ist_for) == -1]
        
        self.Data_train_iso = self.data_train_Ist_for[Iso_for.predict(self.data_train_Ist_for) != -1]
        self.Data_test_iso = self.data_test_Ist_for[Iso_for.predict(self.data_test_Ist_for) != -1]
        print(f"Total outlier remove by Isolation forest:", self.Outlier_iso.shape[0])

    # 3. One class SVM    
    def o_SVM(self):
        self.data_train_o_SVM = self.data_train.copy()
        self.data_test_o_SVM = self.data_test.copy()
        
        o_SVM = OneClassSVM()
        o_SVM.fit(self.data_train_o_SVM)
        self.Outlier_osvm = self.data_train_o_SVM[o_SVM.predict(self.data_train_o_SVM) == -1]
        
        self.Data_train_osvm = self.data_train_o_SVM[o_SVM.predict(self.data_train_o_SVM) != -1]
        self.Data_test_osvm = self.data_test_o_SVM[o_SVM.predict(self.data_test_o_SVM) != -1]
        print(f"Total outlier remove by One Class SVM:", self.Outlier_osvm.shape[0])
    
    # 4. Robust Covariance 
    def robust_cov(self):
        self.data_train_r_cov = self.data_train.copy()
        self.data_test_r_cov = self.data_test.copy()
      
        robust_cov = EllipticEnvelope(contamination= 0.1, random_state=42)
        robust_cov.fit(self.data_train_r_cov)
        self.Outlier_rcov = self.data_train_r_cov[robust_cov.predict(self.data_train_r_cov) == -1]
        
        self.Data_train_rcov = self.data_train_r_cov[robust_cov.predict(self.data_train_r_cov) != -1]
        self.Data_test_rcov = self.data_test_r_cov[robust_cov.predict(self.data_test_r_cov) != -1]
        print(f"Total outlier remove by Robust covariance:", self.Outlier_rcov.shape[0])

    # 5. Emperical Covariance    
    def emp_cov(self):
        self.data_train_e_cov = self.data_train.copy()
        self.data_test_e_cov = self.data_test.copy()
        
        emp_cov = EllipticEnvelope(contamination= 0.1, support_fraction=0.1, random_state=42)
        emp_cov.fit(self.data_train_e_cov)
        self.Outlier_ecov = self.data_train_e_cov[emp_cov.predict(self.data_train_e_cov) == -1]
        
        self.Data_train_ecov = self.data_train_e_cov[emp_cov.predict(self.data_train_e_cov) != -1]
        self.Data_test_ecov = self.data_test_e_cov[emp_cov.predict(self.data_test_e_cov) != -1]
        print(f"Total outlier remove by Emperical covariance:", self.Outlier_ecov.shape[0])
    
    # 6. Compare method 
    def Compare_multivariate_method(self):
        self.LOF()
        self.Ist_for()
        self.o_SVM()
        self.robust_cov()
        self.emp_cov()
        sns.set('notebook')
        Models =  [('Local Outlier Factor', self.Outlier_LOF.shape[0]), ('Isolation forest', self.Outlier_iso.shape[0]),
                  ('One Class SVM', self.Outlier_osvm.shape[0]),('Robust covariance', self.Outlier_rcov.shape[0]),('Emperical covariance', self.Outlier_ecov.shape[0])]
        for name, N_out in Models:
            plt.rcParams["figure.figsize"] = (14,8)
            plt.bar(name,N_out)
        plt.title("Multivariate outlier detection", fontsize = 24, weight ='semibold')
        plt.ylabel("Number of Outliers", fontsize = 16)
        #plt.savefig("Multivariate outlier detection.png", dpi=600, transparent = True)
    
    def fit(self):     
        if self.method == 'LocalOutlierFactor':
            self.LOF()
            self.data_train = self.Data_train_LOF
            self.data_test = self.Data_test_LOF
        elif self.method == 'IsolationForest':
            self.Ist_for()
            self.data_train = self.Data_train_iso
            self.data_test = self.Data_test_iso
        elif self.method == 'OneClassSVM':
            self.o_SVM()
            self.data_train = self.Data_train_osvm
            self.data_test = self.Data_test_osvm
        elif self.method == 'Robust Covariance':
            self.robust_cov()
            self.data_train = self.Data_train_rcov
            self.data_test = self.Data_test_rcov
        elif self.method == 'Emperical Covariance':
            self.emp_cov()
            self.data_train = self.Data_train_ecov
            self.data_test = self.Data_test_ecov
        else:
            self.Compare_multivariate_method()


class Univariate_Outliers_Scaffold(Data_preprocess_Scaffold):
    """
    - Check quality features
    - Remove univariate outliers: using Interquartile Range (IQR)
    - Find the suitable data transform  method to minimize the amount of outliers to be deleted.
      . Imputation
      . Winsorzation
      . Transformation
    - KBIN Discretizer

    Input:
    ------
    Data_train: pandas.DataFrame
        Data for training model after preprocessing.
    Data_test: pandas.DataFrame
        Data for external validation after preprocessing
    acitivity_col: str
        Name of acitivity columns (pIC50, pChEMBL Value)
    handling_method: string ('Winsorization', 'Imputation', 'Transformation')
        Handling outliers method
    transform_method: string ('Uniform Transformer', 'Gaussian Transformer', 'Power Transformer')
        Handling outliers method

    Returns:
    --------
    Data_train, Data_test: pandas.DataFrame
        Data after handling outliers
 
    """
    def __init__(self, data_train, data_test, activity_col,ID, smile_col, handling_method = 'Transformation', transform_method = 'Uniform Transformer', Kbin_handling ='Y', variance_threshold = 'Y'):
        self.activity_col      = activity_col
        self.handling_method   = handling_method
        self.transform_method  = transform_method
        self.Kbin_handling     = Kbin_handling
        self.variance_threshold = variance_threshold
        self.ID = ID
        self.smile_col = smile_col
        self.data_train_0      = data_train
        self.data_test_0       = data_test
        self.id_smile_train = data_train[[self.ID, self.smile_col]]
        self.id_smile_test = data_test[[self.ID, self.smile_col]]
        self.data_train        = self.data_train_0.drop([self.ID, self.smile_col],axis=1)
        self.data_test         = self.data_test_0.drop([self.ID, self.smile_col],axis =1 )
        
        self.scl1 = PowerTransformer()
        self.scl2 = QuantileTransformer(output_distribution = "normal")
        self.scl3 = QuantileTransformer(output_distribution = "uniform")
        
############# 1. Check number of outliers would be removed ####################################    
    def Check_IQR(self):
        self.df_train = self.data_train.copy()
        self.df_test = self.data_test.copy()
         
        for col_name in self.df_train.drop([self.activity_col], axis = 1).select_dtypes('float').columns:
            q1 = self.df_train[col_name].quantile(0.25)
            q3 = self.df_train[col_name].quantile(0.75)
            iqr = q3 - q1
            low = q1-1.5*iqr
            high = q3+1.5*iqr 
            self.df_train = self.df_train[(self.df_train[col_name] <= high) & (self.df_train[col_name] >= low)]
            self.df_test = self.df_test[(self.df_test[col_name] <= high) & (self.df_test[col_name] >= low)]
        print("Total data remove on Train", self.data_train_0.shape[0] -self.df_train.shape[0])
        print("Total data remove on Test", self.data_test_0.shape[0] -self.df_test.shape[0])
        self.data_train_clean = self.df_train
        self.data_test_clean = self.df_test
       
    # Check good or bad features. Good features mean that there would be no outliers  
    def Quality_features(self):
        self.good = []
        self.bad = []
        self.df_train = self.data_train.copy()
        self.df_test = self.data_test.copy()
        for col_name in self.df_train.drop([self.activity_col], axis = 1).select_dtypes("float64").columns:
            q1 = self.df_train[col_name].quantile(0.25)
            q3 = self.df_train[col_name].quantile(0.75)
            iqr = q3-q1     
            remove = self.data_train.shape[0] - (self.df_train[(self.df_train[col_name] <= (q3+1.5*iqr)) & (self.df_train[col_name] >= (q1-1.5*iqr))]).shape[0]
            if remove == 0:
                self.good.append(col_name)
            else:
                self.bad.append(col_name)
        print(f"Number of good features: {len(self.good)}")
        print(f"Number of bad features with data remove > 0: {len(self.bad)}")
        print("*"*75)
    
    def Check_univariate_outliers(self):
        self.Check_IQR()
        self.Quality_features()
        
############# 2. Method Selection ###########################################################
    def Method(self):
        if self.handling_method == 'Winsorization':
            self.Winsorization()
        elif self.handling_method == 'Imputation':
            self.Impute_nan()
        elif self.handling_method == 'Transformation':
            self.Transformation()
        else:
            return None
        
############# 3. Winsorization ###########################################################        
    def Winsorization(self):
        print("Handling with Winsorization method")
        self.df_train = self.data_train.copy()
        self.df_test = self.data_test.copy()
        for col_name in self.df_train.drop([self.activity_col], axis = 1).select_dtypes(include="float64").columns:
            q1 = self.df_train[col_name].quantile(0.25)
            q3 = self.df_train[col_name].quantile(0.75)
            iqr = q3-q1
            self.df_train.loc[(self.df_train[col_name] <= (q1-1.5*iqr)), col_name] = q1-1.5*iqr
            self.df_train.loc[(self.df_train[col_name] >= (q3+1.5*iqr)), col_name] = q3+1.5*iqr
            #for test
            self.df_test.loc[(self.df_test[col_name] <= (q1-1.5*iqr)), col_name] = q1-1.5*iqr
            self.df_test.loc[(self.df_test[col_name] >= (q3+1.5*iqr)), col_name] = q3+1.5*iqr
        self.data_train = self.df_train
        self.data_test = self.df_test
        self.Check_univariate_outliers()
    
############# 4. Imputation - Outlier == NaN ########################################################### 
    
    def Impute_nan(self):
        print("Handling with Imputation method")
        for col_name in self.data_train.drop([self.activity_col], axis = 1).select_dtypes(include="float64").columns:
            q1 = self.data_train[col_name].quantile(0.25)
            q3 = self.data_train[col_name].quantile(0.75)
            iqr = q3-q1
            self.data_train.loc[(self.data_train[col_name] < (q1-1.5*iqr)), col_name] = np.nan
            self.data_train.loc[(self.data_train[col_name] > (q3+1.5*iqr)), col_name] = np.nan
            #for test
            self.data_test.loc[(self.data_test[col_name] < (q1-1.5*iqr)), col_name] = np.nan
            self.data_test.loc[(self.data_test[col_name] > (q3+1.5*iqr)), col_name] = np.nan
        self.Missing_value_cleaning() #Gọi lại hàm ở Class preprocess để chọn phương pháp impute phù hợp
        self.Check_univariate_outliers()
        self.Method()
    
############# 5. Transformation ###########################################################    
 
    def Transformation(self):
       
        if self.transform_method == 'Power Transformer':
            self.scl =self.scl1
            print("Power Transformer technique")
        elif self.transform_method == 'Gaussian Transformer':
            self.scl =self.scl2
            print("Gaussian Transformer technique")
        elif self.transform_method == 'Uniform Transformer':
            self.scl =self.scl3
            print("Uniform Transformer technique")
        else:
            return None

        # Train
        self.data_train_good = self.data_train.drop(self.bad, axis = 1)
        self.data_train_bad = self.data_train[self.bad]
        self.real_bad = self.bad.copy()
        if len(self.real_bad)==0:
            self.data_train = self.data_train
            self.data_test = self.data_test 
        
        else:
            self.scl.fit(self.data_train_bad)

            #with open(SAVE_PREFIX + 'scl.pkl','wb') as f:
                #pickle.dump(self.scl,f)

            self.bad_new = pd.DataFrame(self.scl.transform(self.data_train_bad), columns = self.bad)
            self.data_train = pd.concat([self.data_train_good,self.bad_new], axis = 1)

            #test

            self.data_test_good = self.data_test.drop(self.bad, axis = 1)
            self.data_test_bad = self.data_test[self.bad]

            self.bad_new = pd.DataFrame(self.scl.transform(self.data_test_bad),  columns = self.bad)  
            self.data_test = pd.concat([self.data_test_good,self.bad_new], axis = 1)


            self.Check_univariate_outliers()
            
            if self.Kbin_handling == "Y":
                self.KBin()
            else:
                pass
    
   

    def KBin (self):
        print("Handling with KBin method")
        #Train
        self.data_train_good = self.data_train.drop(self.bad, axis = 1)
        self.data_train_bad = self.data_train[self.bad]
        bad_col_kbin = self.data_train_bad.columns
        
        print("//////", self.bad, len(self.bad))
        if len(self.bad) !=0:
            while True:
                try:
                    #self.n_bins = int(input("Please input number of bins"))
                    self.n_bins = 3
                    #self.encode = input("Please input type of encode")
                    self.encode = 'ordinal'
                    #self.strategy = input("Please input type of strategy")
                    self.strategy = 'quantile'
                    kst = KBinsDiscretizer(n_bins = 3, encode = self.encode, strategy = self.strategy)
                    break
                except:
                    print("Error")
            kst.fit(self.data_train_bad)
            self.bad_new = pd.DataFrame(kst.transform(self.data_train_bad)).astype("int64")
            self.bad_new.columns=["Kbin"+str(i) for i in range(1, len(self.bad_new.columns) + 1)]
            self.data_train_clean = pd.concat([self.data_train_good,self.bad_new], axis = 1)
            self.data_train = self.data_train_clean
        
            #test

            self.data_test_good = self.data_test.drop(self.bad, axis = 1)
            self.data_test_bad = self.data_test[self.bad]
            self.bad_new = pd.DataFrame(kst.transform(self.data_test_bad)).astype("int64")
            self.bad_new.columns=["Kbin"+str(i) for i in range(1, len(self.bad_new.columns) + 1)]

            self.data_test_clean = pd.concat([self.data_test_good,self.bad_new], axis = 1)
            self.data_test = self.data_test_clean

            self.Check_univariate_outliers()
            if self.variance_threshold == 'Y':
                self.remove_low_variance(thresh = 0.05)
            

    def fit(self):
        print('Remove by IQR without handling')
        self.Check_univariate_outliers()
        self.Method()
        self.data_train_final = pd.concat([self.id_smile_train, self.data_train],axis = 1)
        self.data_test_final = pd.concat([self.id_smile_test, self.data_test],axis = 1)
    
class Mutivariate_Outliers_Scaffold():

    """
    Remove multivariate outliers by suitable method 

    Input:
    ------
    data_train: pandas.DataFrame
        Data for training model after univariate outliers handling
    data_test: pandas.DataFrame
        Data for external validation after univariate outliers handling
    method: string 
        Include: 'LocalOutlierFactor', 'IsolationForest','OneClassSVM','Robust Covariance',
        'Emperical Covariance', 'Compare'
    n_jobs: int (-1 for all cpu)
        Number of cpu using for cleaning outliers
    Returns:
    --------
    data_train: pandas.DataFrame
        Data for training model after remove multivariate outliers
    data_test: pandas.DataFrame
        Data for external validation after remove multivariate outliers
    """
    def __init__(self, data_train, data_test, ID, smile_col,  method = 'LocalOutlierFactor', n_jobs =-1):
        self.data_train = data_train
        self.data_test = data_test
        self.method = method
        self.n_jobs = n_jobs
        self.ID = ID
        self.smile_col = smile_col
        #self.data_train =  self.data_train.drop([self.ID, self.smile_col], axis = 1)
        #elf.data_test =  self.data_test.drop([self.ID, self.smile_col], axis = 1)
        #self.id_smile = 
    
    # 1. LOF
    def LOF(self):
        self.data_train_LOF = self.data_train.drop([self.ID, self.smile_col], axis = 1)
        self.data_test_LOF = self.data_test.drop([self.ID, self.smile_col], axis = 1)
        
        LOF = LocalOutlierFactor(n_neighbors = 20, n_jobs =self.n_jobs)
        LOF.fit(self.data_train_LOF)
        self.Outlier_LOF = self.data_train_LOF[LOF.fit_predict(self.data_train_LOF) == -1]
        #self.Data_train_LOF = self.data_train_LOF[LOF.fit_predict(self.data_train_LOF) != -1]
        self.Data_train_LOF = self.data_train[LOF.fit_predict(self.data_train_LOF) != -1]
        print(f"Total outlier remove by LOF:", self.Outlier_LOF.shape[0])
        #Test
        LOF = LocalOutlierFactor(n_neighbors = 20, novelty = True, n_jobs =self.n_jobs)
        LOF.fit(self.data_train_LOF)   
        #self.Data_test_LOF = self.data_test_LOF[LOF.predict(self.data_test_LOF) != -1]
        self.Data_test_LOF = self.data_test[LOF.predict(self.data_test_LOF) != -1]
        
    
    def fit(self):     
        self.LOF()
        self.data_train = self.Data_train_LOF
        self.data_test = self.Data_test_LOF