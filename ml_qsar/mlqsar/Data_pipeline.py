import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from .Data_integration import Data_Integration, Data_Integration_Scaffold
from .Data_preprocess import Data_preprocess,Data_preprocess_Scaffold
from .Outliers_handling import Univariate_Outliers, Mutivariate_Outliers, Univariate_Outliers_Scaffold,Mutivariate_Outliers_Scaffold
from .Rescale import rescale, rescale_scaffold
from .Feature_selection import feature_selection_pipeline, feature_selection_pipeline_scaffold
from .Model_selection import model_selection, model_selection_scaffold
from .Meta_analysis import statistic_data, statistic_test
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
sns.set('notebook')

class QSAR_pipeline:
    """
    - QSAR data pipeline includes:
        - Data integration
        - Data preprocessing
        - Outliers handling
        - Rescaling
        - Feature selection
        - Model building
        - Statistic analysis

    Input:
    ------
    data_path: path
        Directory contains all data types to run QSAR pipeline
    acitivity_col: str
        Name of acitivity columns (pIC50, pChEMBL Value)
    var_thresh: float (default = 0.05)
         Variance threshold to remove features
    save_data: bool (default = False)
        True if want to save the posthoc analysis data
    task_type: str ('C' or 'R')
        Classification (C) or Regression (R)
    target_thresh: int
        Threshold to transform numerical columns to binary
    scoring: str
        if task_type = (C): 'f1', 'average_precision', 'recall'
        if task_type = (R): 'r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'
    refinement: bool (default = False)
        True to remove results equal to zero in boxplot visualization
    
    posthoc_method: list of str (['Wilcoxon'], ['Mannwhitney'], ['Wilcoxon', 'Mannwhitney']) 
        Method for posthoc analysis
    kind_analysis: list of str (['Meta', 'Subgroup']):
        Kind of posthoc analysis
    
    Returns:
    --------
    Data_posthoc: new directory (data_dir in posthoc test)
        Directory contains data for posthoc analysis
    Meta: new directory
        Directory contains posthoc data and figures
    

    --------
    
    """
    def __init__(self, data_path, activity_col, kind_analysis=['Meta'],posthoc_method =['Wilcoxon'], var_thresh = 0.05,
                 task_type  ='C', scoring ='f1',refinement = False, save_data = False, target_thresh=7):
       
        self.data_path = data_path
        self.activity_col = activity_col
        self.kind_analysis = kind_analysis
        self.task_type = task_type
        self.scoring = scoring
        self.posthoc_method = posthoc_method
        self.refinement = refinement
        self.save_data = save_data # saving meta data
        self.target_thresh = target_thresh
        self.var_thresh = var_thresh
        
        os.chdir(self.data_path)
        self.meta_folder = "Data_posthoc"
        # Check whether the specified path exists or not
        isExist = os.path.exists(self.meta_folder)
        if not isExist:
           # Create a new directory because it does not exist
            os.makedirs(self.meta_folder)
            #print("The new directory is created!")
        self.data_save_dir = f"{self.data_path}/{self.meta_folder}/"
        
        # Data raw tracking
        self.feature_0     = []
        self.dup_train     = []
        self.dup_test      = []
        self.dup_col       = []
        self.variance      = []
        self.Outlier_train = []
        self.Outlier_test  = []
        self.feature       = []
        
        self.training_data = []
        self.testing_data = []
        
        self.index = ['Feature', 'Duplicated data train', 'Duplicated data test','Duplicated columns','Variance', 
                      'Outlier_train', "Outlier_test", 'Feature_training', 'Training data','Testing data']
        
        
    def Data_pipeline(self, data, data_name):
        # 1. Data integration
        integration = Data_Integration(data=data, activity_col=self.activity_col, 
                                       task_type =self.task_type, target_thresh=self.target_thresh)
        integration.fit()
        Data_train = integration.data_train
        Data_test = integration.data_test
        self.feature_0.append(Data_train.shape[1]-1)
              

        # 2. Data preprocess
        preprocess = Data_preprocess(Data_train, Data_test, var_thresh = self.var_thresh,
                                     activity_col =self.activity_col)
        preprocess.fit()
        self.variance.append(preprocess.var)
        self.dup_train.append(preprocess.train_dup)
        self.dup_test.append(preprocess.test_dup) 
        self.dup_col.append(preprocess.dup_col) 
        
        Data_train = preprocess.data_train
        Data_test = preprocess.data_test

        # 3. Univariate outliers
        univariate = Univariate_Outliers(Data_train, Data_test, 
                                   activity_col=self.activity_col, handling_method = 'Transformation',
                                   Kbin_handling ='Y', variance_threshold = 'Y')
        univariate.fit()
        Data_train = univariate.data_train
        Data_test = univariate.data_test

        # 4. Multivariate outliers
        mutivariate = Mutivariate_Outliers(Data_train,Data_test, 
                                           method = 'LocalOutlierFactor')
        mutivariate.fit()
        Data_train = mutivariate.data_train
        Data_test = mutivariate.data_test
        
        self.Outlier_train.append(-mutivariate.data_train.shape[0]+univariate.data_train.shape[0])
        self.Outlier_test.append(-mutivariate.data_test.shape[0]+univariate.data_test.shape[0])

        # 5. Rescale

        rescaling = rescale(Data_train,Data_test, 
                            activity_col=self.activity_col, scaler_method = 'MinMaxScaler')
        rescaling.fit()
        Data_train = rescaling.data_train
        Data_test = rescaling.data_test
        
        # 6. Feature engineering
        feature = feature_selection_pipeline(data_train=Data_train, data_test=Data_test, 
                                             activity_col=self.activity_col,task_type =self.task_type,
                                             scoring = self.scoring, method ='RandomForest')
        feature.fit()
        X_train = feature.X_train_new
        X_test  = feature.X_test_new
        y_train = feature.y_train
        y_test  = feature.y_test
        
        self.feature.append(X_train.shape[1])
        self.training_data.append(X_train.shape[0])
        self.testing_data.append(X_test.shape[0])

        # 7. Model building
        model = model_selection(X_train, y_train, SAVE_PREFIX=self.data_save_dir, 
                           data_name=data_name, task_type =self.task_type,scoring = self.scoring)
        model.compare()
        
    def fit(self):
        os.chdir(self.data_path)
        self.data_name = []
        for i in sorted(glob.glob(self.data_path+"/*.csv")):
            self.data_name.append(i[len(self.data_path)+1:-4])
        for i in self.data_name:
            data_path = str(i)+'.csv'
            data = pd.read_csv(data_path)
            self.Data_pipeline(data=data, data_name = i)
            
        for i in self.posthoc_method:
            for j in self.kind_analysis:
                static = statistic_data(data_dir=self.data_save_dir, save_data=self.save_data, refinement =self.refinement,  
                                    scoring = self.scoring, posthoc_method =i, 
                                    kind_analysis=j)
                static.fit()
            

       
        
        self.df_process = pd.DataFrame([self.feature_0, self.dup_train, self.dup_test, self.dup_col,
                                        self.variance, self.Outlier_train, self.Outlier_test,
                                       self.feature, self.training_data, self.testing_data], index = self.index).T
        self.df_process.index = self.data_name
        self.df_process.to_csv(f"{self.data_save_dir}Processing.csv")


class QSAR_pipeline_scaffold:
    """
    - QSAR data pipeline includes:
        - Data integration
        - Data preprocessing
        - Outliers handling
        - Rescaling
        - Feature selection
        - Model building
        - Statistic analysis

    Input:
    ------
    data_path: path
        Directory contains all data types to run QSAR pipeline
    acitivity_col: str
        Name of acitivity columns (pIC50, pChEMBL Value)
    var_thresh: float (default = 0.05)
         Variance threshold to remove features
    save_data: bool (default = False)
        True if want to save the posthoc analysis data
    task_type: str ('C' or 'R')
        Classification (C) or Regression (R)
    target_thresh: int
        Threshold to transform numerical columns to binary
    scoring: str
        if task_type = (C): 'f1', 'average_precision', 'recall'
        if task_type = (R): 'r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'
    refinement: bool (default = False)
        True to remove results equal to zero in boxplot visualization
    
    posthoc_method: list of str (['Wilcoxon'], ['Mannwhitney'], ['Wilcoxon', 'Mannwhitney']) 
        Method for posthoc analysis
    kind_analysis: list of str (['Meta', 'Subgroup']):
        Kind of posthoc analysis
    
    Returns:
    --------
    Data_posthoc: new directory (data_dir in posthoc test)
        Directory contains data for posthoc analysis
    Meta: new directory
        Directory contains posthoc data and figures
    

    --------
    
    """
    def __init__(self, data_path, activity_col, ID, smile_col, kind_analysis=['Meta'],posthoc_method =['Wilcoxon'], var_thresh = 0.05,
                 task_type  ='C', scoring ='f1',refinement = False, save_data = False, target_thresh=7):
       
        self.data_path = data_path
        self.activity_col = activity_col
        self.kind_analysis = kind_analysis
        self.task_type = task_type
        self.scoring = scoring
        self.posthoc_method = posthoc_method
        self.refinement = refinement
        self.save_data = save_data # saving meta data
        self.target_thresh = target_thresh
        self.var_thresh = var_thresh
        self.ID = ID
        self.smile_col = smile_col
        
        os.chdir(self.data_path)
        self.meta_folder = f"Data_posthoc_{self.scoring}"
        # Check whether the specified path exists or not
        isExist = os.path.exists(self.meta_folder)
        if not isExist:
           # Create a new directory because it does not exist
            os.makedirs(self.meta_folder)
            #print("The new directory is created!")
        self.data_save_dir = f"{self.data_path}/{self.meta_folder}/"
        
        # Data raw tracking
        self.feature_0     = []
        self.dup_train     = []
        self.dup_test      = []
        self.dup_col       = []
        self.variance      = []
        self.Outlier_train = []
        self.Outlier_test  = []
        self.feature       = []
        
        self.training_data = []
        self.testing_data = []
        
        self.index = ['Feature', 'Duplicated data train', 'Duplicated data test','Duplicated columns','Variance', 
                      'Outlier_train', "Outlier_test", 'Feature_training', 'Training data','Testing data']
        
        
    def Data_pipeline_scaffold(self, data, data_name):
        # 1. Data integration
        integration = Data_Integration_Scaffold(data=data, activity_col=self.activity_col, ID = self.ID, smile_col=self.smile_col,
                                       task_type =self.task_type, target_thresh=self.target_thresh, random_seed = 42)
        integration.fit()
        Data_train = integration.data_train
        Data_test = integration.data_test
        self.feature_0.append(Data_train.shape[1]-1)
              

        # 2. Data preprocess
        preprocess = Data_preprocess_Scaffold(Data_train, Data_test, var_thresh = self.var_thresh, ID = self.ID, smile_col = self.smile_col,
                                     activity_col =self.activity_col)
        preprocess.fit()
        self.variance.append(preprocess.var)
        self.dup_train.append(preprocess.train_dup)
        self.dup_test.append(preprocess.test_dup) 
        self.dup_col.append(preprocess.dup_col) 
        Data_train = preprocess.data_train_final
        Data_test = preprocess.data_test_final
 

        # 3. Univariate outliers
        univariate = Univariate_Outliers_Scaffold(Data_train, Data_test, 
                                   activity_col=self.activity_col, ID = self.ID, smile_col= self.smile_col,
                                   handling_method = 'Transformation',
                                   Kbin_handling ='Y', variance_threshold = 'Y')
        
        univariate.fit()
        Data_train = univariate.data_train_final
        Data_test = univariate.data_test_final
        #Data_train = univariate.data_train
        #Data_test = univariate.data_test

        # 4. Multivariate outliers
        mutivariate = Mutivariate_Outliers_Scaffold(Data_train,Data_test, ID = self.ID, smile_col=self.smile_col,
                                           method = 'LocalOutlierFactor')
        mutivariate.fit()
        Data_train = mutivariate.Data_train_LOF
        Data_test = mutivariate.Data_train_LOF
        #display(Data_train.head(5))
        self.Outlier_train.append(-mutivariate.data_train.shape[0]+univariate.data_train.shape[0])
        self.Outlier_test.append(-mutivariate.data_test.shape[0]+univariate.data_test.shape[0])

        # 5. Rescale

        rescaling = rescale_scaffold(data_train = Data_train, data_test = Data_test, 
                            activity_col=self.activity_col, ID = self.ID, smile_col = self.smile_col, scaler_method = 'MinMaxScaler')
        rescaling.fit()
        Data_train = rescaling.data_train_final
        Data_test = rescaling.data_test_final
        print("feature selection")
        #display(Data_train.head(5))
        # 6. Feature engineering
        feature = feature_selection_pipeline_scaffold(data_train=Data_train, data_test=Data_test, 
                                             activity_col=self.activity_col, ID = self.ID, smile_col = self.smile_col,
                                             task_type =self.task_type,
                                             scoring = self.scoring, method ='RandomForest')
        feature.fit()
        X_train = feature.X_train_new
        X_test  = feature.X_test_new
        y_train = feature.y_train
        y_test  = feature.y_test
        df_X_train = pd.DataFrame(X_train)
        y_train_df = pd.DataFrame(y_train).reset_index(drop = True)
        smile_id_train = Data_train[[self.ID, self.smile_col]].reset_index(drop = True)
        Data_train_selected = pd.concat([smile_id_train, y_train_df, df_X_train], axis = 1)
        df_X_test = pd.DataFrame(X_test)
        y_test_df = pd.DataFrame(y_test).reset_index(drop = True)
        smile_id_test = Data_test[[self.ID, self.smile_col]].reset_index(drop = True)
        Data_test_selected = pd.concat([smile_id_test, y_test_df,df_X_test], axis = 1)
        
        self.feature.append(X_train.shape[1])
        self.training_data.append(X_train.shape[0])
        self.testing_data.append(X_test.shape[0])

        # # 7. Model building
        model = model_selection_scaffold(data=Data_train_selected,activity_col=self.activity_col, ID = self.ID, 
                               smiles_col  = self.smile_col,
                                SAVE_PREFIX=self.data_save_dir, 
                           data_name=data_name, task_type =self.task_type,scoring = self.scoring)
        model.compare()
        
    def fit(self):
        os.chdir(self.data_path)
        self.data_name = []
        for i in sorted(glob.glob(self.data_path+"/*.csv")):
            self.data_name.append(i[len(self.data_path)+1:-4])
        for i in tqdm(self.data_name):
            data_path = str(i)+'.csv'
            print(data_path)
            data = pd.read_csv(data_path)
            self.Data_pipeline_scaffold(data=data, data_name = i)
            
        for i in self.posthoc_method:
            for j in self.kind_analysis:
                static = statistic_data(data_dir=self.data_save_dir, save_data=self.save_data, refinement =self.refinement,  
                                    scoring = self.scoring, posthoc_method =i, 
                                    kind_analysis=j)
                static.fit()
            

       
        
        self.df_process = pd.DataFrame([self.feature_0, self.dup_train, self.dup_test, self.dup_col,
                                        self.variance, self.Outlier_train, self.Outlier_test,
                                       self.feature, self.training_data, self.testing_data], index = self.index).T
        self.df_process.index = self.data_name
        self.df_process.to_csv(f"{self.data_save_dir}Processing.csv")
        
