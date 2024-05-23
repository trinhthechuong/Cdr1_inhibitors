# Classification
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from imblearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_classif, chi2, RFE, RFECV, f_classif,mutual_info_regression,f_regression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, RepeatedStratifiedKFold
from sklearn.svm import LinearSVC, SVC

# Regression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, LassoCV
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, cross_val_score, RepeatedKFold
from sklearn.svm import LinearSVR, SVR

import warnings
warnings.filterwarnings('ignore')

class feature_selection_pipeline:
    """
    - Remove unnecessary features
    - Show a chart that compares the effectiveness of each method.
    - Based on the chart, choose the best method.

    Input:
    ------
    data_train: pandas.DataFrame
        Data for training model after rescaling
    data_test: pandas.DataFrame
        Data for external validation after rescaling
    acitivity_col: str
        Name of acitivity columns (pIC50, pChEMBL Value)
    task_type: str
        Classification (C) or Regression (R)
    scoring: str
        if task_type = (C): 'f1', 'average_precision', 'recall'
        if task_type = (R): 'r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'
    method: str (default = 'RandomForest')
        Feature selection method include: 'Statistic_Anova','Statistic_Mutual','RandomForest',
        'ExtraTree', 'AdaBoost', 'GradBoost', 'XGBoost', 'Linear', 'SVM', 'RFE'
    
    Returns:
    --------
    Completed Data_train and Data_test
    X_train: np.array
    y_train: np.array
    X_test: np.array
    y_test: np.array
    """
    def __init__(self, data_train, data_test, activity_col, task_type ='C',scoring = 'f1', method ='RandomForest'):
        
        self.activity_col = activity_col
        self.task_type = task_type
        self.scoring = scoring
        self.method = method
        
        self.X_train = data_train.drop([self.activity_col], axis = 1)
        self.y_train = data_train[self.activity_col]
        self.X_test = data_test.drop([self.activity_col], axis = 1)
        self.y_test = data_test[self.activity_col]
        # Check task type
        if len(self.y_train.unique())==2:
            self.task_type == 'C'
            self.cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
        else:
            self.task_type == 'R'
            self.cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)
    
    # 1. Statistic
    def Statistic(self):
        if self.method =='Statistic_Anova':
            self.select = SelectKBest(score_func= f_classif, k= 20)
        elif self.method =='Statistic_Mutual':
            self.select = SelectKBest(score_func= mutual_info_classif, k= 20)
        self.select.fit(self.X_train, self.y_train)
    
    # 2. Random forest
    def random_forest(self):
        if self.task_type =='C':
            forest = RandomForestClassifier(random_state=42)
        else:
            forest = RandomForestRegressor(random_state=42)
        forest.fit(self.X_train, self.y_train)
        self.select =  SelectFromModel(forest, prefit=True)
        
    
    # 3. ExtraTree
    def extra_tree(self):
        if self.task_type =='C':
            ext_tree = ExtraTreesClassifier(random_state=42)
        else:
            ext_tree = ExtraTreesRegressor(random_state=42)
        ext_tree.fit(self.X_train, self.y_train)
        self.select =  SelectFromModel(ext_tree, prefit=True)
    
    # 4. Ada
    def ada(self):
        if self.task_type =='C':
            ada = AdaBoostClassifier(random_state=42)
        else:
            ada = AdaBoostRegressor(random_state=42)
        ada.fit(self.X_train, self.y_train)
        self.select =  SelectFromModel(ada, prefit=True)
        
    # 5. Gradient boosting
    def grad(self):
        if self.task_type =='C':
            grad = GradientBoostingClassifier(random_state=42)
        else:
            grad = GradientBoostingRegressor(random_state=42)
        grad.fit(self.X_train, self.y_train)
        self.select =  SelectFromModel(grad, prefit=True)
    
    # 6. XGBoost
    def XGb(self):
        if self.task_type =='C':
            XGb = XGBClassifier(random_state = 42, verbosity=0, use_label_encoder=False, eval_metrics ='logloss')
        else:
            XGb = XGBRegressor(random_state = 42, verbosity=0, use_label_encoder=False, eval_metrics ='logloss')
        XGb.fit(self.X_train, self.y_train)
        self.select =  SelectFromModel(XGb, prefit=True)
        
    # 7. Linear method
    def Linear(self):
        if self.task_type =='C':
            lasso = LogisticRegression(random_state=42, penalty = 'elasticnet', solver = 'saga', l1_ratio = 0.5, max_iter = 10000)
        else:
            lasso = LassoCV(random_state = 42)
        lasso.fit(self.X_train, self.y_train)
        self.select =  SelectFromModel(lasso, prefit=True)
        
    # 8. Support vector machine
    def SVM(self):
        if self.task_type =='C':
            svm = LinearSVC(random_state=42, penalty = 'l1', dual = False, max_iter = 10000)
        else:
            svm = LinearSVR(random_state=42, dual = False, max_iter = 10000) 
        svm.fit(self.X_train, self.y_train)
        self.select =  SelectFromModel(svm, prefit=True)
        
    
    # 9. RFECV with SVC
    def RFE_SVM(self):
        n = int(input("Please input number of features?"))  
        if self.task_type =='C':
            self.select = RFE(estimator=SVC(kernel="linear"), n_features_to_select=n)
        else:
            self.select = RFE(estimator=SVR(kernel="linear"), n_features_to_select=n) 
        self.select.fit(self.X_train, self.y_train)
        self.X_train_new, self.y_train_new = self.select.transform(self.X_train, self.y_train)
        self.X_test_new, self.y_test_new = self.select.transform(self.X_test, self.y_test)
        
    def fit(self):
        if self.method == 'Statistic_Anova':
            self.Statistic()
        elif self.method == 'Statistic_Mutual':
            self.Statistic()
        elif self.method =='RandomForest':
            self.random_forest()
        elif self.method =='ExtraTree':
            self.extra_tree()
        elif self.method =='AdaBoost':
            self.ada()
        elif self.method =='GradBoost':
            self.grad()
        elif self.method =='XGBoost':
            self.XGb()
        elif self.method =='Linear':
            self.Linear()
        elif self.method =='SVM':
            self.SVM()
        elif self.method == 'RFE':
            self.RFE_SVM()
        
        print(self.select)
        if self.method !='RFE':
            self.X_train_new = self.select.transform(self.X_train)
            self.X_test_new = self.select.transform(self.X_test)
        
       
class feature_selection_pipeline_scaffold:
    """
    - Remove unnecessary features
    - Show a chart that compares the effectiveness of each method.
    - Based on the chart, choose the best method.

    Input:
    ------
    data_train: pandas.DataFrame
        Data for training model after rescaling
    data_test: pandas.DataFrame
        Data for external validation after rescaling
    acitivity_col: str
        Name of acitivity columns (pIC50, pChEMBL Value)
    task_type: str
        Classification (C) or Regression (R)
    scoring: str
        if task_type = (C): 'f1', 'average_precision', 'recall'
        if task_type = (R): 'r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'
    method: str (default = 'RandomForest')
        Feature selection method include: 'Statistic_Anova','Statistic_Mutual','RandomForest',
        'ExtraTree', 'AdaBoost', 'GradBoost', 'XGBoost', 'Linear', 'SVM', 'RFE'
    
    Returns:
    --------
    Completed Data_train and Data_test
    X_train: np.array
    y_train: np.array
    X_test: np.array
    y_test: np.array
    """
    def __init__(self, data_train, data_test, activity_col,ID, smile_col, task_type ='C',scoring = 'f1', method ='RandomForest'):
        self.ID = ID
        self.smile_col = smile_col
        
        self.activity_col = activity_col
        self.task_type = task_type
        self.scoring = scoring
        self.method = method
        
        self.X_train = data_train.drop([self.activity_col,self.ID, self.smile_col], axis = 1)

        self.y_train = data_train[self.activity_col]
        self.X_test = data_test.drop([self.activity_col,self.ID, self.smile_col], axis = 1)
        self.y_test = data_test[self.activity_col]
        # Check task type
        if len(self.y_train.unique())==2:
            self.task_type == 'C'
            self.cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
        else:
            self.task_type == 'R'
            self.cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)
    
    # 1. Statistic
    def Statistic(self):
        if self.method =='Statistic_Anova':
            self.select = SelectKBest(score_func= f_classif, k= 20)
        elif self.method =='Statistic_Mutual':
            self.select = SelectKBest(score_func= mutual_info_classif, k= 20)
        self.select.fit(self.X_train, self.y_train)
    
    # 2. Random forest
    def random_forest(self):
        if self.task_type =='C':
            forest = RandomForestClassifier(random_state=42)
        else:
            forest = RandomForestRegressor(random_state=42)
        forest.fit(self.X_train, self.y_train)
        self.select =  SelectFromModel(forest, prefit=True)
        
    
    # 3. ExtraTree
    def extra_tree(self):
        if self.task_type =='C':
            ext_tree = ExtraTreesClassifier(random_state=42)
        else:
            ext_tree = ExtraTreesRegressor(random_state=42)
        ext_tree.fit(self.X_train, self.y_train)
        self.select =  SelectFromModel(ext_tree, prefit=True)
    
    # 4. Ada
    def ada(self):
        if self.task_type =='C':
            ada = AdaBoostClassifier(random_state=42)
        else:
            ada = AdaBoostRegressor(random_state=42)
        ada.fit(self.X_train, self.y_train)
        self.select =  SelectFromModel(ada, prefit=True)
        
    # 5. Gradient boosting
    def grad(self):
        if self.task_type =='C':
            grad = GradientBoostingClassifier(random_state=42)
        else:
            grad = GradientBoostingRegressor(random_state=42)
        grad.fit(self.X_train, self.y_train)
        self.select =  SelectFromModel(grad, prefit=True)
    
    # 6. XGBoost
    def XGb(self):
        if self.task_type =='C':
            XGb = XGBClassifier(random_state = 42, verbosity=0, use_label_encoder=False, eval_metrics ='logloss')
        else:
            XGb = XGBRegressor(random_state = 42, verbosity=0, use_label_encoder=False, eval_metrics ='logloss')
        XGb.fit(self.X_train, self.y_train)
        self.select =  SelectFromModel(XGb, prefit=True)
        
    # 7. Linear method
    def Linear(self):
        if self.task_type =='C':
            lasso = LogisticRegression(random_state=42, penalty = 'elasticnet', solver = 'saga', l1_ratio = 0.5, max_iter = 10000)
        else:
            lasso = LassoCV(random_state = 42)
        lasso.fit(self.X_train, self.y_train)
        self.select =  SelectFromModel(lasso, prefit=True)
        
    # 8. Support vector machine
    def SVM(self):
        if self.task_type =='C':
            svm = LinearSVC(random_state=42, penalty = 'l1', dual = False, max_iter = 10000)
        else:
            svm = LinearSVR(random_state=42, dual = False, max_iter = 10000) 
        svm.fit(self.X_train, self.y_train)
        self.select =  SelectFromModel(svm, prefit=True)
        
    
    # 9. RFECV with SVC
    def RFE_SVM(self):
        n = int(input("Please input number of features?"))  
        if self.task_type =='C':
            self.select = RFE(estimator=SVC(kernel="linear"), n_features_to_select=n)
        else:
            self.select = RFE(estimator=SVR(kernel="linear"), n_features_to_select=n) 
        self.select.fit(self.X_train, self.y_train)
        self.X_train_new, self.y_train_new = self.select.transform(self.X_train, self.y_train)
        self.X_test_new, self.y_test_new = self.select.transform(self.X_test, self.y_test)
        
    def fit(self):
        if self.method == 'Statistic_Anova':
            self.Statistic()
        elif self.method == 'Statistic_Mutual':
            self.Statistic()
        elif self.method =='RandomForest':
            self.random_forest()
        elif self.method =='ExtraTree':
            self.extra_tree()
        elif self.method =='AdaBoost':
            self.ada()
        elif self.method =='GradBoost':
            self.grad()
        elif self.method =='XGBoost':
            self.XGb()
        elif self.method =='Linear':
            self.Linear()
        elif self.method =='SVM':
            self.SVM()
        elif self.method == 'RFE':
            self.RFE_SVM()
        
        print(self.select)
        if self.method !='RFE':
            self.X_train_new = self.select.transform(self.X_train)
            self.X_test_new = self.select.transform(self.X_test)
        
