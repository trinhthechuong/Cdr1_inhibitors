import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import os
from sklearn.model_selection import train_test_split

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

def scaffold_split(data, smiles_col, test_size = 0.2,random_state = 42):
    """
    Split a molecule dataset into training and test sets based on scaffolds.

    Parameters:
    - data (pd.DataFrame): The dataset containing molecule data.
    - smiles_col (str): The name of the column containing SMILES strings.
    - test_size (float): Proportion of the dataset to include in the test split.
    - random_state (int): Random state for reproducibility.

    Returns:
    - data_train (pd.DataFrame): Training set.
    - data_test (pd.DataFrame): Test set.
    """
    scaffolds = {}
    for idx, row in data.iterrows():
        smiles = row[smiles_col]
        mol = Chem.MolFromSmiles(smiles)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [idx]
        else:
            scaffolds[scaffold].append(idx)

    scaffold_lists = list(scaffolds.values())
    np.random.seed(random_state)
    np.random.shuffle(scaffold_lists)

    num_molecules = len(data)
    num_test = int(np.floor(test_size * num_molecules))
    train_idx, test_idx = [], []
    for scaffold_list in scaffold_lists:
        if len(test_idx) + len(scaffold_list) <= num_test:
            test_idx.extend(scaffold_list)
        else:
            train_idx.extend(scaffold_list)

    data_train = data.iloc[train_idx]
    data_test = data.iloc[test_idx]

    return data_train, data_test


class Data_Integration():
    """
    Create Data Frame from csv file, find missing value (NaN), choose a threshold to make target transformation (Classification)
    remove handcrafted value (Regression), split Data to Train and Test and show the chart
    
    Input:
    -----
    data : pandas.DataFrame
        Data with features and target columns
    acitivity_col: str
        Name of acitivity columns (pIC50, pChEMBL Value)
    task_type: str ('C' or 'R')
        Classification (C) or Regression (R)
    target_thresh: int
        Threshold to transform numerical columns to binary
        
    Returns:
    --------
    Data_train: pandas.DataFrame
        Data for training model
    Data_test: pandas.DataFrame
        Data for external validation  
    """
    def __init__(self, data, activity_col, task_type, target_thresh):
        
        self.data = data
        self.activity_col= activity_col
        self.task_type = task_type
        self.target_thresh = target_thresh
        
        
    # 1. Check nan value - Mark Nan value to np.nan
    def Check_NaN(self, data):
        index = []
        for key, value in enumerate(data):
            if type(value) == float or type(value) == int:
                continue
            else:
                index.append(key)
        data[index] = np.nan 
    
    # 2. Target transformation - Classification
    def target_bin(self, thresh, input_target_style = 'pIC50'):
        if input_target_style != 'pIC50':
            self.thresh = thresh
            t1 = self.data[self.activity_col] < self.thresh 
            self.data.loc[t1, self.activity_col] = 1
            t2 = self.data[self.activity_col] >= self.thresh 
            self.data.loc[t2, self.activity_col] = 0
            self.data[self.activity_col] = self.data[self.activity_col].astype('int64')
        else:
            self.thresh = thresh
            t1 = self.data[self.activity_col] < self.thresh 
            self.data.loc[t1, self.activity_col] = 0
            t2 = self.data[self.activity_col] >= self.thresh 
            self.data.loc[t2, self.activity_col] = 1
            self.data[self.activity_col] = self.data[self.activity_col].astype('int64')
        
    
    # 3. Split data
    #Chia tập dữ liệu  thành tập train và test.
    def Data_split(self):
        
        if self.task_type.title() == "C":
            if len(self.data[self.activity_col].unique()) ==2: 
                y = self.data[self.activity_col]
            else:
                self.target_bin(thresh = self.target_thresh)
                y = self.data[self.activity_col]
            
            self.stratify = y
        
        elif self.task_type.title() == "R":
            y = self.data[self.activity_col]
            self.stratify = None
            
        
       
        X = self.data.drop([self.activity_col], axis =1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, 
                                                            random_state = 42, stratify = self.stratify)


        #index:
        self.idx = X.T.index

        #Train:
        self.df_X_train = pd.DataFrame(X_train, columns = self.idx)
        self.df_y_train = pd.DataFrame(y_train, columns = [self.activity_col])
        self.data_train = pd.concat([self.df_y_train, self.df_X_train], axis = 1)
        

        #test
        self.df_X_test = pd.DataFrame(X_test, columns = self.idx)
        self.df_y_test = pd.DataFrame(y_test, columns = [self.activity_col])
        self.data_test = pd.concat([self.df_y_test, self.df_X_test], axis = 1)
        
        print("Data train:", self.data_train.shape)
        print("Data test:", self.data_test.shape)
        print(75*"*")
        

    def Visualize_target(self):
        if self.task_type.title() == "C":
            sns.set('notebook')
            plt.figure(figsize = (16,5))
            plt.subplot(1,2,1)
            plt.title(f'Training data', weight = 'semibold', fontsize = 16)
            plt.hist(self.data_train.iloc[:,0])
            plt.xlabel(f'Imbalance ratio: {(round((self.data_train.iloc[:,0].values == 1).sum() / (self.data_train.iloc[:,0].values == 0).sum(),3))}')
            plt.subplot(1,2,2)
            plt.title(f'External data', weight = 'semibold', fontsize = 16)
            plt.hist(self.data_test.iloc[:,0])
            plt.xlabel(f'Imbalance ratio: {(round((self.data_test.iloc[:,0].values == 1).sum() / (self.data_test.iloc[:,0].values == 0).sum(),3))}')
            #plt.savefig(self.SAVE_PREFIX+"distribution.png", dpi = 600)
            plt.show()
        else:
            plt.figure(figsize = (16,5))
            plt.subplot(1,2,1)
            plt.hist(self.data_train.iloc[:,0])
            plt.title(f'Train distribution', weight = 'semibold', fontsize = 16)
            plt.subplot(1,2,2)
            
            plt.hist(self.data_test.iloc[:,0])
            plt.title(f'Test distribution',weight = 'semibold', fontsize = 16)
            #plt.savefig(self.SAVE_PREFIX+"distribution.png", dpi = 600)
            plt.show()
         
    def fit(self):
        self.data.apply(self.Check_NaN)
        self.Data_split()
        self.Visualize_target()



class Data_Integration_Scaffold():
    """
    Create Data Frame from csv file, find missing value (NaN), choose a threshold to make target transformation (Classification)
    remove handcrafted value (Regression), split Data to Train and Test and show the chart
    
    Input:
    -----
    data : pandas.DataFrame
        Data with features and target columns
    acitivity_col: str
        Name of acitivity columns (pIC50, pChEMBL Value)
    task_type: str ('C' or 'R')
        Classification (C) or Regression (R)
    target_thresh: int
        Threshold to transform numerical columns to binary
        
    Returns:
    --------
    Data_train: pandas.DataFrame
        Data for training model
    Data_test: pandas.DataFrame
        Data for external validation  
    """
    def __init__(self, data, activity_col,smile_col,ID, task_type, target_thresh,random_seed):
        
        self.data = data
        self.activity_col= activity_col
        self.task_type = task_type
        self.target_thresh = target_thresh
        self.smile_col = smile_col
        self.ID = ID
        self.seed = random_seed
        
    # 1. Check nan value - Mark Nan value to np.nan
    def Check_NaN(self, data):
        index = []
        for key, value in enumerate(data):
            if type(value) == float or type(value) == int:
                continue
            else:
                index.append(key)
        data[index] = np.nan 
    
    # 2. Target transformation - Classification
    def target_bin(self, thresh, input_target_style = 'pIC50'):
        if input_target_style != 'pIC50':
            self.thresh = thresh
            t1 = self.data[self.activity_col] < self.thresh 
            self.data.loc[t1, self.activity_col] = 1
            t2 = self.data[self.activity_col] >= self.thresh 
            self.data.loc[t2, self.activity_col] = 0
            self.data[self.activity_col] = self.data[self.activity_col].astype('int64')
        else:
            self.thresh = thresh
            t1 = self.data[self.activity_col] < self.thresh 
            self.data.loc[t1, self.activity_col] = 0
            t2 = self.data[self.activity_col] >= self.thresh 
            self.data.loc[t2, self.activity_col] = 1
            self.data[self.activity_col] = self.data[self.activity_col].astype('int64')
        
    
    # 3. Split data
    #Chia tập dữ liệu  thành tập train và test.
    def Data_split(self):
        
        if self.task_type.title() == "C":
            if len(self.data[self.activity_col].unique()) ==2: 
                y = self.data[self.activity_col]
            else:
                self.target_bin(thresh = self.target_thresh)
                y = self.data[self.activity_col]
            
            self.stratify = y
        
        elif self.task_type.title() == "R":
            y = self.data[self.activity_col]
            self.stratify = None


        self.data_train, self.data_test = scaffold_split(data = self.data, smiles_col = self.smile_col, test_size = 0.2,random_state = self.seed)
        #index:

        
        print("Data train:", self.data_train.shape)
        print("Data test:", self.data_test.shape)
        print(75*"*")
        

    def Visualize_target(self):
        if self.task_type.title() == "C":
            sns.set('notebook')
            plt.figure(figsize = (16,5))
            plt.subplot(1,2,1)
            plt.title(f'Training data', weight = 'semibold', fontsize = 16)
            plt.hist(self.data_train[self.activity_col])
            plt.xlabel(f'Imbalance ratio: {(round((self.data_train[self.activity_col].values == 1).sum() / (self.data_train[self.activity_col].values == 0).sum(),3))}')
            plt.subplot(1,2,2)
            plt.title(f'External data', weight = 'semibold', fontsize = 16)
            plt.hist(self.data_test[self.activity_col])
            plt.xlabel(f'Imbalance ratio: {(round((self.data_test[self.activity_col].values == 1).sum() / (self.data_test[self.activity_col].values == 0).sum(),3))}')
            #plt.savefig(self.SAVE_PREFIX+"distribution.png", dpi = 600)
            plt.show()
        else:
            plt.figure(figsize = (16,5))
            plt.subplot(1,2,1)
            plt.hist(self.data_train[self.activity_col])
            plt.title(f'Train distribution', weight = 'semibold', fontsize = 16)
            plt.subplot(1,2,2)
            plt.hist(self.data_test[self.activity_col])
            plt.title(f'Test distribution',weight = 'semibold', fontsize = 16)
            #plt.savefig(self.SAVE_PREFIX+"distribution.png", dpi = 600)
            plt.show()
         
    def fit(self):
        original_df = self.data.copy()
        smiles = original_df[self.smile_col].values
        id = original_df[self.ID].values
        #df_smiles = original_df[[self.ID, self.smile_col]]
        self.data.apply(self.Check_NaN)
        self.data[self.ID]= id
        self.data[self.smile_col]= smiles
        self.Data_split()
        self.Visualize_target()