# import numpy as np
# import pandas as pd
# from rdkit import Chem, DataStructs
# import pickle
# from rdkit.Avalon import pyAvalonTools as fpAvalon
# from rdkit.Chem.Pharm2D import Gobbi_Pharm2D,Generate
# # from mordred import Calculator, descriptors
# from tqdm import tqdm # progress bar
# import subprocess
# tqdm.pandas()
# import sys
# sys.path.append("..")

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import PandasTools
import pickle
from rdkit.Avalon import pyAvalonTools as fpAvalon
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D,Generate
from mordred import Calculator, descriptors
from tqdm import tqdm # progress bar
import subprocess
tqdm.pandas()
import sys
sys.path.append("..")

class data_input():
    def __init__(self, data_path, ID, smiles_col):
        self.data_path = data_path
        self.ID_col = ID
        self.SMILES_col = smiles_col
    def read_data(self):
        data = pd.read_csv(self.data_path)
        data = data[[self.ID_col, self.SMILES_col]]
        data["Activity"]=0
        return data
    
class representation_calculation():
    def __init__(self, data, ID_col, SMILES_col, Molecule_col,type_fpt, save_dir, Activity_col = "Activity"):
        # Delete the folder
        self.save_dir = save_dir
        self.ID = ID_col
        self.SMILES = SMILES_col
        self.Mols = Molecule_col
        self.type_fpt = type_fpt
        self.activity = Activity_col
        self.data = data[[self.ID,self.SMILES, self.activity,self.Mols]]
        

    def RDKFp(self, mol, maxPath=5, fpSize=2048, nBitsPerHash=2):
        fp = Chem.RDKFingerprint(mol, maxPath=maxPath, fpSize=fpSize, nBitsPerHash=nBitsPerHash)
        ar = np.zeros((1,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, ar)
        return ar
    
    def Avalon(self, mol):
        fp = fpAvalon.GetAvalonFP(mol, 1024) 
        ar = np.zeros((1,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, ar)
        return ar
    
    def mol2pharm2dgbfp(self,mol):
        fp = Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory) 
        ar = np.frombuffer(fp.ToBitString().encode(), 'u1') - ord('0')
        return ar
    
    def rdk_calculation(self, rdk_type):
        maxPath = rdk_type[0]
        fpSize = rdk_type[1]
        self.RDKF = self.data.copy()
        self.RDKF["FPs"] = self.RDKF.Molecule.progress_apply(self.RDKFp, maxPath=maxPath, fpSize=fpSize)
        X = np.stack(self.RDKF.FPs.values)
        df = pd.DataFrame(X)
        self.RDKF= pd.concat([self.RDKF, df], axis = 1).drop(["FPs", self.Mols], axis =1)
        self.RDKF.to_csv(f"{self.save_dir}RDK{maxPath}.csv", index= False)
        return self.RDKF
    
    def avalon_calculation(self):
        self.avalon= self.data.copy()
        self.avalon["FPs"] = self.avalon.Molecule.apply(self.Avalon)
        X = np.stack(self.avalon.FPs.values)
        df = pd.DataFrame(X)
        self.avalon= pd.concat([self.avalon, df], axis = 1).drop(["FPs", self.Mols], axis =1)
        self.avalon.to_csv(f"{self.save_dir}Avalon.csv", index= False)
        return self.avalon
    
    def ph4_calculation(self):
        self.gobbi = self.data.copy()
        self.gobbi["pharmgb"] = self.gobbi.Molecule.apply(self.mol2pharm2dgbfp)
        X = np.stack(self.gobbi["pharmgb"].values)
        d = pd.DataFrame(X)
        self.gobbi = pd.concat([self.gobbi, d], axis = 1).drop(["pharmgb", self.Mols], axis =1)
        self.gobbi.to_csv(f"{self.save_dir}Ph4_gobbi.csv", index= False)
        return self.gobbi

    def mordred_calculation(self):
        self.mordred = self.data.copy()
        # calc = Calculator(descriptors, ignore_3D=True)
        # df_2d_mordred = calc.pandas(self.mordred.Molecule,quiet=True)
        # # self.mordred_raw = pd.concat([self.mordred[[self.ID, self.SMILES, "Activity"]], df_2d_mordred], axis = 1)
        # Chem.PandasTools.WriteSDF(self.mordred, f'{self.save_dir}conf.sdf', molColName=self.Mols)
        PandasTools.WriteSDF(self.mordred, f'{self.save_dir}conf.sdf', molColName=self.Mols)
        command = f"python -m mordred {f'{self.save_dir}conf.sdf'} -o {f'{self.save_dir}Mordred.csv'}"
        # Execute the command
        subprocess.run(command, shell=True, check=True)


        mordred_df = pd.read_csv(f'{self.save_dir}Mordred.csv')
        mordred_df = mordred_df.drop('name', axis=1)
        self.mordred_raw = pd.concat([self.mordred[[self.ID, self.SMILES, "Activity"]], mordred_df], axis = 1)
        process = preprocess_mordred(self.mordred_raw, activity_col = "Activity", ID = self.ID, smile_col = self.SMILES,
                            material_dir="./Utility/")
        mordred_processed = process.fit()
        mordred_processed.to_csv(f"{self.save_dir}Mordred.csv", index=False)
        return mordred_processed

    def fit(self):
        if self.type_fpt == "rdk5":
            featurized_data = self.rdk_calculation(rdk_type = [5,2048])
        elif self.type_fpt == "rdk6":
            featurized_data = self.rdk_calculation(rdk_type = [6,2048])
        elif self.type_fpt == "rdk7":      
            featurized_data = self.rdk_calculation(rdk_type = [7,4096])
        elif self.type_fpt == "avalon":
            featurized_data = self.avalon_calculation()
        elif self.type_fpt == "ph4_gobbi":
            featurized_data = self.ph4_calculation()
        else:
            featurized_data = self.mordred_calculation()
        return featurized_data
        
       


    

class preprocess_mordred():
    def __init__(self,data, activity_col, ID, smile_col, material_dir):
        self.data_smiles = data[[ID, smile_col]]
        self.data = data.drop([ID, smile_col], axis =1)        
        self.activity_col = activity_col
        self.ID = ID
        self.drop_cols = np.loadtxt(f"{material_dir}Mordred_missing_cols.txt", dtype = 'str')
        self.variance_cols = np.loadtxt(f"{material_dir}Mordred_low_variance_feature.txt", dtype = 'str')
        self.nomial_cols = np.loadtxt(f"{material_dir}Mordred_Nol_col.txt", dtype = 'str')
        with open(f"{material_dir}imp.pkl",'rb') as f:
            self.imp = pickle.load(f)
    
    def check_nan(self,data):
        index = []
        for key, value in enumerate(data):
            if type(value) == float or type(value) == int:
                continue
            else:
                index.append(key)
        data[index] = np.nan
    
    def missing_value(self):
        self.data.drop(self.drop_cols, axis =1, inplace = True)
        null_data_train = self.data[self.data.isnull().any(axis=1)]
        #imputer
        Data=pd.DataFrame(self.imp.transform(self.data))
        Data.columns=self.data.columns
        Data.index=self.data.index
        self.data = Data

    def Nomial(self):
        self.data[self.nomial_cols]=self.data[self.nomial_cols].astype('int64')
    
    def fit(self):
        self.data.apply(self.check_nan)
        self.missing_value()
        self.data = self.data.loc[:, self.variance_cols]
        self.Nomial()
        self.data = pd.concat([self.data_smiles, self.data], axis =1)
        return self.data


    

    


    

    
           
