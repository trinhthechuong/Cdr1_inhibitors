import math
import numpy as np
import pandas as pd
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import seaborn as sns

from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.rdBase import BlockLogs
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Draw
from rdkit.Chem import PandasTools, AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors


from rdkit.Chem import MACCSkeys
from rdkit.Chem.AtomPairs import Pairs, Torsions
from rdkit.Avalon import pyAvalonTools as fpAvalon
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D,Generate
from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
from mol2vec.helpers import depict_identifier, plot_2D_vectors, IdentifierTable, mol_to_svg
from gensim.models import word2vec
from mol2vec import features
from mol2vec import helpers
from map4 import MAP4Calculator
from mhfp.encoder import MHFPEncoder
#from mordred import Calculator, descriptors
from tqdm import tqdm # progress bar
tqdm.pandas()
from cats2d.rd_cats2d import CATS2D

from .standardize import standardization
from .Pubchem import calcPubChemFingerAll



class Featurize():
    """
    Standardize molecules and calculate molecular descriptors and fingerprints
    
    Parameters
    ------
    data : pandas.DataFrame
        Data with ID, smiles and activity columns.
    acitivity_col: str
        Name of acitivity columns (pIC50, pChEMBL Value)
    smile_col: str
        Name of smile columns (smiles, SMILES, Canimical smiles,...)
    ID: str
        Name of identity columns
        
    save_dir: path
        Directory to save data after calculating
    Returns
    --------
    Data_Fp: pandas.DataFrame
        Data after calculating descriptors.
    """
    def __init__(self, data, smile_col, activity_col, ID, save_dir, m2v_path, standardize = True):
        self.smile_col = smile_col
        self.activity_col = activity_col
        self.ID = ID
        self.save_dir = save_dir
        self.standardize = standardize
        self.data = data[[self.ID,self.smile_col, self.activity_col]]
        self.m2v_path = m2v_path
        #PandasTools.AddMoleculeColumnToFrame(self.data,self.smile_col,'Molecule')
    
    # standardize molecules
   
    
    
    def RDKFp(self, mol, maxPath=5, fpSize=2048, nBitsPerHash=2):
        fp = Chem.RDKFingerprint(mol, maxPath=maxPath, fpSize=fpSize, nBitsPerHash=nBitsPerHash)
        ar = np.zeros((1,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, ar)
        return ar
    
    
    def ECFPs(self, mol, radius=1, nBits=2048, useFeatures=False):
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits, useFeatures=useFeatures) 
        ar = np.zeros((1,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, ar)
        return ar
    
    def MACCs(self, mol):
        fp = MACCSkeys.GenMACCSKeys(mol) 
        ar = np.zeros((1,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, ar)
        return ar
    
    def Avalon(self, mol):
        fp = fpAvalon.GetAvalonFP(mol, 1024) 
        ar = np.zeros((1,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, ar)
        return ar
    
    
    def RDKDes(self, mol):
        des_list = [x[0] for x in Descriptors._descList]
        calculator = MoleculeDescriptors.MolecularDescriptorCalculator(des_list)
        ar = calculator.CalcDescriptors(mol)
        return ar
    
    
    def mol2pharm2dgbfp(self,mol):
        fp = Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory) 
        ar = np.frombuffer(fp.ToBitString().encode(), 'u1') - ord('0')
        return ar
    
    def mol2cats(self, mol):
        cats = CATS2D(max_path_len=9)
        fp = cats.getCATs2D(mol)
        ar = np.array(fp)
        return ar
    
    def mol2ap(self, mol):
        fp = Pairs.GetAtomPairFingerprint(mol) # rdkit.DataStructs => convert to np
        ar = np.zeros((1,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, ar)
        return ar
    
    def mol2torsion(self, mol):
        fp = Torsions.GetTopologicalTorsionFingerprintAsIntVect(mol) 
        ar = np.zeros((1,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, ar)
        return ar

    def Apply(self):
        
        if self.standardize == True:
            print("STANDARDIZING MOLECULES...")
            
           
           
            block = BlockLogs()
            std = standardization(data=self.data,ID=self.ID, smiles_col=self.smile_col, active_col=self.activity_col, ro5 =4)
            self.data = std.filter_data()
            del block
            
        else: 
            print("COVERTING SMILES TO MOLECULES...")
            self.data['Molecule'] = self.data[self.smile_col].progress_apply(Chem.MolFromSmiles)

       
        # 1-3. RDKit Fingerprint
        rdk_type = [[5,2048],[6,2048],[7,4096]]
        for i in rdk_type:
            maxPath = i[0]
            fpSize = i[1]
            print(F"CALCULATING RDK{maxPath} FINGERPRINTS...")
            self.RDKF = self.data.copy()
            self.RDKF["FPs"] = self.RDKF.Molecule.progress_apply(self.RDKFp, maxPath=maxPath, fpSize=fpSize)
            X = np.stack(self.RDKF.FPs.values)
            df = pd.DataFrame(X)
            #self.RDKF= pd.concat([self.RDKF, df], axis = 1).drop(["FPs", "Molecule", self.ID, self.smile_col], axis =1)
            self.RDKF= pd.concat([self.RDKF, df], axis = 1).drop(["FPs", "Molecule"], axis =1)
            self.RDKF.to_csv(f"{self.save_dir}RDK{maxPath}.csv", index= False)
            
            
        # 4-6. ECFP
        ecfp_type = [[1,2048],[2,2048],[3,4096]]
        for i in ecfp_type:
            radius = i[0]
            nBits = i[1]
            d = 2*radius
            print(F"CALCULATING ECFP{d} FINGERPRINTS...")
        
            self.ecfp = self.data.copy()
            self.ecfp["FPs"] = self.ecfp.Molecule.progress_apply(self.ECFPs, radius = radius, nBits= nBits)
            X = np.stack(self.ecfp.FPs.values)
            df = pd.DataFrame(X)
            #self.ecfp= pd.concat([self.ecfp, df], axis = 1).drop(["FPs", "Molecule", self.ID, self.smile_col], axis =1)
            self.ecfp= pd.concat([self.ecfp, df], axis = 1).drop(["FPs", "Molecule"], axis =1)
            self.ecfp.to_csv(f"{self.save_dir}ECFP{d}.csv", index= False)
        
        # 7-9. FCFP
        #fcfp_type = [[1,2048],[2,2048],[3,4096]]
        #for i in fcfp_type:
        #    radius = i[0]
        #    nBits = i[1]
        #    d = 2*radius
        #    print(F"CALCULATING FCFP{d} FINGERPRINTS...")
        
        #    self.fcfp = self.data.copy()
        #    self.fcfp["FPs"] = self.fcfp.Molecule.progress_apply(self.ECFPs, radius = radius, nBits= nBits, useFeatures=True)
        #    X = np.stack(self.fcfp.FPs.values)
        #    df = pd.DataFrame(X)
        #    self.fcfp= pd.concat([self.fcfp, df], axis = 1).drop(["FPs", "Molecule", self.ID, self.smile_col], axis =1)
        #    self.fcfp.to_csv(f"{self.save_dir}FCFP{d}.csv", index= False)
        
        # 10. MACCS
        self.maccs = self.data.copy()
        print("CALCULATING MACCs FINGERPRINTS...")
        self.maccs["FPs"] = self.maccs.Molecule.progress_apply(self.MACCs)
        X = np.stack(self.maccs.FPs.values)
        df = pd.DataFrame(X)
        #self.maccs= pd.concat([self.maccs, df], axis = 1).drop(["FPs", "Molecule", self.ID, self.smile_col], axis =1)
        self.maccs= pd.concat([self.maccs, df], axis = 1).drop(["FPs", "Molecule"], axis =1)
        self.maccs.to_csv(f"{self.save_dir}MACCs.csv", index= False)
        
        # 11. Avalon
        self.avalon= self.data.copy()
        print("CALCULATING AVALON FINGERPRINTS...")
        self.avalon["FPs"] = self.avalon.Molecule.progress_apply(self.Avalon)
        X = np.stack(self.avalon.FPs.values)
        df = pd.DataFrame(X)
        #self.avalon= pd.concat([self.avalon, df], axis = 1).drop(["FPs", "Molecule", self.ID, self.smile_col], axis =1)
        self.avalon= pd.concat([self.avalon, df], axis = 1).drop(["FPs", "Molecule"], axis =1)
        self.avalon.to_csv(f"{self.save_dir}Avalon.csv", index= False)
        
        
        # 12. pubchem
        self.pubchem= self.data.copy()
        print("CALCULATING PUBCHEM FINGERPRINTS...")
        self.pubchem["FPs"] = self.pubchem.Molecule.progress_apply(calcPubChemFingerAll)
        X = np.stack(self.pubchem.FPs.values)
        df = pd.DataFrame(X)
        #self.pubchem= pd.concat([self.pubchem, df], axis = 1).drop(["FPs", "Molecule", self.ID, self.smile_col], axis =1)
        self.pubchem= pd.concat([self.pubchem, df], axis = 1).drop(["FPs", "Molecule"], axis =1)
        self.pubchem.to_csv(f"{self.save_dir}Pubchem.csv", index= False)
        
        # 13. map4
        self.map4= self.data.copy()
        print("CALCULATING MAP4 FINGERPRINTS...")
        m4_calc = MAP4Calculator(is_folded=True)
        self.map4['map4'] =  self.map4.Molecule.progress_apply(m4_calc.calculate)
        X = np.stack(self.map4['map4'].values)
        d = pd.DataFrame(X)
        #self.map4= pd.concat([self.map4, d], axis = 1).drop(["map4", "Molecule", self.ID, self.smile_col], axis =1)
        self.map4= pd.concat([self.map4, d], axis = 1).drop(["map4", "Molecule"], axis =1)
        self.map4.to_csv(f"{self.save_dir}Map4.csv", index= False)
        
        # 14. secfp
        self.secfp= self.data.copy()
        print("CALCULATING SECFP FINGERPRINTS...")
        self.secfp["secfp"] = self.secfp[self.smile_col].progress_apply(MHFPEncoder.secfp_from_smiles)
        X = np.stack(self.secfp["secfp"].values)
        d = pd.DataFrame(X)
        #self.secfp= pd.concat([self.secfp, d], axis = 1).drop(["secfp", "Molecule", self.ID, self.smile_col], axis =1)
        self.secfp= pd.concat([self.secfp, d], axis = 1).drop(["secfp", "Molecule"], axis =1)
        self.secfp.to_csv(f"{self.save_dir}Secfp.csv", index= False)
        
        # 15. Gobbi
        self.gobbi = self.data.copy()
        print("CALCULATING PHARMACOPHORE GOBBI FINGERPRINTS...")
        self.gobbi["pharmgb"] = self.gobbi.Molecule.progress_apply(self.mol2pharm2dgbfp)
        X = np.stack(self.gobbi["pharmgb"].values)
        d = pd.DataFrame(X)
        #self.gobbi = pd.concat([self.gobbi, d], axis = 1).drop(["pharmgb", "Molecule", self.ID, self.smile_col], axis =1)
        self.gobbi = pd.concat([self.gobbi, d], axis = 1).drop(["pharmgb", "Molecule"], axis =1)
        #self.gobbi.head(2)
        self.gobbi.to_csv(f"{self.save_dir}Ph4_gobbi.csv", index= False)

        # 16. Cats2d
        self.cats2d = self.data.copy()
        print("CALCULATING PHARMACOPHORE CATS2D FINGERPRINTS...")
        self.cats2d["cats2d"] = self.cats2d.Molecule.progress_apply(self.mol2cats)
        X = np.stack(self.cats2d["cats2d"].values)
        d = pd.DataFrame(X)
        #self.cats2d = pd.concat([self.cats2d, d], axis = 1).drop(["cats2d", "Molecule", self.ID, self.smile_col], axis =1)
        self.cats2d = pd.concat([self.cats2d, d], axis = 1).drop(["cats2d", "Molecule"], axis =1)
        self.cats2d.to_csv(f"{self.save_dir}Cats2d.csv", index= False)
        
        # 17. RDK descriptor
        self.rdkdes= self.data.copy()
        print("CALCULATING RDKit descriptors...")
        self.rdkdes["rdkdes"] = self.rdkdes.Molecule.progress_apply(self.RDKDes)
        X = np.stack(self.rdkdes["rdkdes"].values)
        #d = pd.DataFrame(X)
        d = pd.DataFrame(X, columns = [x[0] for x in Descriptors._descList])
        #self.rdkdes = pd.concat([self.rdkdes, d], axis = 1).drop(["rdkdes", "Molecule", self.ID, self.smile_col], axis =1)
        self.rdkdes = pd.concat([self.rdkdes, d], axis = 1).drop(["rdkdes", "Molecule"], axis =1)
        self.rdkdes.to_csv(f"{self.save_dir}RDKdes.csv", index= False)
        
         # 18. Mordred
       
        # self.mord = self.data.copy()
        # print("CALCULATING Mordred descriptors...")
        # calc = Calculator(descriptors, ignore_3D=True)
        
        # df_2d_mordred = calc.pandas(self.mord.Molecule)
        # self.mord = pd.concat([self.mord[[self.activity_col]], df_2d_mordred], axis = 1)
        # self.mord.to_csv(f"{self.save_dir}Mordred.csv", index= False)
        
        # 19. Mol2vec
        self.mol2vec =self.data.copy()
        print("CALCULATING Mol2vec...")
        model = word2vec.Word2Vec.load(self.m2v_path)
        self.mol2vec['sentence'] = self.mol2vec.progress_apply(lambda x: MolSentence(mol2alt_sentence(x['Molecule'], 1)), axis=1)
        self.mol2vec['mol2vec'] = [DfVec(x) for x in sentences2vec(self.mol2vec['sentence'], model, unseen='UNK')]
        X = np.array([x.vec for x in self.mol2vec['mol2vec']])
        #y = np.array(self.data['pChEMBL Value'].astype(float))
        self.mol2vec = pd.concat([self.data[[self.ID, self.smile_col, self.activity_col]], pd.DataFrame(X)], axis = 1)
        self.mol2vec.to_csv(f"{self.save_dir}Mol2vec.csv", index= False)
            
        # 20. AtomParis
        #self.ap = self.data.copy()
        #print("CALCULATING ATOM PAIRS FINGERPRINTS...")
        #self.ap["FPs"] = self.ap.Molecule.progress_apply(self.mol2ap)
        #X = np.stack(self.ap.FPs.values)
        #df = pd.DataFrame(X)
        #self.ap= pd.concat([self.ap, df], axis = 1).drop(["FPs", "Molecule", self.ID, self.smile_col], axis =1)
        #self.ap.to_csv(f"{self.save_dir}Atom Pairs.csv", index= False)
        
        # 21. torsions
        #self.tor = self.data.copy()
        #print("CALCULATING TORSION FINGERPRINTS...")
        #self.tor["FPs"] = self.tor.Molecule.progress_apply(self.mol2torsion)
        #X = np.stack(self.tor.FPs.values)
        #df = pd.DataFrame(X)
        #self.tor= pd.concat([self.tor, df], axis = 1).drop(["FPs", "Molecule", self.ID, self.smile_col], axis =1)
        #self.tor.to_csv(f"{self.save_dir}Torsion.csv", index= False)
        
        print("FINISH CALCULATING!")      
