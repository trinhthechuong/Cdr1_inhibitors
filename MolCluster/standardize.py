import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit import DataStructs, Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.rdBase import BlockLogs
from medicinal_chemistry import calculate_ro5_properties
class standardization:
    """""
    Input:
    - df: dataframe
        must have Smiles column, ID column and bioactive columns
    - smiles_col: string
        name of smile column
    - active_col: string
        name of bioactive column ~ pIC50
    - mw: int => recomend 600
        molecular weight cutoff, value above cutoff will be removed
    Return:
    - data: dataframe
           
    """"" 
    def __init__(self,data,ID,smiles_col, active_col, ro5 =4):
        self.data = data
        self.ID = ID
        self.smiles_col = smiles_col
        self.active_col = active_col
        self.ro5 = ro5
        
    def standardize(self, smiles):
        # Code borrowed from https://bitsilla.com/blog/2021/06/standardizing-a-molecule-using-rdkit/
        # follows the steps in
        # https://github.com/greglandrum/RSC_OpenScience_Standardization_202104/blob/main/MolStandardize%20pieces.ipynb
        # as described **excellently** (by Greg) in
        # https://www.youtube.com/watch?v=eWTApNX8dJQ
        mol = Chem.MolFromSmiles(smiles)
        # removeHs, disconnect metal atoms, normalize the molecule, reionize the molecule
        clean_mol = rdMolStandardize.Cleanup(mol) 
        # if many fragments, get the "parent" (the actual mol we are interested in) 
        parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
        # try to neutralize molecule
        uncharger = rdMolStandardize.Uncharger() # annoying, but necessary as no convenience method exists
        uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
        # note that no attempt is made at reionization at this step
        # nor at ionization at some pH (rdkit has no pKa caculator)
        # the main aim to to represent all molecules from different sources
        # in a (single) standard way, for use in ML, catalogue, etc.
        te = rdMolStandardize.TautomerEnumerator() # idem
        taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)
        return taut_uncharged_parent_clean_mol
    
    
    def filter_data(self):
        self.data['Canomicalsmiles'] = self.data[self.smiles_col].apply(Chem.CanonSmiles)
        self.data = self.data[self.data['Canomicalsmiles'].progress_apply(calculate_ro5_properties, fullfill = self.ro5)]  
        block = BlockLogs()
        self.data['Molecule'] = self.data['Canomicalsmiles'].progress_apply(self.standardize)
        
        return self.data
