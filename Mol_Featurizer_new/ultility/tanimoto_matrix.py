import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit.Chem import  AllChem
from rdkit import Chem, DataStructs 
from mhfp.encoder import MHFPEncoder
from rdkit.Chem import rdMHFPFingerprint

import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

class similarity_matrix:
    """
    A class that represents a similarity matrix.

    Attributes:
        data_train (DataFrame): The training data.
        ID (str): The column name for the ID.
        smiles_col (str): The column name for the SMILES.

    Methods:
        tanimoto(vector1, vector2): Calculates the Tanimoto similarity between two vectors.
        fp_similarity(fp1, fp2): Calculates the fingerprint similarity between two fingerprints.
        train_process(): Processes the training data.
        fit(): Fits the similarity matrix.
    """

    def __init__(self, data, ID):
        self.data_train = data
        self.ID = ID
        #self.smiles_col = smiles_col
        #self.activity_col = activity_col
    
    def tanimoto(self, vector1, vector2):
        """
        Calculates the Tanimoto similarity between two vectors.

        Args:
            vector1 (array-like): The first vector.
            vector2 (array-like): The second vector.

        Returns:
            float: The Tanimoto similarity between the two vectors.
        """
        a = np.where(vector1 == 1)[0]
        b = np.where(vector2 == 1)[0]
        return len(np.intersect1d(a, b)) / (float(len(a) + len(b)) - len(np.intersect1d(a, b)))
    
    def fp_similarity(self, fp1, fp2):
        """
        Calculates the fingerprint similarity between two fingerprints.

        Args:
            fp1 (array-like): The first fingerprint.
            fp2 (array-like): The second fingerprint.

        Returns:
            float: The fingerprint similarity between the two fingerprints.
        """
        return self.tanimoto(fp1, fp2)
    
    def train_process(self):
        """
        Processes the training data.
        """
        #self.list_training_name = list(self.data_train[self.ID].values)
        df_fp = self.data_train
        #df_fp = self.data_train.drop([self.ID, self.smiles_col], axis=1)
        self.list_training_fp = list(df_fp.values)
        
    def fit(self):
        """
        Fits the similarity matrix.
        """
        self.train_process()
        self.list_data_set = self.ID
        self.list_data_set_fp = self.list_training_fp


        size = len(self.list_data_set_fp)
        self.matrix = pd.DataFrame()
        for m, i in tqdm(enumerate(self.list_data_set_fp), total=size, desc='Filling matrix'):
            for n, j in enumerate(self.list_data_set_fp):
                similarity = self.fp_similarity(i, j)
                self.matrix.loc[self.list_data_set[m], self.list_data_set[n]] = similarity
                #print(f"Filling matrix[{m}][{n}] with similarity: {similarity}")
