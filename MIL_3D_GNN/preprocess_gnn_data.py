import pandas as pd
import sys
sys.path.append("..")
from Mol_Featurizer.ultility.standardize import standardization
from conformation_encode.GraphDataset import BagMoleculeDataset, InstanceMoleculeDataset, MoleculeDataset 
from rdkit.rdBase import BlockLogs


path_train = "/Users/thechuongtrinh/Documents/Workspace/Master_thesis/Cdr1_inhibitors/dataset/Featurized_data/BM_stratified_sampling/training_set/rdk7_train.csv"
path_hard_test = "/Users/thechuongtrinh/Documents/Workspace/Master_thesis/Cdr1_inhibitors/dataset/Featurized_data/BM_stratified_sampling/hard_test_set/rdk7_hard_test.csv"
path_test = "/Users/thechuongtrinh/Documents/Workspace/Master_thesis/Cdr1_inhibitors/dataset/Featurized_data/BM_stratified_sampling/external_test_set/rdk7_test.csv"
path_valid = "/Users/thechuongtrinh/Documents/Workspace/Master_thesis/Cdr1_inhibitors/dataset/Featurized_data/BM_stratified_sampling/validation_set_dl/rdk7_valid.csv"
sample_path = "/Users/thechuongtrinh/Documents/Workspace/Master_thesis/Cdr1_inhibitors/dataset/Featurized_data/BM_stratified_sampling/train_sample.csv"
data_train = pd.read_csv(path_train)
data_test = pd.read_csv(path_test)
data_valid = pd.read_csv(path_valid)
data_hard_test = pd.read_csv(path_hard_test)    
sample_df = pd.read_csv(sample_path).drop([0],axis = 0).reset_index(drop=True)
print(sample_df.head())
#Binary

train_dataset = BagMoleculeDataset(root = "/Users/thechuongtrinh/Documents/Workspace/Master_thesis/Cdr1_inhibitors/dataset/Featurized_data/MIL_3D_GNN/train_test_val_1/",
                            filename = "cdr1_train.csv",
                            data = data_train, smiles_col = "Standardize_smile", activity_col = "Activity",
                            cut_off = 3.5, num_confs = 50, seed = 42, rmsd = 0.5, 
                            energy =10, max_attemp = 1000, num_threads = 4)

test_dataset = BagMoleculeDataset(root = "/Users/thechuongtrinh/Documents/Workspace/Master_thesis/Cdr1_inhibitors/dataset/Featurized_data/MIL_3D_GNN/train_test_val_1/",
                            filename = "cdr1_test.csv", test = True,
                            data = data_test, smiles_col = "Standardize_smile", activity_col = "Activity",
                            cut_off = 3.5, num_confs = 50, seed = 42, rmsd = 0.5, 
                            energy =10, max_attemp = 1000, num_threads = 4)

valid_dataset = BagMoleculeDataset(root = "/Users/thechuongtrinh/Documents/Workspace/Master_thesis/Cdr1_inhibitors/dataset/Featurized_data/MIL_3D_GNN/train_test_val_1/",
                            filename = "cdr1_valid.csv",valid=True,
                            data = data_valid, smiles_col = "Standardize_smile", activity_col = "Activity",
                            cut_off = 3.5, num_confs = 50, seed = 42, rmsd = 0.5, 
                            energy =10, max_attemp = 1000, num_threads = 4)

sample_dataset = BagMoleculeDataset(root = "/Users/thechuongtrinh/Documents/Workspace/Master_thesis/Cdr1_inhibitors/dataset/Featurized_data/MIL_3D_GNN/svm_sampling_1/",
                            filename = "cdr1_sample.csv",
                            data = sample_df, smiles_col = "Standardize_smile", activity_col = "Activity",
                            cut_off = 3.5, num_confs = 100, seed = 0, rmsd = 0.5, 
                            energy =10, max_attemp = 1000, num_threads = 4)

hard_test_dataset = BagMoleculeDataset(root = "/Users/thechuongtrinh/Documents/Workspace/Master_thesis/Cdr1_inhibitors/dataset/Featurized_data/MIL_3D_GNN/hard_test_1/",
                            filename = "cdr1_hard_test.csv",
                            data = data_hard_test, smiles_col = "Standardize_smile", activity_col = "Activity", test = True,
                            cut_off = 3.5, num_confs = 50, seed = 42, rmsd = 0.5, 
                            energy =10, max_attemp = 1000, num_threads = 4)