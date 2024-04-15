import pandas as pd
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import torch_geometric
from torch_geometric.data import Dataset, Data
import os
from tqdm import tqdm
import deepchem as dc
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric.transforms as T
from .conformer_featurizer_onehot import RDKitConformerFeaturizer, RDKitMultipleConformerFeaturizer
from .coord_features import cal_node_spatial_feature, atomic_shape, ring_information, cal_spatial_angle, cal_spatial_distance
from .generate_conformations import generate_conformations
import pickle
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
import time
from rdkit.Chem.MolStandardize import rdMolStandardize
import sys
sys.path.append("..")

from Mol_Featurizer.ultility.standardize import standardization
import glob
#print(f"Torch version: {torch.__version__}")
#print(f"Torch geometric version: {torch_geometric.__version__}")




##################################################################
#         creating a custom dataset for molecule dataset         #
#         applying for one conformation per molecule             #
##################################################################

class MoleculeDataset(Dataset):
    def __init__(self, root, filename, data=None, smiles_col = "Canonicalsmiles", activity_col = "Activity",
                 test=False, valid = False, transform=None, cut_off = 3.5, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.test = test
        self.valid = valid
        self.filename = filename
        self.cut_off = cut_off

        self.smiles_col = smiles_col
        self.activity_col = activity_col
        if data is not None:
            self.create_path(root,data)
        else:
            pass
        super(MoleculeDataset, self).__init__(root, transform, pre_transform)

        
    

    def create_path(self,root,data):
        folder_path = root + "raw/"  # Replace "folder_name" with the desired folder name
        os.makedirs(folder_path, exist_ok=True)
        if self.test == True:
            file_name = self.filename.split(".")[0] + "_test.csv"
            data.to_csv(folder_path + file_name, index = False)
        elif self.valid == True:
            file_name = self.filename.split(".")[0] + "_valid.csv"
            data.to_csv(folder_path + file_name, index = False)
        else:
            file_name = self.filename 
            data.to_csv(folder_path + self.filename, index = False)
        
    def save_conf(self, root, mol, molID, confID, dataID):
        folder_path = root +"_"+ "conformations/"
        os.makedirs(folder_path, exist_ok=True)
        if self.test == True:
            filename = "molecule_" + str(molID) + "_conf_" + str(confID) + "_data_" + str(dataID) + "_test.sdf"
        elif self.valid == True:
            filename = "molecule_" + str(molID) + "_conf_" + str(confID) + "_data_" + str(dataID) + "_valid.sdf"
        else:
            filename = "molecule_" + str(molID) + "_conf_" + str(confID) + "_data_" + str(dataID) + ".sdf"
        writer = Chem.SDWriter(folder_path + filename)
        writer.write(mol, confId=confID)


        
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return self.filename

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()

        if self.test == True:
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        elif self.valid == True:
            return [f'data_valid_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]

    def download(self):
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        #Generate one conformation per molecule
        featurizer = RDKitConformerFeaturizer()
        for index, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            mol = Chem.MolFromSmiles(row[self.smiles_col])
            mol,f = featurizer._featurize(mol)
            #mol_h = Chem.AddHs(mol)
            #pos, conf = featurizer.conformer_generator(mol_h)
            #save file
            self.save_conf(self.processed_dir, mol,molID = index, confID = 0, dataID = index)

            conf = mol.GetConformer()
            pos = torch.tensor(conf.GetPositions())

            #pos = torch.tensor(pos, dtype=torch.float)
            #node features
            node_features = torch.tensor(f.node_features, dtype=torch.float)
            #edge index
            edge_index = torch.tensor(f.edge_index, dtype=torch.int64)
            #edge features
            edge_features = torch.tensor(f.edge_features, dtype=torch.float)
            #spatial information
            spatial_information = cal_node_spatial_feature(pos, cut_off = self.cut_off) #3D node spatial feature
            #atomic shape
            atomic_shape_info, lone_pairs = atomic_shape(row[self.smiles_col]) #atomic shape information, lone pairs
            #ring information
            #ring_info = ring_information(mol, edge_index) #ring information 2D
            #spatial angle
            spatial_angle = cal_spatial_angle(pos,mol, edge_index) #3D spatial angle bond information
            row_inx, col_idx = edge_index

            dist = (pos[col_idx] - pos[row_inx]).pow(2).sum(dim=-1).sqrt().reshape(-1,1) 
            edge_attr_3d = torch.cat([spatial_angle, dist], dim = 1)

            #spatial distance
            spatial_distance = cal_spatial_distance(pos) #3D spatial distance information nodes
            
            node_features_2d = torch.cat([node_features, lone_pairs], dim = 1)
            node_features_3d = torch.cat([atomic_shape_info, spatial_information,spatial_distance], dim = 1)

            #bond_features_2d = torch.cat([edge_features, ring_info], dim = 1)


            

            label = self._get_labels(row[self.activity_col])
            smiles = row[self.smiles_col]
            data = Data(node_features_2d=node_features_2d, 
                        edge_index=edge_index,
                        edge_attr_2d=bond_features_2d,
                        node_features_3d = node_features_3d,
                        edge_attr_3d = edge_attr_3d,
                        pos = pos,
                        y = label,
                        smiles= row[self.smiles_col]
                        ) 

            if self.test ==True:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                 f'data_test_{index}.pt'))
            elif self.valid == True:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                 f'data_valid_{index}.pt'))
            else:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                 f'data_{index}.pt'))

    

    def _get_labels(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        if self.test == True :
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_test_{idx}.pt'))
        elif self.valid == True:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_valid_{idx}.pt'))
            
        else:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{idx}.pt'))   
        return data




##################################################################
#         creating a bag dataset for molecule dataset            #
#         applying for multiple conformation per molecule        #
##################################################################
    


class BagMoleculeDataset(torch.utils.data.Dataset):
    def __init__(self, root, filename, data=None, smiles_col = "Canonicalsmiles", activity_col = "Activity",
                 test=False, valid = False, transform=None, cut_off = 3.5, 
                 num_confs = 10, seed = 42, rmsd = 0.5, energy =10, max_attemp = 1000, num_threads = -1,
                 pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        Returns:
        ----------
        Dataset: Pytorch Geometric Dataset
            Bag dataset, each bag contains multiple conformations of a molecule 
        """
        self.test = test
        self.valid = valid
        self.filename = filename
        self.cut_off = cut_off
        self.num_confs = num_confs
        self.seed = seed
        self.rmsd = rmsd
        self.energy = energy
        self.max_attemp = max_attemp
        self.num_threads = num_threads
        self.smiles_col = smiles_col
        self.activity_col = activity_col

        self.processed_dir = root + "processed/"
        os.makedirs(self.processed_dir, exist_ok=True)
        self.raw_path = root + "raw/"
        os.makedirs(self.raw_path, exist_ok=True)
        file_count = len(os.listdir(self.processed_dir))
        if self.valid == True:
            file_list = glob.glob(self.processed_dir + "/*val*")
            self.file_count = len(file_list)

        elif self.test == True:
            file_list = glob.glob(self.processed_dir + "/*test*")
            self.file_count = len(file_list)
        else:
            file_list = glob.glob(self.processed_dir + "/*")
            filtered_files = [file for file in file_list if "val" not in file and "test" not in file]
            self.file_count = len(filtered_files)

        if self.file_count == 0:
            self.save_data_raw(data)
            self.process()
        else:
            pass
        
        
        #super(BagMoleculeDataset, self).__init__(root, transform, pre_transform)
        
     

    def save_data_raw(self,data):
        folder_path = self.raw_path
        #os.makedirs(folder_path, exist_ok=True)
        if self.test == True:
            #file_name = self.filename.split(".")[0] + "_test.csv"
            file_name = self.filename
            data.to_csv(folder_path + file_name, index = False)
        elif self.valid == True:
            file_name = self.filename
            data.to_csv(folder_path + file_name, index = False)
        else:
            file_name = self.filename 
            data.to_csv(folder_path + self.filename, index = False)
        
    def save_conf(self, root, mol, molID, confID, dataID):
        folder_path = root + "conformations_bag/"
        os.makedirs(folder_path, exist_ok=True)
        if self.test == True:
            filename = "molecule_" + str(molID) + "_conf_" + str(confID) + "_data_" + str(dataID) + "_test.sdf"
        elif self.valid == True:
            filename = "molecule_" + str(molID) + "_conf_" + str(confID) + "_data_" + str(dataID) + "_valid.sdf"
        else:
            filename = "molecule_" + str(molID) + "_conf_" + str(confID) + "_data_" + str(dataID) + ".sdf"
        writer = Chem.SDWriter(folder_path + filename)
        writer.write(mol, confId=confID)

        # folder_path = root + "_"+"conformations_bag/"
        # os.makedirs(folder_path, exist_ok=True)
        # with open(folder_path + f'conformations_{index}.pkl', 'wb') as f:
        #     pickle.dump(data, f)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """

        return self.filename

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        self.data = pd.read_csv(self.raw_path).reset_index()

        if self.test == True:
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        elif self.valid == True:
            return [f'data_valid_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]

    def download(self):
        pass

    def process(self):
        data_path = self.raw_path + "/" + self.filename
        self.data = pd.read_csv(data_path).reset_index()
        std = standardization(data=self.data,ID='ID', smiles_col=self.smiles_col, active_col= self.activity_col, ro5 =0)
        std_data = std.filter_data()
        featurizer = RDKitMultipleConformerFeaturizer(num_confs = self.num_confs, seed = self.seed,
                                                     rmsd = self.rmsd, energy =self.energy, max_attemp = self.max_attemp)

        def process_data(index, row):
            
        #for index, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            #mol = Chem.MolFromSmiles(row[self.smiles_col])
            try:
                #bag_list = []
                mol = row["Molecule"]
                mol,f, conf_num = featurizer._featurize(mol)

                if len(mol.GetConformers()) == 0:
                    print(f"Molecule {index} has no conformation")
                    pass
                else:
    

                #Saving conformations
                #self.save_conf(self.processed_dir, mol,index)

                #node features
                    node_features = torch.tensor(f.node_features, dtype=torch.float)
                #edge index
                    edge_index = torch.tensor(f.edge_index, dtype=torch.int64)
                    #edge features
                    edge_features = torch.tensor(f.edge_features, dtype=torch.float)
                    #ring information
                    #ring_info = ring_information(mol, edge_index) #ring information 2D
                    #atomic shape
                    atomic_shape_info, lone_pairs = atomic_shape(row["Standardize_smile"]) #atomic shape information, lone pairs
                    GraphDataBag = []
                    #bag_lable = self._get_labels(row["Activity"])
                    for conf in mol.GetConformers():
                        #Saving SDF file
                        self.save_conf(self.processed_dir, mol = mol,molID = index, confID = conf.GetId(), dataID = index)

                        #Getting 3D coordinates
                        pos = torch.tensor(conf.GetPositions(), dtype=torch.float)

                        #spatial information
                        spatial_information = cal_node_spatial_feature(pos, cut_off = self.cut_off) #3D node spatial feature


                        #spatial angle
                        spatial_angle = cal_spatial_angle(pos,mol, edge_index) #3D spatial angle bond information
                        row_inx, col_idx = edge_index

                        dist = (pos[col_idx] - pos[row_inx]).pow(2).sum(dim=-1).sqrt().reshape(-1,1) 
                        edge_attr_3d = torch.cat([spatial_angle, dist], dim = 1)
                        #spatial distance
                        spatial_distance = cal_spatial_distance(pos) #3D spatial distance information nodes
                    
                        node_features_2d = torch.cat([node_features, lone_pairs], dim = 1)
                        node_features_3d = torch.cat([atomic_shape_info, spatial_information,spatial_distance], dim = 1)

                        #bond_features_2d = torch.cat([edge_features, ring_info], dim = 1)
                        #bond_features_3d = torch.cat([spatial_angle, spatial_distance], dim = 1)

                    

                        label = self._get_labels(row[self.activity_col])
                        smiles = row["Standardize_smile"]
                        conf_id = conf.GetId()
                        data = Data(node_features_2d=node_features_2d, 
                                edge_index=edge_index,
                                #edge_attr_2d=bond_features_2d,
                                edge_attr_2d = edge_features,
                                node_features_3d = node_features_3d,
                                edge_attr_3d = edge_attr_3d,
                                pos = pos,
                                y = label,
                                smiles= smiles,
                                conf_id = conf_id)
                        
                        GraphDataBag.append(data)

                    label = self._get_labels(row[self.activity_col])
                    smiles = row["Standardize_smile"]
                    bag_data = Data(instance_data = GraphDataBag,
                                        y = label, 
                                        smiles = smiles,
                                        num_instances = len(GraphDataBag))
                    
                    #bag_list.append(bag_data)
                    return bag_data
            except:
                print(f"Error: molecule {index} has no conformation")
                pass

        with joblib_progress("Processing data...", total= std_data.shape[0]):  
            process_data = Parallel(n_jobs=self.num_threads)(delayed(process_data)(index, row) for index, row in std_data.iterrows())
        
        index = -1
        for bag in process_data:
            if bag is not None:
                index += 1
                if self.test ==True:
                    torch.save(bag, 
                        os.path.join(self.processed_dir, 
                                    f'data_test_{index}.pt'))
                elif self.valid == True:
                    torch.save(bag, 
                        os.path.join(self.processed_dir, 
                                    f'data_valid_{index}.pt'))
                else:
                    torch.save(bag, 
                        os.path.join(self.processed_dir, 
                                    f'data_{index}.pt'))
            else:
                pass

    def _get_labels(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

    def __len__(self):
        if self.valid == True:
            file_list = glob.glob(self.processed_dir + "/*val*")
            count = len(file_list)
            return count
        
        if self.test == True:
            file_list = glob.glob(self.processed_dir + "/*test*")
            count = len(file_list)
            return count
        
        else: 
            file_list = glob.glob(self.processed_dir + "/*")
            filtered_files = [file for file in file_list if "val" not in file and "test" not in file]
            return len(filtered_files)-1

    def __getitem__(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        if self.test == True :
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_test_{idx}.pt'))
        elif self.valid == True:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_valid_{idx}.pt'))
            
        else:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{idx}.pt'))   
        return data
    



##################################################################
#         creating a instances dataset for molecule dataset      #
#         applying for multiple conformation per molecule        #
##################################################################
    


class InstanceMoleculeDataset(Dataset):
    def __init__(self, root, filename, data=None, smiles_col = "Canonicalsmiles", activity_col = "Activity",
                 test=False, valid = False, transform=None, cut_off = 3.5, 
                 num_confs = 10, seed = 42, rmsd = 0.5, energy =10, max_attemp = 1000, num_threads = -1,
                 pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        Returns:
        ----------
        Dataset: Pytorch Geometric Dataset
            Bag dataset, each bag contains multiple conformations of a molecule 
        """
        self.test = test
        self.valid = valid
        self.filename = filename
        self.cut_off = cut_off
        self.num_confs = num_confs
        self.seed = seed
        self.rmsd = rmsd
        self.energy = energy
        self.max_attemp = max_attemp
        self.num_threads = num_threads
        self.smiles_col = smiles_col
        self.activity_col = activity_col
        

        
        if data is not None:
            self.create_path(root,data)
        else:
            pass
        super(InstanceMoleculeDataset, self).__init__(root, transform, pre_transform)
        
    

    def create_path(self,root,data):
        folder_path = root + "raw/"  # Replace "folder_name" with the desired folder name
        os.makedirs(folder_path, exist_ok=True)
        if self.test == True:
            file_name = self.filename.split(".")[0] + "_test.csv"
            data.to_csv(folder_path + file_name, index = False)
        elif self.valid == True:
            file_name = self.filename.split(".")[0] + "_valid.csv"
            data.to_csv(folder_path + file_name, index = False)
        else:
            file_name = self.filename 
            data.to_csv(folder_path + self.filename, index = False)
        
    # def save_conf(self, root, data, index):
    #     folder_path = root + "_"+"conformations_instances/"
    #     os.makedirs(folder_path, exist_ok=True)
    #     with open(folder_path + f'conformations_{index}.pkl', 'wb') as f:
    #         pickle.dump(data, f)
            
    def save_conf(self, root, mol, molID, confId, dataID):
        folder_path = root +"_"+ "conformations_instances/"
        os.makedirs(folder_path, exist_ok=True)
        if self.test == True:
            filename = "molecule_" + str(molID) + "_conf_" + str(confId) + "_data_" + str(dataID) + "_test.sdf"
        elif self.valid == True:
            filename = "molecule_" + str(molID) + "_conf_" + str(confId) + "_data_" + str(dataID) + "_valid.sdf"
        else:
            filename = "molecule_" + str(molID) + "_conf_" + str(confId) + "_data_" + str(dataID) + ".sdf"

        writer = Chem.SDWriter(folder_path + filename)
        writer.write(mol, confId=confId)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        #print(self.has_process)
        #print("raw file names")
        return self.filename

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        #print("processed file names")
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()

        if self.test == True:
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        elif self.valid == True:
            return [f'data_valid_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]

    def download(self):
        #print("download")
        pass

    def process(self):
        #print("process")
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        featurizer = RDKitMultipleConformerFeaturizer(num_confs = self.num_confs, seed = self.seed,
                                                     rmsd = self.rmsd, energy =self.energy, max_attemp = self.max_attemp)
        
        energy_dict = {}
        self.file_index = -1
        for index, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            mol = Chem.MolFromSmiles(row[self.smiles_col])
            mol,f, energy = featurizer._featurize(mol)
            key_e = "molecule_" + str(index)
            value_e = energy
            energy_dict.update({key_e:value_e})


            #Saving conformations
            #self.save_conf(self.processed_dir, mol,index)

            #node features
            node_features = torch.tensor(f.node_features, dtype=torch.float)
            #edge index
            edge_index = torch.tensor(f.edge_index, dtype=torch.int64)
            #edge features
            edge_features = torch.tensor(f.edge_features, dtype=torch.float)
            #ring information
            ring_info = ring_information(mol, edge_index) #ring information 2D
            #atomic shape
            atomic_shape_info, lone_pairs = atomic_shape(row[self.smiles_col]) #atomic shape information, lone pairs
            #GraphDataBag = []
            #bag_lable = self._get_labels(row["Activity"])
            #self.file_index = -1
            for conf in mol.GetConformers():
                self.file_index += 1
                #saving SDF file
                self.save_conf(self.processed_dir, mol = mol,molID = index, confId = conf.GetId(), dataID = self.file_index)

                #Getting 3D coordinates
                pos = torch.tensor(conf.GetPositions(), dtype=torch.float)

                #spatial information
                spatial_information = cal_node_spatial_feature(pos, cut_off = self.cut_off) #3D node spatial feature


                #spatial angle
                spatial_angle = cal_spatial_angle(pos,mol, edge_index) #3D spatial angle bond information
                #spatial distance
                spatial_distance = cal_spatial_distance(pos) #3D spatial distance information nodes
            
                node_features_2d = torch.cat([node_features, lone_pairs], dim = 1)
                node_features_3d = torch.cat([atomic_shape_info, spatial_information,spatial_distance], dim = 1)

                bond_features_2d = torch.cat([edge_features, ring_info], dim = 1)

                #self.save_conf(self.processed_dir, mol,index)

                label = self._get_labels(row[self.activity_col])
                smiles = row[self.smiles_col]
                conf_id = conf.GetId()
                data = Data(node_features_2d=node_features_2d, 
                        edge_index=edge_index,
                        edge_attr_2d=bond_features_2d,
                        node_features_3d = node_features_3d,
                        edge_attr_3d = spatial_angle,
                        pos = pos,
                        y = label,
                        smiles= row[self.smiles_col],
                        conf_id = conf_id)
                 

                if self.test ==True:
                    torch.save(data, 
                        os.path.join(self.processed_dir, 
                                    f'data_test_{self.file_index}.pt'))
                elif self.valid == True:
                    torch.save(data, 
                        os.path.join(self.processed_dir, 
                                    f'data_valid_{self.file_index}.pt'))
                else:
                    torch.save(data, 
                        os.path.join(self.processed_dir, 
                                    f'data_{self.file_index}.pt'))
                    
        #Saving energy dictionary
        enery_path = folder_path = self.processed_dir +"_"+ "conformations_instances/" + "energy_dict.pkl"
        with open(enery_path, 'wb') as f:
            pickle.dump(energy_dict, f)

    def _get_labels(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

    def len(self):
        return self.file_index+1

    def get(self, file_index):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        if self.test == True :
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_test_{file_index}.pt'))
        elif self.valid == True:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_valid_{file_index}.pt'))
            
        else:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{file_index}.pt'))   
        return data
    
###########################################################################
#         Parallel creating a instances dataset for molecule dataset      #
#         applying for multiple conformation per molecule                 #
###########################################################################
class InstanceMoleculeDataset_Parallel(torch.utils.data.Dataset):
    def __init__(self, root, filename, data=None, smiles_col = "Canonicalsmiles", activity_col = "Activity", 
                 test=False, valid = False, transform=None, cut_off = 3.5, 
                 num_confs = 10, seed = 42, rmsd = 0.5, energy =10, max_attemp = 1000, num_threads = -1,
                 pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        Returns:
        ----------
        Dataset: Pytorch Geometric Dataset
            Bag dataset, each bag contains multiple conformations of a molecule 
        """
        self.test = test
        self.valid = valid
        self.filename = filename
        self.cut_off = cut_off
        self.num_confs = num_confs
        self.seed = seed
        self.rmsd = rmsd
        self.energy = energy
        self.max_attemp = max_attemp
        self.num_threads = num_threads
        self.smiles_col = smiles_col
        self.activity_col = activity_col
        #self.device =  torch.device("mps" if torch.cuda.is_available() else "cpu")

        self.processed_dir = root + "processed/"
        os.makedirs(self.processed_dir, exist_ok=True)
        self.raw_path = root + "raw/"
        os.makedirs(self.raw_path, exist_ok=True)

        if self.valid == True:
            file_list = glob.glob(self.processed_dir + "/*val*")
            self.file_count = len(file_list)

        elif self.test == True:
            file_list = glob.glob(self.processed_dir + "/*test*")
            self.file_count = len(file_list)
        else:
            self.file_count = len(os.listdir(self.processed_dir))

        if self.file_count == 0:
            self.save_data_raw(root,data)
            self.process()
        else:
            pass
        # super(InstanceMoleculeDataset_Parallel, self).__init__(root, transform, pre_transform)
    def save_data_raw(self,root,data):
        folder_path = self.raw_path   # Replace "folder_name" with the desired folder name
        #os.makedirs(folder_path, exist_ok=True)
        if self.test == True:
            #file_name = self.filename.split(".")[0] + "_test.csv"
            file_name = self.filename
            data.to_csv(folder_path + file_name, index = False)
        elif self.valid == True:
            file_name = self.filename
            data.to_csv(folder_path + file_name, index = False)
        else:
            file_name = self.filename 
            data.to_csv(folder_path + self.filename, index = False)

    def save_conf(self, root, mol, molID, confId):
        folder_path = root + "conformations_instances/"
        os.makedirs(folder_path, exist_ok=True)
        if self.test == True:
            filename = "molecule_" + str(molID) + "_conf_" + str(confId) + "_test.sdf"
        elif self.valid == True:
            filename = "molecule_" + str(molID) + "_conf_" + str(confId) + "_data_" + "_valid.sdf"
        else:
            filename = "molecule_" + str(molID) + "_conf_" + str(confId) + "_data_"  + ".sdf"

        writer = Chem.SDWriter(folder_path + filename)
        writer.write(mol, confId=confId)
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        #print(self.has_process)
        #print("raw file names")
        return self.filename
    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        #print("processed file names")
       #print(self.processed_dir)
        data_path = self.raw_path + "/" + self.filename
        self.data = pd.read_csv(self.raw_path).reset_index()

        if self.test == True:
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        elif self.valid == True:
            return [f'data_valid_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]
        
    def download(self):
        #print("download")
        pass
    def process(self):
        #print("process")
        data_path = self.raw_path + "/" + self.filename
        self.data = pd.read_csv(data_path).reset_index()
        std =  standardization(data=self.data,ID='ID', smiles_col=self.smiles_col, active_col= self.activity_col, ro5 =0)
        data_std = std.filter_data()
        featurizer = RDKitMultipleConformerFeaturizer(num_confs = self.num_confs, seed = self.seed,
                                                     rmsd = self.rmsd, energy =self.energy, max_attemp = self.max_attemp)
        
        def process_data(index, row):
            Graph_data = []
            mol = row["Molecule"]
            #standardize_mol, standardize_smiles = self.standardize_mol(mol)
            try:
                mol, f, conf_num = featurizer._featurize(mol)
                #cids = len(mol.GetConformers())
                if len(mol.GetConformers()) == 0:
                    print(f"Error: molecule {index} has no conformations")
                    

        
                # if energy is None:
                #     print(f"Error: molecule {index} has no energy")
                #     return
                #     #energy = [0]*self.num_confs 

                #node features
                node_features = torch.tensor(f.node_features, dtype=torch.float)

                #edge index
                edge_index = torch.tensor(f.edge_index, dtype=torch.int64)
                #edge features
                edge_features = torch.tensor(f.edge_features, dtype=torch.float)
                #ring information
                ring_info = ring_information(mol, edge_index) #ring information 2D
                #atomic shape
                atomic_shape_info, lone_pairs = atomic_shape(row["Standardize_smile"]) #atomic shape information, lone pairs

                for conf in mol.GetConformers():
                    #saving SDF file
                    self.save_conf(self.processed_dir, mol = mol,molID = index, confId = conf.GetId())

                    #Getting 3D coordinates
                    pos = torch.tensor(conf.GetPositions(), dtype=torch.float)

                    #spatial information
                    spatial_information = cal_node_spatial_feature(pos, cut_off = self.cut_off) #3D node spatial feature


                    #spatial angle
                    spatial_angle = cal_spatial_angle(pos,mol, edge_index) #3D spatial angle bond information
                    #spatial distance
                    spatial_distance = cal_spatial_distance(pos) #3D spatial distance information nodes
                
                    node_features_2d = torch.cat([node_features, lone_pairs], dim = 1)
                    node_features_3d = torch.cat([atomic_shape_info, spatial_information,spatial_distance], dim = 1)

                    bond_features_2d = torch.cat([edge_features, ring_info], dim = 1)

                    #self.save_conf(self.processed_dir, mol,index)

                    label = self._get_labels(row[self.activity_col])
                    conf_id = conf.GetId()
                    data = Data(node_features_2d=node_features_2d, 
                            edge_index=edge_index,
                            edge_attr_2d=bond_features_2d,
                            node_features_3d = node_features_3d,
                            edge_attr_3d = spatial_angle,
                            #energy = energy[conf.GetId()],
                            pos = pos,
                            y = label,
                            smiles= row["Standardize_smile"],
                            conf_id = conf_id)
                    Graph_data.append(data) 
                return Graph_data, conf_num
            except:
                print(f"Error: molecule {index} has problem")
                pass
                 
                    
        #start = time.time()
        with joblib_progress("Processing data...", total= self.data.shape[0]):  
            process_data = Parallel(n_jobs=self.num_threads)(delayed(process_data)(index, row) for index, row in data_std.iterrows())
            


        self.file_index = -1
        conf_dict = {}
        #print("Done parallel processing")
        for mol_index, bag in enumerate(process_data):
            if bag is None:
                print(f"Error: molecule {mol_index} has problem")
                continue
            #append conf dictionary
            key_e = "mol"+str(mol_index)
        
            #print(key_e, type(key_e))
            value_e = bag[1]
            conf_dict.update({key_e:value_e})

            #print("Saving data")
            for data_index, data in enumerate(bag[0]):
                self.file_index += 1
                if self.test == True :
                    torch.save(data, 
                        os.path.join(self.processed_dir, 
                                    f'data_test_{self.file_index}.pt'))
                elif self.valid == True:
                    torch.save(data, 
                        os.path.join(self.processed_dir, 
                                    f'data_valid_{self.file_index}.pt'))
                else:
                    torch.save(data,
                        os.path.join(self.processed_dir, 
                                    f'data_{self.file_index}.pt'))
        #Saving energy dictionary
        #print("Saving energy dictionary")
        if self.test == True:
            conf_path = self.processed_dir + "conformations_instances/" + "num_conf_test.pkl"
        elif self.valid == True:
            conf_path = self.processed_dir + "conformations_instances/" + "num_conf_val.pkl"
        else:
            conf_path = self.processed_dir + "conformations_instances/" + "num_conf.pkl"    
        with open(conf_path, 'wb') as f:
            pickle.dump(conf_dict, f)

    def _get_labels(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)
    def __len__(self):
        if self.valid == True:
            file_list = glob.glob(self.processed_dir + "/*val*")
            count = len(file_list)
            return count
        
        if self.test == True:
            file_list = glob.glob(self.processed_dir + "/*test*")
            count = len(file_list)
            return count
        
        else: 
            file_list = glob.glob(self.processed_dir + "/*")
            filtered_files = [file for file in file_list if "val" not in file and "test" not in file]
        #file_count = len(os.listdir(self.processed_dir))
            return len(filtered_files)-1
    def __getitem__(self, file_index):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        if self.test:
            return torch.load(os.path.join(self.processed_dir, 
                                 f'data_test_{file_index}.pt'))
        elif self.valid:
            return torch.load(os.path.join(self.processed_dir, 
                                 f'data_valid_{file_index}.pt'))
            
        else:
            return torch.load(os.path.join(self.processed_dir, 
                                 f'data_{file_index}.pt'))