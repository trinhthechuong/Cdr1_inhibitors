import sys
sys.path.append("..")
from Utility.Featurizer_deploy import representation_calculation
from conformation_encode.GraphDataset import BagMoleculeDataset
from Mol_Standardize.standardize import standardization
import shutil
import streamlit as st
import os
import numpy as np
import pandas as pd

class featurizer():
    def __init__(self, data, ID = "ID", smiles_col = "Standardize_smile", active_col = "Activity", save_dir = "Cdr1_classification/", n_jobs = -1):
        self.save_dir = save_dir
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        self.data = data
        self.ID = ID
        self.smiles_col = smiles_col
        self.active_col = active_col
        self.n_jobs = n_jobs
    def standardized_data(self):
        # Initialize a progress bar
        #print("Starting standardization...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Update status to indicate the process has started
        status_text.text("Starting standardization...")
        progress_bar.progress(10)
        
        # Perform the standardization
        std = standardization(data=self.data, ID=self.ID, smiles_col=self.smiles_col, active_col=self.active_col, ro5=0)
        self.data_std = std.filter_data()
        
        # Update progress bar and status text to indicate completion
        progress_bar.progress(100)
        status_text.text("Standardization completed.")
    
        return self.data_std
    
    def featurized_data(self, data_featurized):
        # Initialize a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        progress_step = 1 / 7  # Since we have 7 steps, each step will be ~14.3%

        #data_featurized = self.data_std.copy()
        #print("RDKit fingerprint (rdk5) calculation...")
        rdk5_cal = representation_calculation(data_featurized, self.ID, self.smiles_col, Molecule_col="Molecule", type_fpt='rdk5', save_dir=self.save_dir)
        rdk5_df = rdk5_cal.fit()
        progress_bar.progress(progress_step)
        status_text.text("RDKit fingerprint (rdk5) calculated...")

        #print("RDKit fingerprint (rdk6) calculation...")
        rdk6_cal = representation_calculation(data_featurized, self.ID, self.smiles_col, Molecule_col="Molecule", type_fpt='rdk6', save_dir=self.save_dir)
        rdk6_df = rdk6_cal.fit()
        progress_bar.progress(2 * progress_step)
        status_text.text("RDKit fingerprint (rdk6) calculated...")

        #print("RDKit fingerprint (rdk7) calculation...")
        rdk7_cal = representation_calculation(data_featurized, self.ID, self.smiles_col, Molecule_col="Molecule", type_fpt='rdk7', save_dir=self.save_dir)
        rdk7_df = rdk7_cal.fit()
        progress_bar.progress(3 * progress_step)
        status_text.text("RDKit fingerprint (rdk7) calculated...")

        #print("Avalon fingerprint calculation...")
        avalon_cal = representation_calculation(data_featurized, self.ID, self.smiles_col, Molecule_col="Molecule", type_fpt='avalon', save_dir=self.save_dir)
        avalon_df = avalon_cal.fit()
        progress_bar.progress(4 * progress_step)
        status_text.text("Avalon fingerprint calculated...")

        #print("Pharmacophore features calculation...")
        cal_ph4 = representation_calculation(data_featurized, self.ID, self.smiles_col, Molecule_col="Molecule", type_fpt='ph4_gobbi', save_dir=self.save_dir)
        ph4_0_df = cal_ph4.fit()
        ph4_0_df.columns = ph4_0_df.columns.astype(str)
        ph4_id_smiles = ph4_0_df[[self.ID, self.smiles_col]]
        ph4_good_features = np.loadtxt('./Utility/ph4_feature.txt', dtype='str')
        ph4_processed = pd.concat([ph4_id_smiles, ph4_0_df[ph4_good_features]], axis=1)
        progress_bar.progress(5 * progress_step)
        status_text.text("Pharmacophore features calculated...")

        #print("Mordred descriptors calculation...")
        mordred_cal = representation_calculation(data_featurized, self.ID, self.smiles_col, Molecule_col="Molecule", type_fpt='mordred', save_dir=self.save_dir)
        mordred_df = mordred_cal.fit()
        progress_bar.progress(6 * progress_step)
        status_text.text("Mordred descriptors calculated...")

        # Graph
        #print("Graph dataset creation...")
        num_cores = os.cpu_count()
        if int(self.n_jobs) <= num_cores:
            num_threads = int(self.n_jobs)
        else:
            num_threads = -1
        graph_dataset = BagMoleculeDataset(root=f"{self.save_dir}graphdata/",
                                        filename="cdr1_screen.csv", valid=True,
                                        data=data_featurized, smiles_col=self.smiles_col, activity_col=self.active_col,
                                        cut_off=3.5, num_confs=50, seed=42, rmsd=0.5,
                                        energy=10, max_attemp=1000, num_threads=num_threads)
        progress_bar.progress(1.0)
        status_text.text("Graph dataset created...")

        return rdk5_df, rdk6_df, rdk7_df, avalon_df, ph4_processed, mordred_df, graph_dataset
    
    def run(self):
        data_std = self.standardized_data()
        rdk5_df, rdk6_df, rdk7_df, avalon_df, ph4_processed, mordred_df, graph_dataset = self.featurized_data(data_featurized = data_std)
        return rdk5_df, rdk6_df, rdk7_df, avalon_df, ph4_processed, mordred_df, graph_dataset
        