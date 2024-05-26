import pandas as pd
import numpy as np
import torch
from torch_geometric.loader import DataLoader
import random
import json
import torch 
import sys
sys.path.append("..")
from MIL_3D_GNN.MIL_3D_GNN import instance_GNN
import pickle
import streamlit as st

class gnn_predict_st():
    def __init__(self, model_path, params_path, dataset, seed):
        self.loader = DataLoader(dataset, batch_size=1, shuffle=False)
        self.seed = seed
        self.model_path = model_path
        self.params_path = params_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def seed_everything(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def predict(self):
        self.seed_everything()
        tuned_params = json.load(open(self.params_path, "r"))
        const_config = {"node_dim_2d": 34,
                        "node_dim_3d": 18,
                        "edge_dim_2d": 11,
                        "edge_dim_3d": 9,
                        "mlp_layers_instance": 1,
                        "n_block": 3,
                        "pooling_every_n": 1}
        config = {**const_config, **tuned_params}
        model = instance_GNN(config)
        checkpoint = torch.load(self.model_path, map_location="cpu")
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        all_labels = []
        all_probas = []
        all_preds = []

        # Initialize Streamlit progress bar
        progress_bar = st.progress(0)
        progress_text = st.empty()
        total = len(self.loader)

        for idx, data in enumerate(self.loader):
            data = data.to(self.device)
            with torch.no_grad():
                output = model(data)
                all_probas.append(output.cpu().detach().numpy())
                all_preds.append(np.rint(output.cpu().detach().numpy()))
                all_labels.append(data.y.cpu().detach().numpy())
            
            # Update progress
            progress_percentage = (idx + 1) / total
            progress_bar.progress(progress_percentage)
            progress_text.text(f"Processing 3D Graph neural network: {idx + 1}/{total}")

        all_probas = np.concatenate(all_probas).ravel()
        all_preds = np.concatenate(all_preds).ravel()
        all_labels = np.concatenate(all_labels).ravel()
        return all_probas, all_preds



class ensemble_predict():
    def __init__(self, fpt_datasets, graph_dataset, ID_col, SMILES_col):
        self.datasets = fpt_datasets
        self.ID_col = ID_col
        self.graph_dataset = graph_dataset
        self.SMILES_col = SMILES_col

    def process(self):
        X_s = []
        for df in self.datasets:
            X_s.append(df.drop([self.ID_col, self.SMILES_col, "Activity"], axis=1))
        return X_s

    def predict(self):
        base_1 = pickle.load(open("./Ensemble_models/rdk5_catboost.pkl", "rb"))
        base_2 = pickle.load(open("./Ensemble_models/rdk6_cxgb.pkl", "rb"))
        base_3 = pickle.load(open("./Ensemble_models/rdk7_logistic.pkl", "rb"))
        base_4 = pickle.load(open("./Ensemble_models/mordred_catboost.pkl", "rb"))
        base_5 = pickle.load(open("./Ensemble_models/avalon_catboost.pkl", "rb"))
        base_6 = pickle.load(open("./Ensemble_models/ph4_mlp.pkl", "rb"))
        base_models = [base_1, base_2, base_3, base_4, base_5, base_6]
        voting_model = pickle.load(open("./Ensemble_models/soft_voting.pkl", "rb"))
        dict_proba_base_models = {}
        X_s = self.process()

        # Initialize Streamlit progress bar
        progress_bar = st.progress(0)
        progress_text = st.empty()
        total = len(base_models) + 1  # +1 for GNN model

        for i in range(len(base_models)):
            y_proba = base_models[i].predict_proba(X_s[i].values)[:, 1]
            dict_proba_base_models[f"model_{i}"] = y_proba

            # Update progress
            progress_percentage = (i + 1) / total
            progress_bar.progress(progress_percentage)
            progress_text.text(f"Processing machine learning model {i + 1}/{total - 1}")

        model_path = "./MIL_3D_GNN/mil_3d_gnn.pth"
        params_path = "./MIL_3D_GNN/config.json"
        gnn_results = gnn_predict_st(model_path, params_path, self.graph_dataset, seed=42)
        self.gnn_probas, _ = gnn_results.predict()
        dict_proba_base_models[f"GNN"] = self.gnn_probas
        self.df_proba_base_models = pd.DataFrame(dict_proba_base_models)
        self.df_proba_base_models.columns = ["rdk5", "rdk6", "rdk7", "mordred", "avalon", "ph4", "gnn"]

        y_pred_meta_model = voting_model.predict(self.df_proba_base_models)
        y_proba_meta_model = voting_model.predict_proba(self.df_proba_base_models)[:, 1]
        df_predictions = pd.DataFrame({"ID": self.datasets[0][self.ID_col], "Standardized SMILES": self.datasets[0][self.SMILES_col], "Probability": y_proba_meta_model, "Prediction": y_pred_meta_model})
        
        # Update progress for final step
        progress_bar.progress(1.0)
        progress_text.text("Completed")

        return df_predictions

