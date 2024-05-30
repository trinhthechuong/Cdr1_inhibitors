from MIL_3D_GNN import instance_GNN
import sys
sys.path.append("..")
from conformation_encode.GraphDataset import BagMoleculeDataset
import torch
import os
import random
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, average_precision_score, balanced_accuracy_score
from tqdm import tqdm
from torch_geometric.data import DataLoader
import warnings
warnings.filterwarnings('ignore')
import mlflow

class run_experiment():
    def __init__(self,model, optimizer, train_loader, valid_loader, test_loader, num_epochs, seed, device, scheduler = None):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.device = device
        self.seed = seed
        self.scheduler = scheduler

    
    def save_model(self, epoch, exp_name):
        folder = f'./{exp_name}/'
        os.makedirs(folder, exist_ok=True)
        save_path = f'./{exp_name}/GNN_{epoch}.pth'
        torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                }, save_path)
        
    def seed_everything(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def generate_weights(self, targets):
        if sum(targets) == 0 or sum(targets) == len(targets):
            return torch.ones(len(targets))
        else:
            n_inactive = len(targets) - sum(targets)
            weights_active = len(targets)/(2*sum(targets))
            weights_inactive = len(targets)/(2*n_inactive)
            
            weights_value = weights_active/weights_inactive

            weights = torch.ones(len(targets))  # Initialize weights with ones

            # Update weights based on the target values
            weights[targets == 1] = weights_value
        return weights    
    
    def train(self, epoch):
        self.seed_everything(self.seed)
        self.model.train()
        loss_all = 0
        loss_func = torch.nn.BCELoss
        all_labels = []
        all_probas = []
        all_preds = []
        step = 0
        
        for data in tqdm(self.train_loader):
            data = data.to(self.device)
            y_true = data.y
            weights = self.generate_weights(y_true)
            self.optimizer.zero_grad()
            output = self.model(data).to(self.device)
            loss = loss_func(weight = weights)(output, data.y.float())
            loss.backward()
            loss_all += loss.item()
            self.optimizer.step()
            step += 1
            
            all_probas.append(output.cpu().detach().numpy())
            all_preds.append(np.rint(output.cpu().detach().numpy()))
            all_labels.append(data.y.cpu().detach().numpy())

        all_probas = np.concatenate(all_probas).ravel()
        all_preds = np.concatenate(all_preds).ravel()
        all_labels = np.concatenate(all_labels).ravel()

        loss_train = loss_all/len(self.train_loader)
        conf_m = confusion_matrix(all_labels, all_preds)
        specificity = conf_m[0,0]/(conf_m[0,0]+conf_m[0,1])
        recall = conf_m[1,1]/(conf_m[1,0]+conf_m[1,1])
        ba = balanced_accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        ap = average_precision_score(all_labels, all_probas)

        return loss_train, f1, ap, ba, recall, specificity
    
    def eval(self, loader, epoch):
        self.seed_everything(self.seed)
        loss_func = torch.nn.BCELoss
        self.model.eval()
        validation_loss = 0
        all_labels = []
        all_probas = []
        all_preds = []
        for data in tqdm(loader):
            data = data.to(self.device)
            y_true = data.y
            weights = self.generate_weights(y_true).to(self.device)
            with torch.no_grad():
                output = self.model(data)
                loss = loss_func(weight = weights)(output, data.y.float())
                validation_loss += loss.item()
                all_probas.append(output.cpu().detach().numpy())
                all_preds.append(np.rint(output.cpu().detach().numpy()))
                all_labels.append(data.y.cpu().detach().numpy())
           
        all_probas = np.concatenate(all_probas).ravel()
        all_preds = np.concatenate(all_preds).ravel()
        all_labels = np.concatenate(all_labels).ravel()

        loss_val = validation_loss /len(loader)
        f1 = f1_score(all_labels, all_preds)
        conf_m = confusion_matrix(all_labels, all_preds)
        specificity = conf_m[0,0]/(conf_m[0,0]+conf_m[0,1])
        recall = conf_m[1,1]/(conf_m[1,0]+conf_m[1,1])
        ba = balanced_accuracy_score(all_labels, all_preds)
        ap = average_precision_score(all_labels, all_probas)
        return loss_val, f1, ap, ba, recall, specificity
    
    def run(self, experiment_name, id_trial):
        #early_stopping = EarlyStopping(tolerance=5, min_delta=1e-4)
        self.seed_everything(self.seed)
        self.model.to(self.device)
        best_val_loss = None
        with mlflow.start_run():
            val_loss_0 = 0
            for epoch in range(self.num_epochs ):
                try:
                    lr = self.optimizer.param_groups[0]['lr']
                    print(f'Epoch: {epoch}, Lr: {lr}')
                    loss_train, f1_train, ap_train, ba_train, recall_train, speci_train = self.train(epoch)
                    mlflow.log_metric(key="f1-train", value=float(f1_train), step=epoch)
                    mlflow.log_metric(key="AP-train", value=float(ap_train), step=epoch)
                    mlflow.log_metric(key="Train loss", value=float(loss_train), step=epoch)
                    mlflow.log_metric(key="BA-train", value=float(ba_train), step=epoch)
                    mlflow.log_metric(key="Recall-train", value=float(recall_train), step=epoch)
                    mlflow.log_metric(key="Specificity-train", value=float(speci_train), step=epoch)
                    loss_val, f1_val, ap_val, ba_val, recall_val, speci_val = self.eval(self.valid_loader, epoch)
                    delta = loss_val - val_loss_0
                    val_loss_0 = loss_val
                    mlflow.log_metric(key="f1-val", value=float(f1_val), step=epoch)
                    mlflow.log_metric(key="AP-val", value=float(ap_val), step=epoch)
                    mlflow.log_metric(key="Val loss", value=float(loss_val), step=epoch)
                    mlflow.log_metric(key="BA-val", value=float(ba_val), step=epoch)
                    mlflow.log_metric(key="Recall-val", value=float(recall_val), step=epoch)
                    mlflow.log_metric(key="Specificity-val", value=float(speci_val), step=epoch)
                    loss_test, f1_test, ap_test, ba_test, recall_test, speci_test = self.eval(self.test_loader, epoch)
                    mlflow.log_metric(key="f1-test", value=float(f1_test), step=epoch)
                    mlflow.log_metric(key="AP-test", value=float(ap_test), step=epoch)
                    mlflow.log_metric(key="Test loss", value=float(loss_test), step=epoch)
                    mlflow.log_metric(key="BA-test", value=float(ba_test), step=epoch)
                    mlflow.log_metric(key="Recall-test", value=float(recall_test), step=epoch)
                    mlflow.log_metric(key="Specificity-test", value=float(speci_test), step=epoch)
                    print('Epoch: {:03d}, Train Loss: {:.7f}, Train F1: {:.7f}, Train AP: {:.7f}, Train BA: {:.7f}, Train Recall: {:.7f}, Train Specificity: {:.7f} \
                        Test Loss: {:.7f}, Test F1: {:.7f}, Test AP: {:.7f}, Test BA: {:.7f}, Test Recall {:.7f}, Test Specificity: {:.7f}\
                            Val Loss: {:.7f}, Val F1: {:.7f}, Val AP: {:.7f}, Val BA: {:.7f}, Val Recall {:.7f}, Val Specificity: {:.7f}'.format(epoch, loss_train, f1_train, ap_train, ba_train,
                                                                                                recall_train, speci_train,
                                                                                                loss_test, f1_test, ap_test, ba_test, recall_test, speci_test,
                                                                                                loss_val, f1_val, ap_val, ba_val, recall_val, speci_val))
                    if self.scheduler is not None:
                        self.scheduler.step()
                    if best_val_loss is None or loss_val < best_val_loss:
                        loss_test_best, f1_test_best, ap_test_best, ba_test_best, recall_test_best, speci_test_best = self.eval(self.test_loader, epoch)
                        self.save_model(epoch, experiment_name)
                        print("Saving.....")
                        best_val_loss = loss_val
                except Exception as e:
                    print("Gradients exploded")
                    pass    