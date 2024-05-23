import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score, average_precision_score, recall_score, roc_auc_score
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
import json

data_train = pd.read_csv("/Users/thechuongtrinh/Documents/Workspace/Master_thesis/Cdr1_inhibitors/dataset/Ensemble_model/validation_stacking/ml_gnn/data_train_ensemble")
data_test = pd.read_csv("/Users/thechuongtrinh/Documents/Workspace/Master_thesis/Cdr1_inhibitors/dataset/Ensemble_model/validation_stacking/ml_gnn/CatB_stacking_test.csv")
data_test = data_test.drop(columns=["Predicted","Proba"], axis=1)
data_val = pd.read_csv("/Users/thechuongtrinh/Documents/Workspace/Master_thesis/Cdr1_inhibitors/dataset/Ensemble_model/validation_stacking/ml_gnn/CatB_stacking_val.csv")
data_val = data_val.drop(columns=["Predicted","Proba"], axis=1)
data_hard_test = pd.read_csv("/Users/thechuongtrinh/Documents/Workspace/Master_thesis/Cdr1_inhibitors/dataset/Ensemble_model/validation_stacking/ml_gnn/CatB_stacking_hard_test.csv")
data_hard_test = data_hard_test.drop(columns=["Predicted","Proba"], axis=1)

X_train = data_train.drop(columns=["ID","Standardize_smile","Activity"])

y_train = data_train["Activity"]
X_test = data_test.drop(columns=["ID","Standardize_smile","Activity"])

y_test = data_test["Activity"]
X_val = data_val.drop(columns=["ID","Standardize_smile","Activity"])

y_val = data_val["Activity"]
X_hard_test = data_hard_test.drop(columns=["ID","Standardize_smile","Activity"])
y_hard_test = data_hard_test["Activity"]

with open("/Users/thechuongtrinh/Documents/Workspace/Master_thesis/Cdr1_inhibitors/Ensemble_models/soft_voting_hyperparams.json") as file:
    searching_params = json.load(file)

base_model_1 = SVC(probability = True, max_iter = 10000,C = searching_params["C"],gamma = searching_params["gamma"], class_weight = searching_params["class_weight"])
base_model_2 = CatBoostClassifier(random_state=42,verbose=False,
                                    iterations = searching_params["iterations"],
                                    learning_rate = searching_params["learning_rate"],
                                    depth = searching_params["depth"],
                                    l2_leaf_reg = searching_params["l2_leaf_reg"],
                                    auto_class_weights = "Balanced")
base_model_3 = MLPClassifier(max_iter = 10000, validation_fraction = 0.1, random_state = 42,alpha=searching_params["alpha"], hidden_layer_sizes = searching_params["hidden_layer_sizes"])
classifiers = [('svc', base_model_1), ('catboost', base_model_2), ('mlp', base_model_3)]
voting_clf = VotingClassifier(estimators=classifiers, voting='soft')
voting_clf.fit(X_train, y_train)
y_proba_voting_val = voting_clf.predict_proba(X_val)[:,1]
y_pred_voting_val = voting_clf.predict(X_val)

data_val.insert(3, "Predicted", y_pred_voting_val)
data_val.insert(4, "Proba", y_proba_voting_val)


y_proba_voting_test = voting_clf.predict_proba(X_test)[:,1]
y_pred_voting_test = voting_clf.predict(X_test)
data_test.insert(3, "Predicted", y_pred_voting_test)
data_test.insert(4, "Proba", y_proba_voting_test)

y_proba_voting_hard_test = voting_clf.predict_proba(X_hard_test)[:,1]
y_pred_voting_hard_test = voting_clf.predict(X_hard_test)
data_hard_test.insert(3, "Predicted", y_pred_voting_hard_test)
data_hard_test.insert(4, "Proba", y_proba_voting_hard_test)

ap_val = average_precision_score(y_val, y_proba_voting_val)
f1_score_val = f1_score(y_val, (y_proba_voting_val >= 0.5).astype(int))
roc_auc_val = roc_auc_score(y_val, y_proba_voting_val)
balanced_accuracy_val = balanced_accuracy_score(y_val, (y_proba_voting_val >= 0.5).astype(int))

ap_test = average_precision_score(y_test, y_proba_voting_test)
f1_score_test = f1_score(y_test, (y_proba_voting_test >= 0.5).astype(int))
roc_auc_test = roc_auc_score(y_test, y_proba_voting_test)
balanced_accuracy_test = balanced_accuracy_score(y_test, (y_proba_voting_test >= 0.5).astype(int))

cm_hard_test = confusion_matrix(y_hard_test, (y_proba_voting_hard_test >= 0.5).astype(int))
#cal fpr
try:
    tn, fp, fn, tp = cm_hard_test.ravel()
    ht_fpr = fp / (fp + tn)
except:
    ht_fpr = 0
soft_voting_prefix = "/Users/thechuongtrinh/Documents/Workspace/Master_thesis/Cdr1_inhibitors/Ensemble_models"
with open(soft_voting_prefix+"/soft_voting.pkl", "wb") as f:
    pickle.dump(voting_clf, f)

print(f"Test Results for Voting Model: Average Precision: {ap_test:.3f}, F1 Score: {f1_score_test:.3f}, ROC AUC: {roc_auc_test:.3f}, Balanced Accuracy: {balanced_accuracy_test:.3f}, HT_FPR: {ht_fpr:.3f}")
print(f"Validation Results for Voting Model: Average Precision: {ap_val:.3f}, F1 Score: {f1_score_val:.3f}, ROC AUC: {roc_auc_val:.3f}, Balanced Accuracy: {balanced_accuracy_val:.3f}")
data_test.to_csv("/Users/thechuongtrinh/Documents/Workspace/Master_thesis/Cdr1_inhibitors/dataset/Ensemble_model/validation_soft_voting/soft_voting_external_test.csv", index=False)
data_val.to_csv("/Users/thechuongtrinh/Documents/Workspace/Master_thesis/Cdr1_inhibitors/dataset/Ensemble_model/validation_soft_voting/soft_voting_val.csv", index=False)
data_hard_test.to_csv("/Users/thechuongtrinh/Documents/Workspace/Master_thesis/Cdr1_inhibitors/dataset/Ensemble_model/validation_soft_voting/soft_voting_hard_test.csv", index=False)