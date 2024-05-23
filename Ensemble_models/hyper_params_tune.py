import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score, average_precision_score, roc_auc_score
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
import optuna 
from optuna.samplers import TPESampler

data_train = pd.read_csv("/Users/thechuongtrinh/Documents/Workspace/Master_thesis/Cdr1_inhibitors/dataset/Ensemble_model/validation_stacking/ml_gnn/data_train_ensemble")
data_test = pd.read_csv("/Users/thechuongtrinh/Documents/Workspace/Master_thesis/Cdr1_inhibitors/dataset/Ensemble_model/validation_stacking/ml_gnn/CatB_stacking_test.csv")
data_val = pd.read_csv("/Users/thechuongtrinh/Documents/Workspace/Master_thesis/Cdr1_inhibitors/dataset/Ensemble_model/validation_stacking/ml_gnn/CatB_stacking_val.csv")
data_hard_test = pd.read_csv("/Users/thechuongtrinh/Documents/Workspace/Master_thesis/Cdr1_inhibitors/dataset/Ensemble_model/validation_stacking/ml_gnn/CatB_stacking_hard_test.csv")

X_train = data_train.drop(columns=["ID","Standardize_smile","Activity"])
y_train = data_train["Activity"]
X_test = data_test.drop(columns=["ID","Standardize_smile","Activity","Predicted","Proba"])
y_test = data_test["Activity"]
X_val = data_val.drop(columns=["ID","Standardize_smile","Activity","Predicted","Proba"])
y_val = data_val["Activity"]
X_hard_test = data_hard_test.drop(columns=["ID","Standardize_smile","Activity","Predicted","Proba"])
y_hard_test = data_hard_test["Activity"]

##############################################
#    Define the objective function           #
##############################################

def objective(trial):
    searching_params = {"C" : trial.suggest_float("C", 1e-5, 1e5, log = True),
                        "gamma" : trial.suggest_categorical("gamma",["scale", "auto"]),
                        "class_weight" : trial.suggest_categorical("class_weight", ["balanced", None]),
                        "iterations":trial.suggest_int("iterations", 100, 1000),
                        "learning_rate":trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
                        "depth":trial.suggest_int("depth", 4, 10),
                        "l2_leaf_reg":trial.suggest_float("l2_leaf_reg", 1e-8, 100.0, log=True),
                        "alpha":trial.suggest_float("alpha", 1e-4, 1e-1, log=True),
                        "hidden_layer_sizes":trial.suggest_int("hidden_layer_sizes", 100, 300),
                       
    }
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
    y_proba_voting_test = voting_clf.predict_proba(X_test)[:,1]
    y_proba_voting_hard_test = voting_clf.predict_proba(X_hard_test)[:,1]

    ap_val = average_precision_score(y_val, y_proba_voting_val)


    ap_test = average_precision_score(y_test, y_proba_voting_test)
    f1_score_test = f1_score(y_test, (y_proba_voting_test >= 0.5).astype(int))
    roc_auc_test = roc_auc_score(y_test, y_proba_voting_test)
    balanced_accuracy_test = balanced_accuracy_score(y_test, (y_proba_voting_test >= 0.5).astype(int))
    cm_hard_test = confusion_matrix(y_hard_test, (y_proba_voting_hard_test >= 0.5).astype(int))
    #cal fpr
    try:
        tn, fp, _, _ = cm_hard_test.ravel()
        ht_fpr = fp / (fp + tn)
    except:
        ht_fpr = 0
    print(f"Test Results for Voting Model: Average Precision: {ap_test:.3f}, F1 Score: {f1_score_test:.3f}, ROC_AUC: {roc_auc_test:.3f}, BaAcc: {balanced_accuracy_test:.3f}, HT-FPR: {ht_fpr:.3f}")
    return ap_val

sampler = TPESampler(seed=42)
study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=100)
print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# with open('tunning_stacking_model.pkl', 'wb') as f:
#     pickle.dump(study, f)

