import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score, average_precision_score, recall_score, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import pickle


#Loading datasets
data_prefix = "/Users/thechuongtrinh/Documents/Workspace/Master_thesis/Cdr1_inhibitors/dataset/Featurized_data/BM_stratified_sampling/"
#RDK5 fingerprints
rdk5_train = pd.read_csv(data_prefix + "training_set/rdk5_train.csv")
rdk5_test = pd.read_csv(data_prefix + "external_test_set/rdk5_test.csv")
rdk5_valid = pd.read_csv(data_prefix + "validation_set_dl/rdk5_valid.csv")
rdk5_hard_test = pd.read_csv(data_prefix + "hard_test_set/rdk5_hard_test.csv")
#RDK6 fingerprints
rdk6_train = pd.read_csv(data_prefix + "training_set/rdk6_train.csv")
rdk6_test = pd.read_csv(data_prefix + "external_test_set/rdk6_test.csv")
rdk6_valid = pd.read_csv(data_prefix + "validation_set_dl/rdk6_valid.csv")
rdk6_hard_test = pd.read_csv(data_prefix + "hard_test_set/rdk6_hard_test.csv")
#RDK7 fingerprints
rdk7_train = pd.read_csv(data_prefix + "training_set/rdk7_train.csv")
rdk7_test = pd.read_csv(data_prefix + "external_test_set/rdk7_test.csv")
rdk7_valid = pd.read_csv(data_prefix + "validation_set_dl/rdk7_valid.csv")
rdk7_hard_test = pd.read_csv(data_prefix + "hard_test_set/rdk7_hard_test.csv")
#Mordred descriptors
mordred_train = pd.read_csv(data_prefix + "training_set/mordred_train.csv")
mordred_test = pd.read_csv(data_prefix + "external_test_set/mordred_test.csv")
mordred_valid = pd.read_csv(data_prefix + "validation_set_dl/mordred_valid.csv")
mordred_hard_test = pd.read_csv(data_prefix + "hard_test_set/mordred_hard_test.csv")
#Avalon fingerprints
avalon_train = pd.read_csv(data_prefix + "training_set/avalon_train.csv")
avalon_test = pd.read_csv(data_prefix + "external_test_set/avalon_test.csv")
avalon_valid = pd.read_csv(data_prefix + "validation_set_dl/avalon_valid.csv")
avalon_hard_test = pd.read_csv(data_prefix + "hard_test_set/avalon_hard_test.csv")
#PH4 fingerprints
ph4_train = pd.read_csv(data_prefix + "training_set/ph4_train.csv")
ph4_test = pd.read_csv(data_prefix + "external_test_set/ph4_test.csv")
ph4_valid = pd.read_csv(data_prefix + "validation_set_dl/ph4_valid.csv")
ph4_hard_test = pd.read_csv(data_prefix + "hard_test_set/ph4_hard_test.csv")

# Train the base models
base_models_prefix = "/Users/thechuongtrinh/Documents/Workspace/Master_thesis/Cdr1_inhibitors/Ensemble_models"
clf_rdk5 = CatBoostClassifier(verbose = 0, random_state = 42)
clf_rdk5.fit(rdk5_train.drop(["ID","Standardize_smile", "Activity"], axis = 1), rdk5_train["Activity"])
with open(base_models_prefix+"/rdk5_catboost.pkl", "wb") as f:
    pickle.dump(clf_rdk5, f)
y_proba_rdk5 = clf_rdk5.predict_proba(rdk5_train.drop(["ID","Standardize_smile", "Activity"], axis = 1))[:,1]
y_proba_test_rdk5 = clf_rdk5.predict_proba(rdk5_test.drop(["ID","Standardize_smile", "Activity"], axis = 1))[:,1]
y_proba_valid_rdk5 = clf_rdk5.predict_proba(rdk5_valid.drop(["ID","Standardize_smile", "Activity"], axis = 1))[:,1]
y_proba_hard_test_rdk5 = clf_rdk5.predict_proba(rdk5_hard_test.drop(["ID","Standardize_smile", "Activity"], axis = 1))[:,1]
y_pred_hard_test_rdk5 = clf_rdk5.predict(rdk5_hard_test.drop(["ID","Standardize_smile", "Activity"], axis = 1))
cm_rdk5 = confusion_matrix(rdk5_hard_test["Activity"], y_pred_hard_test_rdk5)
tn, fp, fn, tp = cm_rdk5.ravel()
# Calculate the false positive rate
false_positive_rate_hard_test = fp / (fp + tn)
print("False positive rate of hard test set on RDK5: ", false_positive_rate_hard_test)
# ef = EF(rdk5_test["Activity"], y_proba_test_rdk5, 0.01)
# print("Enrichment factor of test set on RDK5: ", ef)
X_hard_test_rdk5 = pd.DataFrame(np.array([y_proba_hard_test_rdk5]).T, columns = ["rdk5"])
X_test_rdk5 = pd.DataFrame(np.array([y_proba_test_rdk5]).T, columns = ["rdk5"])
X_val_rdk5 = pd.DataFrame(np.array([y_proba_valid_rdk5]).T, columns = ["rdk5"])
X_rdk5_ensemble = pd.DataFrame(np.array([y_proba_rdk5]).T, columns = ["rdk5"])

clf_rdk6 = XGBClassifier(random_state = 42)
clf_rdk6.fit(rdk6_train.drop(["ID","Standardize_smile", "Activity"], axis = 1), rdk6_train["Activity"])
with open(base_models_prefix+"/rdk6_cxgb.pkl", "wb") as f:
    pickle.dump(clf_rdk6, f)
y_proba_rdk6 = clf_rdk6.predict_proba(rdk6_train.drop(["ID","Standardize_smile", "Activity"], axis = 1))[:,1]
y_proba_test_rdk6 = clf_rdk6.predict_proba(rdk6_test.drop(["ID","Standardize_smile", "Activity"], axis = 1))[:,1]
y_proba_valid_rdk6 = clf_rdk6.predict_proba(rdk6_valid.drop(["ID","Standardize_smile", "Activity"], axis = 1))[:,1]
y_proba_hard_test_rdk6 = clf_rdk6.predict_proba(rdk6_hard_test.drop(["ID","Standardize_smile", "Activity"], axis = 1))[:,1]
y_pred_hard_test_rdk6 = clf_rdk6.predict(rdk6_hard_test.drop(["ID","Standardize_smile", "Activity"], axis = 1))
cm_rdk6 = confusion_matrix(rdk6_hard_test["Activity"], y_pred_hard_test_rdk6)
tn, fp, fn, tp = cm_rdk6.ravel()
# Calculate the false positive rate
false_positive_rate_hard_test = fp / (fp + tn)
print("False positive rate of hard test set on RDK6: ", false_positive_rate_hard_test)
# ef = EF(rdk6_test["Activity"], y_proba_test_rdk6, 0.01)
# print("Enrichment factor of test set on RDK6: ", ef)
X_hard_test_rdk6 = pd.DataFrame(np.array([y_proba_hard_test_rdk6]).T, columns = ["rdk6"])
X_test_rdk6 = pd.DataFrame(np.array([y_proba_test_rdk6]).T, columns = ["rdk6"])
X_val_rdk6 = pd.DataFrame(np.array([y_proba_valid_rdk6]).T, columns = ["rdk6"])
X_rdk6_ensemble = pd.DataFrame(np.array([y_proba_rdk6]).T, columns = ["rdk6"])

clf_rdk7 = LogisticRegression(penalty = 'l2', max_iter = 100000)
clf_rdk7.fit(rdk7_train.drop(["ID","Standardize_smile", "Activity"], axis = 1), rdk7_train["Activity"])
with open(base_models_prefix+"/rdk7_logistic.pkl", "wb") as f:
    pickle.dump(clf_rdk7, f)
y_proba_rdk7 = clf_rdk7.predict_proba(rdk7_train.drop(["ID","Standardize_smile", "Activity"], axis = 1))[:,1]
y_proba_test_rdk7 = clf_rdk7.predict_proba(rdk7_test.drop(["ID","Standardize_smile", "Activity"], axis = 1))[:,1]
y_proba_valid_rdk7 = clf_rdk7.predict_proba(rdk7_valid.drop(["ID","Standardize_smile", "Activity"], axis = 1))[:,1]
y_proba_hard_test_rdk7 = clf_rdk7.predict_proba(rdk7_hard_test.drop(["ID","Standardize_smile", "Activity"], axis = 1))[:,1]
y_pred_hard_test_rdk7 = clf_rdk7.predict(rdk7_hard_test.drop(["ID","Standardize_smile", "Activity"], axis = 1))
cm_rdk7 = confusion_matrix(rdk7_hard_test["Activity"], y_pred_hard_test_rdk7)
tn, fp, fn, tp = cm_rdk7.ravel()
# Calculate the false positive rate
false_positive_rate_hard_test = fp / (fp + tn)
print("False positive rate of hard test set on RDK7: ", false_positive_rate_hard_test)
# ef = EF(rdk7_test["Activity"], y_proba_test_rdk7, 0.01)
# print("Enrichment factor of test set on RDK7: ", ef)
X_hard_test_rdk7 = pd.DataFrame(np.array([y_proba_hard_test_rdk7]).T, columns = ["rdk7"])
X_val_rdk7 = pd.DataFrame(np.array([y_proba_valid_rdk7]).T, columns = ["rdk7"])
X_test_rdk7 = pd.DataFrame(np.array([y_proba_test_rdk7]).T, columns = ["rdk7"])
X_rdk7_ensemble = pd.DataFrame(np.array([y_proba_rdk7]).T,  columns = ["rdk7"])

clf_mordred = CatBoostClassifier(verbose = 0, random_state = 42)
clf_mordred.fit(mordred_train.drop(["ID","Standardize_smile", "Activity"], axis = 1), mordred_train["Activity"])
with open(base_models_prefix+"/mordred_catboost.pkl", "wb") as f:
    pickle.dump(clf_mordred, f)
y_proba_mordred = clf_mordred.predict_proba(mordred_train.drop(["ID","Standardize_smile", "Activity"], axis = 1))[:,1]
y_proba_test_mordred = clf_mordred.predict_proba(mordred_test.drop(["ID","Standardize_smile", "Activity"], axis = 1))[:,1]
y_proba_valid_mordred = clf_mordred.predict_proba(mordred_valid.drop(["ID","Standardize_smile", "Activity"], axis = 1))[:,1]
y_proba_hard_test_mordred = clf_mordred.predict_proba(mordred_hard_test.drop(["ID","Standardize_smile", "Activity"], axis = 1))[:,1]
y_pred_hard_test_mordred = clf_mordred.predict(mordred_hard_test.drop(["ID","Standardize_smile", "Activity"], axis = 1))
cm_mordred = confusion_matrix(mordred_hard_test["Activity"], y_pred_hard_test_mordred)
tn, fp, fn, tp = cm_mordred.ravel()
# Calculate the false positive rate
false_positive_rate_hard_test = fp / (fp + tn)
print("False positive rate of hard test set on Mordred: ", false_positive_rate_hard_test)
# ef = EF(mordred_test["Activity"], y_proba_test_mordred, 0.01)
# print("Enrichment factor of test set on Mordred: ", ef)
X_hard_test_mordred = pd.DataFrame(np.array([y_proba_hard_test_mordred]).T, columns = ["mordred"])
X_val_mordred = pd.DataFrame(np.array([y_proba_valid_mordred]).T, columns = ["mordred"])
X_test_mordred = pd.DataFrame(np.array([y_proba_test_mordred]).T, columns = ["mordred"])
X_mordred_ensemble = pd.DataFrame(np.array([y_proba_mordred]).T, columns = ["mordred"])

clf_avalon = CatBoostClassifier(verbose = 0, random_state = 42)
clf_avalon.fit(avalon_train.drop(["ID","Standardize_smile", "Activity"], axis = 1), avalon_train["Activity"])
with open(base_models_prefix+"/avalon_catboost.pkl", "wb") as f:
    pickle.dump(clf_avalon, f)
y_proba_avalon = clf_avalon.predict_proba(avalon_train.drop(["ID","Standardize_smile", "Activity"], axis = 1))[:,1]
y_proba_test_avalon = clf_avalon.predict_proba(avalon_test.drop(["ID","Standardize_smile", "Activity"], axis = 1))[:,1]
y_proba_val_avalon = clf_avalon.predict_proba(avalon_valid.drop(["ID","Standardize_smile", "Activity"], axis = 1))[:,1]
y_proba_hard_test_avalon = clf_avalon.predict_proba(avalon_hard_test.drop(["ID","Standardize_smile", "Activity"], axis = 1))[:,1]
y_pred_hard_test_avalon = clf_avalon.predict(avalon_hard_test.drop(["ID","Standardize_smile", "Activity"], axis = 1))
cm_avalon = confusion_matrix(avalon_hard_test["Activity"], y_pred_hard_test_avalon)
tn, fp, fn, tp = cm_avalon.ravel()
# Calculate the false positive rate
false_positive_rate_hard_test = fp / (fp + tn)
print("False positive rate of hard test set on Avalon: ", false_positive_rate_hard_test)
# ef = EF(avalon_test["Activity"], y_proba_test_avalon, 0.01)
# print("Enrichment factor of test set on Avalon: ", ef)
X_hard_test_avalon = pd.DataFrame(np.array([y_proba_hard_test_avalon]).T, columns = ["avalon"])
X_val_avalon = pd.DataFrame(np.array([y_proba_val_avalon]).T, columns = ["avalon"])
X_test_avalon = pd.DataFrame(np.array([y_proba_test_avalon]).T, columns = ["avalon"])
X_avalon_ensemble = pd.DataFrame(np.array([y_proba_avalon]).T,  columns = ["avalon"])

clf_ph4 = MLPClassifier(alpha = 0.01, max_iter = 10000, validation_fraction = 0.1, random_state = 42, hidden_layer_sizes = 150)
clf_ph4.fit(ph4_train.drop(["ID","Standardize_smile", "Activity"], axis = 1), ph4_train["Activity"])
with open(base_models_prefix+"/ph4_mlp.pkl", "wb") as f:
    pickle.dump(clf_ph4, f)
y_proba_ph4 = clf_ph4.predict_proba(ph4_train.drop(["ID","Standardize_smile", "Activity"], axis = 1))[:,1]
y_proba_test_ph4 = clf_ph4.predict_proba(ph4_test.drop(["ID","Standardize_smile", "Activity"], axis = 1))[:,1]
y_proba_val_ph4 = clf_ph4.predict_proba(ph4_valid.drop(["ID","Standardize_smile", "Activity"], axis = 1))[:,1]
y_proba_hard_test_ph4 = clf_ph4.predict_proba(ph4_hard_test.drop(["ID","Standardize_smile", "Activity"], axis = 1))[:,1]
y_pred_hard_test_ph4 = clf_ph4.predict(ph4_hard_test.drop(["ID","Standardize_smile", "Activity"], axis = 1))
cm_ph4 = confusion_matrix(ph4_hard_test["Activity"], y_pred_hard_test_ph4)
tn, fp, fn, tp = cm_ph4.ravel()
# Calculate the false positive rate
false_positive_rate_hard_test = fp / (fp + tn)
print("False positive rate of hard test set on PH4: ", false_positive_rate_hard_test)
# ef = EF(ph4_test["Activity"], y_proba_test_ph4, 0.01)
# print("Enrichment factor of test set on PH4: ", ef)
X_hard_test_ph4 = pd.DataFrame(np.array([y_proba_hard_test_ph4]).T, columns = ["ph4"])
X_val_ph4 = pd.DataFrame(np.array([y_proba_val_ph4]).T, columns = ["ph4"])
X_test_ph4 = pd.DataFrame(np.array([y_proba_test_ph4]).T, columns = ["ph4"])
X_ph4_ensemble = pd.DataFrame(np.array([y_proba_ph4]).T, columns = ["ph4"])

#Loading probas GNN
gnn_proba = np.loadtxt("/Users/thechuongtrinh/Documents/Workspace/Master_thesis/Cdr1_inhibitors/Ensemble_models/gnn_probas_no_sampling.txt")
gnn_proba_test = np.loadtxt("/Users/thechuongtrinh/Documents/Workspace/Master_thesis/Cdr1_inhibitors/Ensemble_models/gnn_probas_test.txt")
gnn_proba_val = np.loadtxt("/Users/thechuongtrinh/Documents/Workspace/Master_thesis/Cdr1_inhibitors/Ensemble_models/gnn_probas_valid.txt")
gnn_proba_hard_test = np.loadtxt("/Users/thechuongtrinh/Documents/Workspace/Master_thesis/Cdr1_inhibitors/Ensemble_models/gnn_probas_hard_test.txt")
cm_gnn = confusion_matrix(rdk5_hard_test["Activity"], (gnn_proba_hard_test >= 0.5).astype('int'))
tn, fp, fn, tp = cm_gnn.ravel()
# Calculate the false positive rate
false_positive_rate_hard_test = fp / (fp + tn)
print("False positive rate of hard test set on GNN: ", false_positive_rate_hard_test)
# ef = EF(rdk5_test["Activity"], gnn_proba_test, 0.01)
# print("Enrichment factor of test set on GNN: ", ef)
X_gnn = pd.DataFrame(np.array([gnn_proba]).T, columns = ["gnn"])
X_test_gnn = pd.DataFrame(np.array([gnn_proba_test]).T, columns = ["gnn"])
X_val_gnn = pd.DataFrame(np.array([gnn_proba_val]).T, columns = ["gnn"])
X_hard_test_gnn = pd.DataFrame(np.array([gnn_proba_hard_test]).T, columns = ["gnn"])

y_train_df = pd.DataFrame(rdk5_train["Activity"], columns=["Activity"])
y_test_df = pd.DataFrame(rdk5_test["Activity"], columns=["Activity"])
y_val_df = pd.DataFrame(rdk5_valid["Activity"], columns=["Activity"])
y_hard_test_df = pd.DataFrame(rdk5_hard_test["Activity"], columns=["Activity"])
X_ensemble = pd.concat([X_rdk5_ensemble, X_rdk6_ensemble, X_rdk7_ensemble, X_mordred_ensemble, X_avalon_ensemble, X_ph4_ensemble,X_gnn], axis = 1)
train_smiles_df = rdk5_train[["ID","Standardize_smile"]]
prefix_data_train = "/Users/thechuongtrinh/Documents/Workspace/Master_thesis/Cdr1_inhibitors/dataset/Ensemble_model/validation_stacking/ml_gnn/"
train_ensemble_df = pd.concat([train_smiles_df,y_train_df,X_ensemble], axis = 1).to_csv(prefix_data_train+"data_train_ensemble", index = False)

X_test = pd.concat([X_test_rdk5, X_test_rdk6, X_test_rdk7, X_test_mordred, X_test_avalon, X_test_ph4, X_test_gnn], axis = 1)
test_smiles_df = rdk5_test[["ID","Standardize_smile"]]
#test_ensemble_df = pd.concat([test_smiles_df, y_test_df, X_test], axis = 1)


X_val = pd.concat([X_val_rdk5, X_val_rdk6, X_val_rdk7, X_val_mordred, X_val_avalon, X_val_ph4, X_val_gnn], axis = 1)
val_smiles_df = rdk5_valid[["ID","Standardize_smile"]]
#val_ensemble_df = pd.concat([rdk5_valid,y_val_df,X_val], axis = 1)

X_hard_test = pd.concat([X_hard_test_rdk5, X_hard_test_rdk6, X_hard_test_rdk7, X_hard_test_mordred, X_hard_test_avalon, X_hard_test_ph4, X_hard_test_gnn], axis = 1)
hard_test_smiles_df = rdk5_hard_test[["ID","Standardize_smile"]]
#hard_test_ensemble_df = pd.concat([hard_test_smiles_df, y_hard_test_df, X_hard_test], axis = 1)

y_train = rdk5_train["Activity"]
y_test = rdk5_test["Activity"]
y_val = rdk5_valid["Activity"]
y_hard_test = rdk5_hard_test["Activity"]

# Train the ensemble model
def Classification():
        models, names = list(), list()
        #1. Logistics
        model = LogisticRegression(penalty = 'l2', max_iter = 100000)
        steps = [('m', model)]
        models.append(Pipeline(steps=steps))
        names.append('Logistic')
        
        #2 KNN
        model = KNeighborsClassifier(n_neighbors = 20)
        steps = [('m', model)]
        models.append(Pipeline(steps=steps))
        names.append('KNN')
        
        #3 svm
        model = SVC(probability = True, max_iter = 10000)
        steps = [('m', model)]
        models.append(Pipeline(steps=steps))
        names.append('SVM')
      
        #9 RF
        model = RandomForestClassifier(random_state = 42)
        steps = [('m', model)]
        models.append(Pipeline(steps=steps))
        names.append('RF')
        
        #10 ExT
        model = ExtraTreesClassifier(random_state = 42)
        steps = [('m', model)]
        models.append(Pipeline(steps=steps))
        names.append('ExT')
        
        #11 ADA
        model = AdaBoostClassifier(random_state = 42)
        steps = [('m', model)]
        models.append(Pipeline(steps=steps))
        names.append('Ada')
        
        #12 Grad
        model = GradientBoostingClassifier(random_state = 42)
        steps = [('m', model)]
        models.append(Pipeline(steps=steps))
        names.append('Grad')
     
        #13 XGB
        model = XGBClassifier(random_state = 42, verbosity=0, use_label_encoder=False, eval_metrics ='logloss')
        steps = [('m', model)]
        models.append(Pipeline(steps=steps))
        names.append('XGB')
        
        #14 Cat
        model = CatBoostClassifier(verbose = 0, random_state = 42)
        steps = [('m', model)]
        models.append(Pipeline(steps=steps))
        names.append('CatB')
        
        #15 MLP
        model = MLPClassifier(alpha = 0.01, max_iter = 10000, validation_fraction = 0.1, random_state = 42, hidden_layer_sizes = 150)
        steps = [('m', model)]
        models.append(Pipeline(steps=steps))
        names.append('MLP') 
        return models, names

models, names = Classification()

def external_validation(model, X_train, y_train, X_test, y_test, data_smiles):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    df_prediction = pd.concat([data_smiles, pd.DataFrame(y_test, columns = ["Activity"]), pd.DataFrame(y_pred, columns = ["Predicted"]), pd.DataFrame(y_prob, columns = ["Proba"]),X_test], axis = 1)
    f1 = f1_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)
    ap = average_precision_score(y_test, y_prob)
    roc = roc_auc_score(y_test, y_prob)
    return f1, bacc, ap, roc, df_prediction

def fprr(model, X_train,y_train, X_hard_test, y_hard_test, data_smiles):
     model.fit(X_train, y_train)
     try:
        y_pred = model.predict(X_hard_test)
        y_proba = model.predict_proba(X_hard_test)[:, 1]
        df_prediction = pd.concat([data_smiles, pd.DataFrame(y_hard_test, columns = ["Activity"]), pd.DataFrame(y_pred, columns = ["Predicted"]), pd.DataFrame(y_proba, columns = ["Proba"]),X_hard_test], axis = 1)
        cm = confusion_matrix(y_hard_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        return fp / (fp + tn), df_prediction
     except:
         return 0     
#evaluate the models and store results
print("*"*18)
print("*" +" "+ "STACKING MODEL"+" "+"*")
print("*"*18)
for i in range(len(models)):
    model = models[i]
    name = names[i]
    f1, bacc, ap, roc, df_prediction_val = external_validation(model, X_ensemble, y_train, X_val, y_val, val_smiles_df)
    print(f">  Validation Results for {names[i]}:Average Precision: {ap:.3f}, F1 Score: {f1:.3f}, ROC AUC : {roc:.3f}, Balanced Accuracy: {bacc:.3f}")
    df_prediction_val.to_csv(f"/Users/thechuongtrinh/Documents/Workspace/Master_thesis/Cdr1_inhibitors/dataset/Ensemble_model/validation_stacking/ml_gnn/{name}_stacking_val.csv", index = False)
    f1, bacc, ap, roc, df_prediction_test = external_validation(model, X_ensemble, y_train, X_test, y_test, test_smiles_df)
    print(f">  Test Results for {names[i]}: Average Precision: {ap:.3f}, F1 Score: {f1:.3f}, ROC AUC: {roc:.3f}, Balanced Accuracy: {bacc:.3f}")
    df_prediction_test.to_csv(f"/Users/thechuongtrinh/Documents/Workspace/Master_thesis/Cdr1_inhibitors/dataset/Ensemble_model/validation_stacking/ml_gnn/{name}_stacking_test.csv", index = False)
    fp,df_prediction_hard_test = fprr(model, X_ensemble, y_train, X_hard_test, y_hard_test, hard_test_smiles_df)
    df_prediction_hard_test.to_csv(f"/Users/thechuongtrinh/Documents/Workspace/Master_thesis/Cdr1_inhibitors/dataset/Ensemble_model/validation_stacking/ml_gnn/{name}_stacking_hard_test.csv", index = False)
    print(f">  False Positive Rate of Hard Test Set for {names[i]}: {fp:.4f}")