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
import matplotlib.pyplot as plt
import pickle

rdk5_train = pd.read_csv("./data/Official/Featurizer_data/rdk5_train.csv")
rdk5_test = pd.read_csv("./data/Official/Featurizer_data/rdk5_test.csv")
rdk5_valid = pd.read_csv("./data/Official/Featurizer_data/rdk5_valid.csv")
rdk6_train = pd.read_csv("./data/Official/Featurizer_data/rdk6_train.csv")
rdk6_test = pd.read_csv("./data/Official/Featurizer_data/rdk6_test.csv")
rdk6_valid = pd.read_csv("./data/Official/Featurizer_data/rdk6_valid.csv")
rdk7_train = pd.read_csv("./data/Official/Featurizer_data/rdk7_train.csv")
rdk7_test = pd.read_csv("./data/Official/Featurizer_data/rdk7_test.csv")
rdk7_valid = pd.read_csv("./data/Official/Featurizer_data/rdk7_valid.csv")
mordred_train = pd.read_csv("./data/Official/Featurizer_data/mordred_train.csv")
mordred_test = pd.read_csv("./data/Official/Featurizer_data/mordred_test.csv")
mordred_valid = pd.read_csv("./data/Official/Featurizer_data/mordred_valid.csv")
avalon_train = pd.read_csv("./data/Official/Featurizer_data/avalon_train.csv")
avalon_test = pd.read_csv("./data/Official/Featurizer_data/avalon_test.csv")
avalon_valid = pd.read_csv("./data/Official/Featurizer_data/avalon_valid.csv")
ph4_train = pd.read_csv("./data/Official/Featurizer_data/ph4_train.csv")
ph4_test = pd.read_csv("./data/Official/Featurizer_data/ph4_test.csv")
ph4_valid = pd.read_csv("./data/Official/Featurizer_data/ph4_valid.csv")

rdk5_hard_test = pd.read_csv("./data/Official/Featurizer_data/rdk5_hard_test.csv")
rdk6_hard_test = pd.read_csv("./data/Official/Featurizer_data/rdk6_hard_test.csv")
rdk7_hard_test = pd.read_csv("./data/Official/Featurizer_data/rdk7_hard_test.csv")
mordred_hard_test = pd.read_csv("./data/Official/Featurizer_data/mordred_hard_test.csv")
avalon_hard_test = pd.read_csv("./data/Official/Featurizer_data/avalon_hard_test.csv")
ph4_hard_test = pd.read_csv("./data/Official/Featurizer_data/ph4_hard_test.csv")


# Train the model
clf_rdk5 = CatBoostClassifier(verbose = 0, random_state = 42)
clf_rdk5.fit(rdk5_train.drop(["ID","Standardize_smile", "Activity"], axis = 1), rdk5_train["Activity"])
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
X_hard_test_rdk5 = pd.DataFrame(np.array([y_proba_hard_test_rdk5]).T, columns = ["rdk5"])
X_test_rdk5 = pd.DataFrame(np.array([y_proba_test_rdk5]).T, columns = ["rdk5"])
X_val_rdk5 = pd.DataFrame(np.array([y_proba_valid_rdk5]).T, columns = ["rdk5"])
X_rdk5_ensemble = pd.DataFrame(np.array([y_proba_rdk5]).T, columns = ["rdk5"])

clf_rdk6 = XGBClassifier(random_state = 42)
clf_rdk6.fit(rdk6_train.drop(["ID","Standardize_smile", "Activity"], axis = 1), rdk6_train["Activity"])
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
X_hard_test_rdk6 = pd.DataFrame(np.array([y_proba_hard_test_rdk6]).T, columns = ["rdk6"])
X_test_rdk6 = pd.DataFrame(np.array([y_proba_test_rdk6]).T, columns = ["rdk6"])
X_val_rdk6 = pd.DataFrame(np.array([y_proba_valid_rdk6]).T, columns = ["rdk6"])
X_rdk6_ensemble = pd.DataFrame(np.array([y_proba_rdk6]).T, columns = ["rdk6"])

clf_rdk7 = LogisticRegression(penalty = 'l2', max_iter = 100000)
clf_rdk7.fit(rdk7_train.drop(["ID","Standardize_smile", "Activity"], axis = 1), rdk7_train["Activity"])
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
X_hard_test_rdk7 = pd.DataFrame(np.array([y_proba_hard_test_rdk7]).T, columns = ["rdk7"])
X_val_rdk7 = pd.DataFrame(np.array([y_proba_valid_rdk7]).T, columns = ["rdk7"])
X_test_rdk7 = pd.DataFrame(np.array([y_proba_test_rdk7]).T, columns = ["rdk7"])
X_rdk7_ensemble = pd.DataFrame(np.array([y_proba_rdk7]).T,  columns = ["rdk7"])

clf_mordred = XGBClassifier(random_state = 42)
clf_mordred.fit(mordred_train.drop(["ID","Standardize_smile", "Activity"], axis = 1), mordred_train["Activity"])
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
X_hard_test_mordred = pd.DataFrame(np.array([y_proba_hard_test_mordred]).T, columns = ["mordred"])
X_val_mordred = pd.DataFrame(np.array([y_proba_valid_mordred]).T, columns = ["mordred"])
X_test_mordred = pd.DataFrame(np.array([y_proba_test_mordred]).T, columns = ["mordred"])
X_mordred_ensemble = pd.DataFrame(np.array([y_proba_mordred]).T, columns = ["mordred"])

clf_avalon = CatBoostClassifier(verbose = 0, random_state = 42)
clf_avalon.fit(avalon_train.drop(["ID","Standardize_smile", "Activity"], axis = 1), avalon_train["Activity"])
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
X_hard_test_avalon = pd.DataFrame(np.array([y_proba_hard_test_avalon]).T, columns = ["avalon"])
X_val_avalon = pd.DataFrame(np.array([y_proba_val_avalon]).T, columns = ["avalon"])
X_test_avalon = pd.DataFrame(np.array([y_proba_test_avalon]).T, columns = ["avalon"])
X_avalon_ensemble = pd.DataFrame(np.array([y_proba_avalon]).T,  columns = ["avalon"])

clf_ph4 = MLPClassifier(alpha = 0.01, max_iter = 10000, validation_fraction = 0.1, random_state = 42, hidden_layer_sizes = 150)
clf_ph4.fit(ph4_train.drop(["ID","Standardize_smile", "Activity"], axis = 1), ph4_train["Activity"])
y_proba_ph4 = clf_ph4.predict_proba(ph4_train.drop(["ID","Standardize_smile", "Activity"], axis = 1))[:,1]
y_proba_test_ph4 = clf_ph4.predict_proba(ph4_test.drop(["ID","Standardize_smile", "Activity"], axis = 1))[:,1]
y_proba_val_avalon = clf_ph4.predict_proba(ph4_valid.drop(["ID","Standardize_smile", "Activity"], axis = 1))[:,1]
y_proba_hard_test_ph4 = clf_ph4.predict_proba(ph4_hard_test.drop(["ID","Standardize_smile", "Activity"], axis = 1))[:,1]
y_pred_hard_test_ph4 = clf_ph4.predict(ph4_hard_test.drop(["ID","Standardize_smile", "Activity"], axis = 1))
cm_ph4 = confusion_matrix(ph4_hard_test["Activity"], y_pred_hard_test_ph4)
tn, fp, fn, tp = cm_ph4.ravel()
# Calculate the false positive rate
false_positive_rate_hard_test = fp / (fp + tn)
print("False positive rate of hard test set on PH4: ", false_positive_rate_hard_test)
X_hard_test_ph4 = pd.DataFrame(np.array([y_proba_hard_test_ph4]).T, columns = ["ph4"])
X_val_ph4 = pd.DataFrame(np.array([y_proba_val_avalon]).T, columns = ["ph4"])
X_test_ph4 = pd.DataFrame(np.array([y_proba_test_ph4]).T, columns = ["ph4"])
X_ph4_ensemble = pd.DataFrame(np.array([y_proba_ph4]).T, columns = ["ph4"])

#Loading probas GNN
gnn_proba = np.loadtxt("/Users/thechuongtrinh/Documents/Workspace/Master_thesis/Cdr1/gnn_probas_no_sampling.txt")
gnn_proba_test = np.loadtxt("/Users/thechuongtrinh/Documents/Workspace/Master_thesis/Cdr1/gnn_probas_test.txt")
gnn_proba_val = np.loadtxt("/Users/thechuongtrinh/Documents/Workspace/Master_thesis/Cdr1/gnn_probas_valid.txt")
gnn_proba_hard_test = np.loadtxt("/Users/thechuongtrinh/Documents/Workspace/Master_thesis/Cdr1/gnn_probas_hard_test.txt")
X_gnn = pd.DataFrame(np.array([gnn_proba]).T, columns = ["gnn"])
X_test_gnn = pd.DataFrame(np.array([gnn_proba_test]).T, columns = ["gnn"])
X_val_gnn = pd.DataFrame(np.array([gnn_proba_val]).T, columns = ["gnn"])
X_hard_test_gnn = pd.DataFrame(np.array([gnn_proba_hard_test]).T, columns = ["gnn"])
y_train_df = pd.DataFrame(rdk5_train["Activity"], columns=["Activity"])
y_test_df = pd.DataFrame(rdk5_test["Activity"], columns=["Activity"])
y_val_df = pd.DataFrame(rdk5_valid["Activity"], columns=["Activity"])
y_hard_test_df = pd.DataFrame(rdk5_hard_test["Activity"], columns=["Activity"])
X_ensemble = pd.concat([X_rdk5_ensemble, X_rdk6_ensemble, X_rdk7_ensemble, X_mordred_ensemble, X_avalon_ensemble, X_ph4_ensemble,X_gnn], axis = 1)
train_ensemble_df = pd.concat([X_ensemble, y_train_df], axis = 1).to_csv("./data/Official/Featurizer_data/train_ensemble.csv", index = False)
X_test = pd.concat([X_test_rdk5, X_test_rdk6, X_test_rdk7, X_test_mordred, X_test_avalon, X_test_ph4, X_test_gnn], axis = 1)
test_ensemble_df = pd.concat([X_test, y_test_df], axis = 1).to_csv("./data/Official/Featurizer_data/test_ensemble.csv", index = False)
X_val = pd.concat([X_val_rdk5, X_val_rdk6, X_val_rdk7, X_val_mordred, X_val_avalon, X_val_ph4, X_val_gnn], axis = 1)
val_ensemble_df = pd.concat([X_val, y_val_df], axis = 1).to_csv("./data/Official/Featurizer_data/val_ensemble.csv", index = False)
X_hard_test = pd.concat([X_hard_test_rdk5, X_hard_test_rdk6, X_hard_test_rdk7, X_hard_test_mordred, X_hard_test_avalon, X_hard_test_ph4, X_hard_test_gnn], axis = 1)
hard_test_ensemble_df = pd.concat([X_hard_test, y_hard_test_df], axis = 1).to_csv("./data/Official/Featurizer_data/hard_test_ensemble.csv", index = False)
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
def external_validation(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)
    ap = average_precision_score(y_test, y_prob)
    return f1, recall, bacc, ap
def fprr(model, X_train,y_train, X_hard_test, y_hard_test):
     model.fit(X_train, y_train)
     y_pred = model.predict(X_hard_test)
     cm = confusion_matrix(y_hard_test, y_pred)
     tn, fp, fn, tp = cm.ravel()
     return fp / (fp + tn)     
#evaluate the models and store results
for i in range(len(models)):
    model = models[i]
    name = names[i]
    f1, recall, bacc, ap = external_validation(model, X_ensemble, y_train, X_val, y_val)
    print(f">  Validation Results for {names[i]}: F1 Score: {f1:.3f}, Recall: {recall:.3f}, Balanced Accuracy: {bacc:.3f}, Average Precision: {ap:.3f}")
    f1, recall, bacc, ap = external_validation(model, X_ensemble, y_train, X_test, y_test)
    print(f">  Test Results for {names[i]}: F1 Score: {f1:.3f}, Recall: {recall:.3f}, Balanced Accuracy: {bacc:.3f}, Average Precision: {ap:.3f}")
    fp = fprr(model, X_ensemble, y_train, X_hard_test, y_hard_test)
    print(f">  False Positive Rate of Hard Test Set for {names[i]}: {fp:.4f}")

meta_model = MLPClassifier(alpha = 0.01, max_iter = 10000, validation_fraction = 0.1, random_state = 42, hidden_layer_sizes = 150)
# meta_model = CatBoostClassifier(verbose = 0, random_state = 42)
meta_model.fit(X_ensemble, y_train)
y_pred_val = meta_model.predict(X_val)
y_prob_val = meta_model.predict_proba(X_val)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, y_prob_val)
# # convert to f score
fscore = (2 * precision * recall) / (precision + recall)
# # locate the index of the largest f score
ix = np.argmax(fscore)
print('Best Threshold=%f, F-Score=%.3f for Validation' % (thresholds[ix], fscore[ix]))
print('Best Precision=%f, Recall=%.3f for Validation' % (precision[ix], recall[ix]))
bacc = balanced_accuracy_score(y_val, (y_prob_val >= thresholds[ix]).astype('int'))
print("Balanced Accuracy with best threshold for Validation: ", bacc)
selected_threshold = thresholds[ix]
f1_test = f1_score(y_test, (meta_model.predict_proba(X_test)[:, 1] >= selected_threshold).astype('int'))
recall_test = recall_score(y_test, (meta_model.predict_proba(X_test)[:, 1] >= selected_threshold).astype('int'))
bacc_test = balanced_accuracy_score(y_test, (meta_model.predict_proba(X_test)[:, 1] >= selected_threshold).astype('int'))
print("F1 Score with best threshold on external test set: ", f1_test)
print("Recall with best threshold on external test set: ", recall_test)
print("Balanced Accuracy with best threshold on external test set: ", bacc_test)


# Save the model
pickle.dump(clf_rdk5, open('./Models/catboost_rdk5.pkl', 'wb'))
pickle.dump(clf_rdk6, open('./Models/xgb_rdk6.pkl', 'wb'))
pickle.dump(clf_rdk7, open('./Models/logistic_rdk7.pkl', 'wb'))
pickle.dump(clf_mordred, open('./Models/xgb_mordred.pkl', 'wb'))
pickle.dump(clf_avalon, open('./Models/catboost_avalon.pkl', 'wb'))
pickle.dump(clf_ph4, open('./Models/mlp_ph4.pkl', 'wb'))
pickle.dump(meta_model, open('./Models/meta_model.pkl', 'wb'))
print("*"*100)

