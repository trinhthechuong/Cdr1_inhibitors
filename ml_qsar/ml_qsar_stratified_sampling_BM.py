import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, balanced_accuracy_score, recall_score, roc_auc_score, average_precision_score, confusion_matrix
import os
from tqdm import tqdm

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


def cv_evaluate(model, fold_1, fold_2, fold_3, fold_4, fold_5, drop_cols, target_col):
    f1_scores, bacc_scores, roc_auc_scores, aps = list(), list(), list(), list()
    #Fold 1 as validation
    data = pd.concat([fold_2, fold_3, fold_4, fold_5], axis = 0)
    X_train = data.drop(drop_cols, axis = 1)
    y_train = data[target_col]
    model.fit(X_train, y_train)
    y_pred = model.predict(fold_1.drop(drop_cols, axis = 1))
    y_prob = model.predict_proba(fold_1.drop(drop_cols, axis = 1))[:, 1]
    f1 = f1_score(fold_1[target_col], y_pred)
    bacc = balanced_accuracy_score(fold_1[target_col], y_pred)
    roc_auc = roc_auc_score(fold_1[target_col], y_prob)
    ap = average_precision_score(fold_1[target_col], y_prob)
    f1_scores.append(f1)
    bacc_scores.append(bacc)
    roc_auc_scores.append(roc_auc)
    aps.append(ap)
    #Fold 2 as validation
    data = pd.concat([fold_1, fold_3, fold_4, fold_5], axis = 0)
    X_train = data.drop(drop_cols, axis = 1)
    y_train = data[target_col]
    model.fit(X_train, y_train)
    y_pred = model.predict(fold_2.drop(drop_cols, axis = 1))
    y_prob = model.predict_proba(fold_2.drop(drop_cols, axis = 1))[:, 1]
    f1_2 = f1_score(fold_2[target_col], y_pred)
    bacc_2 = balanced_accuracy_score(fold_2[target_col], y_pred)
    roc_auc_2 = roc_auc_score(fold_2[target_col], y_prob)
    ap_2 = average_precision_score(fold_2[target_col], y_prob)
    f1_scores.append(f1_2)
    bacc_scores.append(bacc_2)
    roc_auc_scores.append(roc_auc_2)
    aps.append(ap_2)
    #Fold 3 as validation
    data = pd.concat([fold_1, fold_2, fold_4, fold_5], axis = 0)
    X_train = data.drop(drop_cols, axis = 1)
    y_train = data[target_col]
    model.fit(X_train, y_train)
    y_pred = model.predict(fold_3.drop(drop_cols, axis = 1))
    y_prob = model.predict_proba(fold_3.drop(drop_cols, axis = 1))[:, 1]
    f1_3 = f1_score(fold_3[target_col], y_pred)
    bacc_3 = balanced_accuracy_score(fold_3[target_col], y_pred)
    roc_auc_3 = roc_auc_score(fold_3[target_col], y_prob)
    ap_3 = average_precision_score(fold_3[target_col], y_prob)
    f1_scores.append(f1_3)
    bacc_scores.append(bacc_3)
    roc_auc_scores.append(roc_auc_3)
    aps.append(ap_3)
    #Fold 4 as validation
    data = pd.concat([fold_1, fold_2, fold_3, fold_5], axis = 0)
    X_train = data.drop(drop_cols, axis = 1)
    y_train = data[target_col]
    model.fit(X_train, y_train)
    y_pred = model.predict(fold_4.drop(drop_cols, axis = 1))
    y_prob = model.predict_proba(fold_4.drop(drop_cols, axis = 1))[:, 1]
    f1_4 = f1_score(fold_4[target_col], y_pred)
    bacc_4 = balanced_accuracy_score(fold_4[target_col], y_pred)
    roc_auc_4 = roc_auc_score(fold_4[target_col], y_prob)
    ap_4 = average_precision_score(fold_4[target_col], y_prob)
    f1_scores.append(f1_4)
    bacc_scores.append(bacc_4)
    roc_auc_scores.append(roc_auc_4)
    aps.append(ap_4)
    #Fold 5 as validation
    data = pd.concat([fold_1, fold_2, fold_3, fold_4], axis = 0)
    X_train = data.drop(drop_cols, axis = 1)
    y_train = data[target_col]
    model.fit(X_train, y_train)
    y_pred = model.predict(fold_5.drop(drop_cols, axis = 1))
    y_prob = model.predict_proba(fold_5.drop(drop_cols, axis = 1))[:, 1]
    f1_5 = f1_score(fold_5[target_col], y_pred)
    bacc_5 = balanced_accuracy_score(fold_5[target_col], y_pred)
    roc_auc_5 = roc_auc_score(fold_5[target_col], y_prob)
    ap_5 = average_precision_score(fold_5[target_col], y_prob)
    f1_scores.append(f1_5)
    bacc_scores.append(bacc_5)
    roc_auc_scores.append(roc_auc_5)
    aps.append(ap_5)
    return f1_scores, bacc_scores, roc_auc_scores, aps
    
    
def external_validation(model, data_train, data_test, hard_test, drop_cols, target_col):
    df_prediction_test = data_test[drop_cols]
    df_prediction_hard_test = hard_test[drop_cols]
    X_train = data_train.drop(drop_cols, axis = 1)
    y_train = data_train[target_col]
    model.fit(X_train, y_train)
    y_pred_test = model.predict(data_test.drop(drop_cols, axis = 1))
    y_prob_test = model.predict_proba(data_test.drop(drop_cols, axis = 1))[:, 1]
    df_prediction_test["Predicted_class"] = y_pred_test
    df_prediction_test["Predicted_prob"] = y_prob_test
    #Hard test
    y_pred_hard_test = model.predict(hard_test.drop(drop_cols, axis = 1))
    cm_hard_test = confusion_matrix(hard_test[target_col], y_pred_hard_test)
    tn, fp, fn, tp = cm_hard_test.ravel()
    ht_fpr = fp/(fp+tn)
    df_prediction_hard_test["Predicted_class"] = y_pred_hard_test
    df_prediction_hard_test["Predicted_prob"] = model.predict_proba(hard_test.drop(drop_cols, axis = 1))[:, 1]
    f1 = f1_score(data_test[target_col], y_pred_test)
    bacc = balanced_accuracy_score(data_test[target_col], y_pred_test)
    roc_auc = roc_auc_score(data_test[target_col], y_prob_test)
    ap = average_precision_score(data_test[target_col], y_prob_test)
    return f1, bacc, roc_auc,ap, ht_fpr, df_prediction_test, df_prediction_hard_test

data_types = ["rdk5","rdk6","rdk7","avalon","mordred","ph4"]

for data_type in tqdm(data_types):
    print(f"Processing {data_type}")
    fold_1_path = f"/Users/thechuongtrinh/Documents/Workspace/Master_thesis/Cdr1_inhibitors/dataset/Featurized_data/BM_stratified_sampling/Fold_1/{data_type}.csv"
    fold_1_df = pd.read_csv(fold_1_path)
    fold_2_path = f"/Users/thechuongtrinh/Documents/Workspace/Master_thesis/Cdr1_inhibitors/dataset/Featurized_data/BM_stratified_sampling/Fold_2/{data_type}.csv"
    fold_2_df = pd.read_csv(fold_2_path)
    fold_3_path = f"/Users/thechuongtrinh/Documents/Workspace/Master_thesis/Cdr1_inhibitors/dataset/Featurized_data/BM_stratified_sampling/Fold_3/{data_type}.csv"
    fold_3_df = pd.read_csv(fold_3_path)
    fold_4_path = f"/Users/thechuongtrinh/Documents/Workspace/Master_thesis/Cdr1_inhibitors/dataset/Featurized_data/BM_stratified_sampling/Fold_4/{data_type}.csv"
    fold_4_df = pd.read_csv(fold_4_path)
    fold_5_path = f"/Users/thechuongtrinh/Documents/Workspace/Master_thesis/Cdr1_inhibitors/dataset/Featurized_data/BM_stratified_sampling/Fold_5/{data_type}.csv"
    fold_5_df = pd.read_csv(fold_5_path)
    test_path = f"/Users/thechuongtrinh/Documents/Workspace/Master_thesis/Cdr1_inhibitors/dataset/Featurized_data/BM_stratified_sampling/external_test_set/{data_type}_test.csv"
    test_df = pd.read_csv(test_path)
    hard_test_path = f"/Users/thechuongtrinh/Documents/Workspace/Master_thesis/Cdr1_inhibitors/dataset/Featurized_data/BM_stratified_sampling/hard_test_set/{data_type}_hard_test.csv"
    hard_test_df = pd.read_csv(hard_test_path)
    drop_cols = ["ID", "Standardize_smile", "Activity"]
    target_col = "Activity"

    data_train = pd.read_csv(f"/Users/thechuongtrinh/Documents/Workspace/Master_thesis/Cdr1_inhibitors/dataset/Featurized_data/BM_stratified_sampling/training_set/{data_type}_train.csv")
    models, names = Classification()
    results_f1, results_recall,results_bacc, results_roc_auc,results_aps = list(), list(), list(), list(), list()
    results_external = list()
    SAVE_PREFIX = "/Users/thechuongtrinh/Documents/Workspace/Master_thesis/Cdr1_inhibitors/dataset/ml_model_selection/external_validation/"
    os.makedirs(SAVE_PREFIX, exist_ok = True)
    for i in range(len(models)):
        scores_f1, scores_bacc, scores_roc_auc, scores_aps = cv_evaluate(models[i], fold_1_df, fold_2_df, fold_3_df, fold_4_df, fold_5_df, drop_cols, target_col)
        f1_external, bacc_external, roc_auc_external, average_precision_external, hard_test_fpr, prediction_df_test, prediction_hard_test = external_validation(models[i], data_train, test_df, hard_test_df, drop_cols, target_col)
        prediction_df_test.to_csv(f"/Users/thechuongtrinh/Documents/Workspace/Master_thesis/Cdr1_inhibitors/dataset/ml_model_selection/external_validation/{data_type}_{names[i]}_prediction_test.csv")
        prediction_hard_test.to_csv(f"/Users/thechuongtrinh/Documents/Workspace/Master_thesis/Cdr1_inhibitors/dataset/ml_model_selection/external_validation/{data_type}_{names[i]}_prediction_hard_test.csv")
        results_external.append([f1_external, bacc_external, roc_auc_external,average_precision_external, hard_test_fpr])
        results_f1.append(scores_f1)
        results_bacc.append(scores_bacc)
        results_roc_auc.append(scores_roc_auc)
        results_aps.append(scores_aps)
        print(f"> Cross Validation Results for {names[i]}: F1 Score: {np.mean(scores_f1):.3f}, Balanced Accuracy: {np.mean(scores_bacc):.3f}, ROC AUC: {np.mean(scores_roc_auc):.3f}, Average Precision: {np.mean(scores_aps):.3f}")
        print(f"> External Validation Results for {names[i]}: F1 Score: {f1_external:.3f}, Balanced Accuracy: {bacc_external:.3f}, ROC AUC: {roc_auc_external:.3f}, Average Precision: {average_precision_external:.3f}, Hard Test FPR: {hard_test_fpr:.3f}")
    #Cross validation
    a_f1 = np.stack(results_f1)
    df_metrics_f1 = pd.DataFrame(a_f1.T, columns =names)
    #df_metrics_f1.to_csv(SAVE_PREFIX +data_type+"_model_selection_f1.csv")
    a_bacc = np.stack(results_bacc)
    df_metrics_bacc = pd.DataFrame(a_bacc.T, columns = names)
    #df_metrics_bacc.to_csv(SAVE_PREFIX +data_type+"_model_selection_bacc.csv")
    a_roc_auc = np.stack(results_roc_auc)
    df_metrics_roc_auc = pd.DataFrame(a_roc_auc.T, columns = names)
    #df_metrics_roc_auc.to_csv(SAVE_PREFIX +data_type+"_model_selection_roc_auc.csv")
    a_aps = np.stack(results_aps)
    df_metrics_aps = pd.DataFrame(a_aps.T, columns = names)
    #df_metrics_aps.to_csv(SAVE_PREFIX +data_type+"_model_selection_aps.csv")

    #External Validation
    a_external = np.stack(results_external)
    df_metrics_external = pd.DataFrame(a_external.T, columns = names)
    df_metrics_external.insert(0, "Metrics", ["F1", "BACC", "ROC_AUC", "Average_Precision","Hard_Test_FPR"])
    df_metrics_external.to_csv(SAVE_PREFIX +data_type+"_external_validation.csv")
    print("*"*100)




    


        