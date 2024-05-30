from scipy.stats import wilcoxon
import numpy as np
import warnings
warnings.filterwarnings("ignore")
metric = ["ap","f1","roc","bacc","ht_fpr"]
for m in metric:
    with_gnn = np.loadtxt(f"/Users/thechuongtrinh/Documents/Workspace/Master_thesis/Cdr1_inhibitors/dataset/Ensemble_model/validation_stacking/ml_gnn/{m}_list_gnn.txt")
    no_gnn = np.loadtxt(f"/Users/thechuongtrinh/Documents/Workspace/Master_thesis/Cdr1_inhibitors/dataset/Ensemble_model/validation_stacking/without_gnn/{m}_list_no_gnn.txt")
    statistic, p_value = wilcoxon(with_gnn, no_gnn)
    print(f"{m} p-value:", p_value)