import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from numpy import where, random
from conformation_encode.scaffold_split import scaffold_split

rdk7_active_normal_path = "rdk7_normal_active.csv"
rdk7_inactive_normal_path = "rdk7_inactive_normal.csv"
rdk7_active_outlier_path = "rdk7_outliers_active.csv"
rdk7_inactive_outliers_path = "/Users/thechuongtrinh/Documents/Workspace/Master_thesis/Cdr1/data/Official/Featurizer_data/rdk7_hard_test.csv"

rdk7_active_normal_df = pd.read_csv(rdk7_active_normal_path)
rdk7_inactive_normal_df = pd.read_csv(rdk7_inactive_normal_path)
rdk7_active_outlier_df = pd.read_csv(rdk7_active_outlier_path)
rdk7_inactive_outliers_df = pd.read_csv(rdk7_inactive_outliers_path)

# data_visualize = pd.concat([rdk7_active_normal_df,rdk7_active_outlier_df,rdk7_inactive_normal_df,rdk7_inactive_outliers_df], axis=0).reset_index(drop = True)
# print(data_visualize.shape[0])
# active_normal = ["active_normal"]*rdk7_active_normal_df.shape[0]
# active_outliers = ["active_outliers"]*rdk7_active_outlier_df.shape[0]
# inactive_normal = ["inactive_normal"]*rdk7_inactive_normal_df.shape[0]
# inactive_outliers = ["inactive_outliers"]*rdk7_inactive_outliers_df.shape[0]
# activity_type = active_normal + active_outliers + inactive_normal + inactive_outliers
# data_visualize["Activity_type"] = activity_type

# tsne = TSNE(n_components=2, random_state=42)
# pca = PCA(n_components=50)
# X  = data_visualize.drop(["ID","Standardize_smile","Activity", "Activity_type"], axis = 1)
# X_transformed= pca.fit_transform(X)
# y = data_visualize["Activity_type"]
# X_transformed = tsne.fit_transform(X)
# sns.scatterplot(X_transformed[:, 0], X_transformed[:, 1], hue = y,palette="rocket")
# plt.show()

def splitting_data(data, test_size, valid_size ,seed):
    data_train, data_test = scaffold_split(data, smiles_col = "Standardize_smile", test_size = test_size,random_state = seed)
    data_train = data_train.reset_index(drop=True)
    data_test = data_test.reset_index(drop=True)
    data_train, data_valid = scaffold_split(data_train, smiles_col = "Standardize_smile", test_size = valid_size,random_state = seed)
    data_train = data_train.reset_index(drop=True)
    data_valid = data_valid.reset_index(drop=True)
    return data_train, data_valid, data_test

train_active_rdk7, valid_active_rdk7, test_active_rdk7 = splitting_data(rdk7_active_normal_df, 0.341, 0.323, 42)
train_inactive_rdk7, valid_inactive_rdk7, test_inactive_rdk7 = splitting_data(rdk7_inactive_normal_df, 0.252, 0.2, 42)
train_rdk7 = pd.concat([train_active_rdk7, train_inactive_rdk7,rdk7_active_outlier_df], axis = 0).reset_index(drop=True)
test_rdk7 = pd.concat([test_active_rdk7, test_inactive_rdk7], axis = 0).reset_index(drop=True)
valid_rdk7 = pd.concat([valid_active_rdk7, valid_inactive_rdk7], axis = 0).reset_index(drop=True)

train_total = pd.concat([train_rdk7, valid_rdk7], axis = 0).reset_index(drop = True)

train_type = ["train"]*train_rdk7.shape[0]
test_type = ["test"]*test_rdk7.shape[0]
valid_type = ["valid"]*valid_rdk7.shape[0]
data_type = train_type + test_type + valid_type

data = pd.concat([train_total, test_rdk7], axis=0)
data["Data_type"] = data_type



tsne = TSNE(n_components=2, random_state=42)
pca = PCA(n_components=50)
X  = data.drop(["ID","Standardize_smile","Activity", "Data_type"], axis = 1)
X= pca.fit_transform(X)
y = data["Data_type"]
X_transformed = tsne.fit_transform(X)
X_train_transform = X_transformed[0:train_rdk7.shape[0]]
X_valid_transfrom = X_transformed[train_rdk7.shape[0]:train_rdk7.shape[0]+valid_rdk7.shape[0]]
print(X_valid_transfrom.shape)
X_test_transform = X_transformed[train_rdk7.shape[0]+valid_rdk7.shape[0]:]
y_train = train_rdk7["Activity"]
y_test = test_rdk7["Activity"]
y_valid = valid_rdk7["Activity"]
print(test_rdk7["Activity"].values.sum())
# sns.scatterplot(X_train_transform[:, 0], X_train_transform[:, 1], hue = y_train,palette="rocket")
# sns.scatterplot(X_valid_transfrom[:, 0], X_valid_transfrom[:, 1], hue = y_valid,palette="Oranges")
# sns.scatterplot(X_test_transform[:, 0], X_test_transform[:, 1], hue = y_test, palette="viridis")


# import seaborn as sns
# import matplotlib.pyplot as plt

# Scatter plots for training, validation, and testing data
# Scatter plots for training, validation, and testing data
# Scatter plots for training, validation, and testing data
train_plot = sns.scatterplot(X_train_transform[:, 0], X_train_transform[:, 1], hue=y_train, palette="rocket")
valid_plot = sns.scatterplot(X_valid_transfrom[:, 0], X_valid_transfrom[:, 1], hue=y_valid, palette="Oranges")
test_plot = sns.scatterplot(X_test_transform[:, 0], X_test_transform[:, 1], hue=y_test, palette="viridis")

# Create legend handles and labels
legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Train Class 0'),
                  plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Train Class 1'),
                  plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Validation Class 0'),
                  plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Validation Class 1'),
                  plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='Test Class 0'),
                  plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='Test Class 1')]

# Add legend with custom handles and labels
plt.legend(handles=legend_handles, loc='upper right')

# Show the plot
plt.show()

plt.show()