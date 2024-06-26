from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import f1_score, average_precision_score
from itertools import chain
import numpy as np
def scaffold_split(data, smiles_col, test_size = 0.2,random_state = 42):
    """
    Split a molecule dataset into training and test sets based on scaffolds.

    Parameters:
    - data (pd.DataFrame): The dataset containing molecule data.
    - smiles_col (str): The name of the column containing SMILES strings.
    - test_size (float): Proportion of the dataset to include in the test split.
    - random_state (int): Random state for reproducibility.

    Returns:
    - data_train (pd.DataFrame): Training set.
    - data_test (pd.DataFrame): Test set.
    """
    scaffolds = {}
    for idx, row in data.iterrows():
        smiles = row[smiles_col]
        mol = Chem.MolFromSmiles(smiles)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [idx]
        else:
            scaffolds[scaffold].append(idx)

    scaffold_lists = list(scaffolds.values())
    np.random.seed(random_state)
    np.random.shuffle(scaffold_lists)

    num_molecules = len(data)
    num_test = int(np.floor(test_size * num_molecules))
    train_idx, test_idx = [], []
    for scaffold_list in scaffold_lists:
        if len(test_idx) + len(scaffold_list) <= num_test:
            test_idx.extend(scaffold_list)
        else:
            train_idx.extend(scaffold_list)

    data_train = data.iloc[train_idx]
    data_test = data.iloc[test_idx]

    return data_train, data_test

def scaffold_cv_repeated(data, model, activity_col, ID, smiles_col, k =10, n_repeated = 3, random_state = 42, score = 'f1'):
    """
    Perform repeated k-fold cross-validation based on BM scaffolds.

    Parameters:
    - data (pd.DataFrame): The dataset containing molecule data.
    - model: The model to be evaluated.
    - activity_col (str): The name of the column containing activity data.
    - ID (str): The name of the column containing molecule IDs.
    - smiles_col (str): The name of the column containing SMILES strings.
    - k (int): Number of folds.
    - n_repeated (int): Number of repetitions.
    - random_state (int): Random state for reproducibility.
    - score (str): The metric to be used for evaluation. Either 'f1' or 'average_precision'.

    Returns:
    - scores (list): List of scores for each repetition.
    """
    scaffolds = {}
    for idx, row in data.iterrows():
        smiles = row[smiles_col]
        mol = Chem.MolFromSmiles(smiles)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [idx]
        else:
            scaffolds[scaffold].append(idx)
    scaffold_lists = list(scaffolds.values())
    scores = []
    shuffle_seeds = []
    for i in range(n_repeated):
        shuffle_seeds.append(random_state + i)
    for shuffle_seed in shuffle_seeds:
        np.random.seed(shuffle_seed)
        np.random.shuffle(scaffold_lists)
        index_buckets = [[] for _ in range(k)]
        for group_indices in scaffold_lists:
            bucket_chosen = int(np.argmin([len(bucket) for bucket in index_buckets]))
            index_buckets[bucket_chosen].extend(group_indices)

        
        if score == 'f1':
            for i in range(k):
                train_indices = list(chain.from_iterable(index_buckets[:i] + index_buckets[i + 1:]))
                val_indices = index_buckets[i]
                data_train = data.loc[train_indices,:]
                data_test = data.loc[val_indices,:]
                X_train = data_train.drop([activity_col, ID,smiles_col ],axis = 1)
                y_train = data_train[activity_col]
                X_test = data_test.drop([activity_col, ID,smiles_col ],axis = 1)
                y_test = data_test[activity_col]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = f1_score(y_test, y_pred)
                scores.append(score)
        else:
            for i in range(k):
                train_indices = list(chain.from_iterable(index_buckets[:i] + index_buckets[i + 1:]))
                val_indices = index_buckets[i]
                data_train = data.loc[train_indices,:]
                data_test = data.loc[val_indices,:]
                X_train = data_train.drop([activity_col, ID,smiles_col ],axis = 1)
                y_train = data_train[activity_col]
                X_test = data_test.drop([activity_col, ID,smiles_col ],axis = 1)
                y_test = data_test[activity_col]
                model.fit(X_train, y_train)
                y_proba = model.predict_proba(X_test)
                score = average_precision_score(y_test, y_proba[:,1])
                scores.append(score)
    
    return scores