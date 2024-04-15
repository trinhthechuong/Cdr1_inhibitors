
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mols2grid
import matplotlib.cm as cm

from rdkit.ML.Cluster import Butina
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.rdBase import BlockLogs
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from tqdm.auto import tqdm
from mhfp.encoder import MHFPEncoder

from sklearn.manifold import TSNE

tqdm.pandas()
sns.set()
sns.set(rc={'figure.figsize': (12, 8)})
sns.set_style('darkgrid')



class Butina_clustering:
    """""
    Input:
    - df: dataframe
        must have Smiles column, Molecules column, ID column and bioactive columns
    - smiles_col: string
        name of smile column
    - mol_col: string
        name of mol column
    - active_col: string
        name of bioactive column ~ pIC50
    - mw: int => recomend 600
        molecular weight cutoff, value above cutoff will be removed
    - thresh: int => recomend 7
        bioactive cutoff, values above cutoff will be selected as active compounds
        in case of binary value, thresh must be 0.5  
    - dis_cutoff: float => recommend 0.7 for pharmacophore; 0.6 for docking
        dissimilarity, opposite to Tanimoto similarity cutoff
    - cps: float
        minimum number of compounds in cluster to select cluster center
    - radius: int (2)
        ECFP value
    - nBits: int (2048)
        ECFP value
    
    Return:
    - active_set: dataframe
        diverse active molecules selected, contaning Smiles, Molecules, ID and bioactive columns
    - cluster_centers: list
        list of active molecules selected
    - df_active: dataframe
        all active molecules with cluster index
    """""
    def __init__(self, df,ID, smiles_col, active_col, mol_col = 'ROMol', activity_thresh = 7, 
                dis_cutoff = 0.7, cps = 5):
        self.data = df
        self.ID = ID
        self.smiles_col = smiles_col
        self.active_col = active_col
        self.mol_col = mol_col
        self.thresh = activity_thresh
        self.cutoff = dis_cutoff
        self.cps = cps
  
    

    
    def create_cps(self, df):
        compounds = []
        for _, chembl_id, mol in df[[self.ID, self.mol_col]].itertuples():
            mol.SetProp('_Name',chembl_id)
            compounds.append(mol)
        #self.smiles  = self.data[self.smiles_col].values. tolist()
        #self.fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=self.radius, nBits = self.nBits) for mol in compounds]
        fingerprints = [MHFPEncoder.secfp_from_smiles(smile) for smile in self.data["Canomicalsmiles"].values]
        return compounds, fingerprints
        
    def tanimoto(self, vector1, vector2):
        a = np.where(vector1 == 1)[0]
        b = np.where(vector2 == 1)[0]
        return len(np.intersect1d(a, b)) / (float(len(a) + len(b)) - len(np.intersect1d(a, b)))
    
    def BulkTanimotoSimilarity_dev(self, fp_i, list_fp0_i):
        tanimoto_sim= []
        for fps in list_fp0_i:
            tanimoto_sim.append(self.tanimoto(fp_i, fps))
        return tanimoto_sim

    def tanimoto_distance_matrix(self, fp_list):
        """Calculate distance matrix for fingerprint list"""
        dissimilarity_matrix = []
        # Notice how we are deliberately skipping the first and last items in the list
        # because we don't need to compare them against themselves
        for i in range(1, len(fp_list)):
            # Compare the current fingerprint against all the previous ones in the list
            similarities = self.BulkTanimotoSimilarity_dev(fp_list[i], fp_list[:i])
            # Since we need a distance matrix, calculate 1-x for every element in similarity matrix
            dissimilarity_matrix.extend([1 - x for x in similarities])
        return dissimilarity_matrix
    
    def cluster_fingerprints(self, fingerprints, cutoff=0.2):
        """Cluster fingerprints
        Parameters:
            fingerprints
            cutoff: threshold for the clustering
        """
        # Calculate Tanimoto distance matrix
        self.distance_matrix = self.tanimoto_distance_matrix(fingerprints)
        # Now cluster the data with the implemented Butina algorithm:
        clusters = Butina.ClusterData(self.distance_matrix, len(fingerprints), cutoff, isDistData=True)
        clusters = sorted(clusters, key=len, reverse=True)
        return clusters
    
    def active_clustering(self):
        self.df_active = self.data[self.data[self.active_col] > self.thresh]
        
        self.cls_data = self.df_active.copy().reset_index(drop = True)
        self.compounds,  self.fingerprints = self.create_cps(df = self.cls_data)

        clusters = self.cluster_fingerprints(self.fingerprints, cutoff=self.cutoff)

        # Give a short report about the numbers of clusters and their sizes
        num_clust_g1 = sum(1 for c in clusters if len(c) == 1)
        num_clust_g5 = sum(1 for c in clusters if len(c) > 5)
        num_clust_g25 = sum(1 for c in clusters if len(c) > 25)
        num_clust_g100 = sum(1 for c in clusters if len(c) > 100)

        print("total # clusters: ", len(clusters))
        print("# clusters with only 1 compound: ", num_clust_g1)
        print("# clusters with >5 compounds: ", num_clust_g5)
        print("# clusters with >25 compounds: ", num_clust_g25)
        print("# clusters with >100 compounds: ", num_clust_g100)
        return clusters
        
    def data_processing(self):
            
            #self.df_all, self.df_active = self.filter_data()
            
            
        clusters = self.active_clustering()
        
       
            
        a = [c for c in clusters if len(c) > self.cps] # n = least compounds in cluster
        cluster_centers = [self.compounds[c[0]] for c in a]
        print("cluster_centers",len(cluster_centers))
       
        # active
        center_id = []
        for i in cluster_centers:
            center_id.append(i.GetProp('_Name'))
        
        idx =[]
        for key, value in enumerate(self.cls_data[self.ID]):
            if value in center_id:
                idx.append(key)
       
        active_set = self.cls_data.iloc[idx,:]
        
        #mark cluster
        self.cls_data['Cluster'] = np.zeros(len(self.cls_data))
        cls_df = pd.DataFrame(clusters).T
        for i in cls_df.columns:
            idx = cls_df.iloc[:,i].dropna().values
            self.cls_data.loc[idx, 'Cluster'] = i
    
        score = silhouette_score(self.fingerprints,self.cls_data['Cluster'], random_state= 42)
        print('Silhouette Score:', score)
        return active_set, cluster_centers, self.cls_data
    

class Molecule_clustering:
    """""
    Input:
    - data: dataframe
        must have Smiles column, ID column and bioactive columns
    - ID: string
        identification of molecules
    - mol_col: string
        name of column containing Molecules
    - active_col: string
        name of bioactive column ~ pIC50
    - cluser_range: range
        range of clusters to analyze
    - method: string
        clustering method: {'KMeans', 'Agglomerative'}
    - linkage: string
        if method == Agglomerative; linkage :{‘ward’, ‘complete’, ‘average’, ‘single’}
    - thresh: int => recomend 7
        bioactive cutoff, values above cutoff will be selected as active compounds
        *in case of binary value, thresh must be 0.5  
    - radius: int (2)
        ECFP value
    - nBits: int (2048)
        ECFP value
    Return:
    - df: dataframe
        containing cluster index columns
        
    """""
    def __init__(self,data,  ID, active_col, mol_col, cluster_range,
                 method ='KMeans', linkage ='ward', thresh =7, radius=2, nBits = 2048):
        
        self.data = data
        self.method = method
        self.linkage = linkage
        self.ID = ID
        self.active_col = active_col
        self.mol_col =mol_col
        self.cluster_range=cluster_range
        self.thresh = thresh
        self.radius = radius
        self.nBits = nBits
        self.df = self.data[self.data[self.active_col] > self.thresh]
        
        # method select
        if self.method == 'KMeans':
            self.kmeans_kwargs = {"init": "random",
                         "n_init": "auto",
                         "max_iter": 300,
                         }
        elif self.method == 'Agglomerative':
            self.kmeans_kwargs = {'linkage':self.linkage} #{‘ward’, ‘complete’, ‘average’, ‘single’}
            
    def mol2ecfp(self,mol):
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits = 2048)
        ar = np.zeros((1,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, ar)
        return ar
    
        
    def Silhouette_Score_method(self, X, cluster_range):
        score_list = []
        for k in tqdm(cluster_range):
            if self.method == 'KMeans':
                km = KMeans(n_clusters=k,**self.kmeans_kwargs, random_state= 42)
            elif self.method == 'Agglomerative':
                km = AgglomerativeClustering(n_clusters=k,**self.kmeans_kwargs)
            cluster_labels = km.fit_predict(X)
            score = silhouette_score(X,cluster_labels, random_state= 42)
            score_list.append([k,score])

        score_df = pd.DataFrame(score_list,columns=["K","Silhouette Score"])
        display(score_df.head(5))

        ax = sns.lineplot(x="K",y="Silhouette Score",data=score_df)
        ax.set_title(f'Silhouette Score - {self.method}', weight = 'bold', fontsize = 24)
        ax.set_xlabel('Number of Cluster', fontsize =16)
        ax.set_ylabel('Silhouette Score', fontsize =16)
        ax.set_xticks(cluster_range)
        plt.show()
        
    def silhouette_plot(self, X,cluster_labels):
        """
        Adapted from https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
        """
        sns.set_style('whitegrid')
        sample_df = pd.DataFrame(silhouette_samples(X,cluster_labels),columns=["Silhouette"])
        sample_df['Cluster'] = cluster_labels
        n_clusters = max(cluster_labels+1)
        color_list = [cm.nipy_spectral(float(i) / n_clusters) for i in range(0,n_clusters)]
        ax = sns.scatterplot()
        ax.set_xlim([-0.1, 1])
        ax.set_ylim([0, len(X) + (n_clusters + 1) * 10])
        silhouette_avg = silhouette_score(X, cluster_labels)
        y_lower = 10
        unique_cluster_ids = sorted(sample_df.Cluster.unique())
        for i in unique_cluster_ids:
            cluster_df = sample_df.query('Cluster == @i')
            cluster_size = len(cluster_df)
            y_upper = y_lower + cluster_size
            ith_cluster_silhouette_values = cluster_df.sort_values("Silhouette").Silhouette.values
            color = color_list[i]
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                        0, ith_cluster_silhouette_values,
                        facecolor=color, edgecolor=color, alpha=0.7)
            ax.text(-0.05, y_lower + 0.5 * cluster_size, str(i),fontsize="small")
            y_lower = y_upper + 10
        ax.axvline(silhouette_avg,color="red",ls="--")
        ax.set_xlabel("Silhouette Score")
        ax.set_ylabel("Cluster")
        ax.set(yticklabels=[]) 
        ax.yaxis.grid(False)
        plt.show()
    
    def select_k_cluster(self):
        #self.data['mol'] = self.data[self.smile_col].apply(Chem.MolFromSmiles)
        self.df['fp'] = self.df[self.mol_col].apply(self.mol2ecfp)
        self.X = np.stack(self.df['fp'])
        
        self.Silhouette_Score_method(self.X, self.cluster_range)
        
        self.clustering()
        return self.cls_df
        
    def clustering(self):
        num_clusters = int(input("Number of cluser:"))
        if self.method == 'KMeans':
            km_opt = KMeans(n_clusters=num_clusters,**self.kmeans_kwargs, random_state= 42)
        elif self.method == 'Agglomerative':
            km_opt = AgglomerativeClustering(n_clusters=num_clusters,**self.kmeans_kwargs)
        km_opt.fit(self.X)
        self.clusters_opt = km_opt.labels_
        self.silhouette_plot(self.X,self.clusters_opt)

        self.cls_df = self.df[[self.ID,'fp',self.mol_col,]]
        self.cls_df.loc[:,'Cluster']=self.clusters_opt

        ax = pd.Series(self.clusters_opt).value_counts().sort_index().plot(kind="bar")
        ax.set_xlabel("Cluster Number", fontsize = 18)
        ax.set_ylabel("Cluster Size", fontsize = 18)
        ax.tick_params(axis='x', rotation=0)
        plt.show()
        
        
