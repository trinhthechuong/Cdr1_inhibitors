import os
import glob
import numpy as np
import pandas as pd
from pandas import DataFrame

import seaborn as sns
from seaborn import heatmap
sns.set('notebook')
import scikit_posthocs as sp
from typing import Union, List, Tuple
from scipy.stats import t
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.axes import SubplotBase
from matplotlib.colorbar import ColorbarBase, Colorbar
from matplotlib.colors import ListedColormap


class statistic_test:
    """
    - Statistical comparison of data type

    Input:
    ------
    meta_data: pandas.DataFrame
        Data for comparison
    save_data: bool (default = False)
        True if want to save the posthoc analysis data
    scoring: str
        if task_type = (C): 'f1', 'average_precision', 'recall'
        if task_type = (R): 'r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'
    posthoc_method: str ('Wilcoxon', 'Mannwhitney') 
        Method for posthoc analysis
    refinement: bool (default = False)
        True to remove results equal to zero in boxplot visualization
    
    Returns:
    --------
    pc: pandas.DataFrame
        Posthoc analysis data
    heatmap: fig
        Posthoc analysis visualization
    box_plot: fig
        Performance comparison among all data types
    """

    def __init__(self, meta_data, save_data = False, scoring = 'f1', kind_analysis ='Meta',
                 posthoc_method ='Wilcoxon', refinement = False):
        self.meta_data = meta_data
        self.results = meta_data.values.T
        self.names = meta_data.columns
        self.save_data = save_data
        self.scoring = scoring
        self.posthoc_method = posthoc_method
        self.refinement = refinement
        self.kind_analysis = kind_analysis
        self.meta_folder = "Meta_folder"
        # Check whether the specified path exists or not
        isExist = os.path.exists(self.meta_folder)
        if not isExist:
           # Create a new directory because it does not exist
            os.makedirs(self.meta_folder)
            #print("The new directory is created!")
        
    def posthoc(self):
       #Create dataframe for Posthocs
        a = np.stack(self.results)
        self.df_metrics = pd.DataFrame(a.T, columns = self.names)
        self.df_melt = pd.melt(self.df_metrics.reset_index(), id_vars=['index'], value_vars=self.df_metrics.columns)
        self.df_melt.columns = ['index', 'Method', 'Scores']
        
        if self.posthoc_method =='Wilcoxon':
            self.pc = sp.posthoc_wilcoxon(self.df_melt, val_col='Scores', group_col='Method', p_adjust='holm')
            self.pc.to_csv(f"{self.meta_folder}/{self.posthoc_method}_{self.kind_analysis}.csv")
        elif self.posthoc_method == 'Mannwhitney':
            self.pc =sp.posthoc_mannwhitney(self.df_melt, val_col='Scores', group_col='Method', p_adjust='holm')
            self.pc.to_csv(f"{self.meta_folder}/{self.posthoc_method}_{self.kind_analysis}.csv")
            
        plt.figure(figsize = (10,7))
        plt.title(f"Posthoc {self.posthoc_method}", fontsize = 24, weight = 'semibold')
        heatmap_args = {'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}
        self.sign_plot(self.pc, **heatmap_args)
        plt.savefig(f"{self.meta_folder}/{self.posthoc_method}_{self.kind_analysis}_heatmap.png", dpi = 300)
        if self.save_data == True:
            print("META DATA SAVING...")
            self.pc.to_csv(f"{ self.meta_folder}/{self.posthoc_method}_{self.kind_analysis}.csv")


    
    def sign_plot(
        self,
        x: Union[List, np.ndarray, DataFrame],
        g: Union[List, np.ndarray] = None,
        flat: bool = False,
        labels: bool = True,
        cmap: List = None,
        cbar_ax_bbox: List = None,
        ax: SubplotBase = None,
        **kwargs) -> Union[SubplotBase, Tuple[SubplotBase, Colorbar]]:

        for key in ['cbar','vmin','vmax','center']:
            if key in kwargs:
                del kwargs
          
        if isinstance(x, DataFrame):
            df = x.copy()
        else:
            x = np.array(x)
            g = g or np.arange(x.shape[0])
            df = DataFrame(np.copy(x), index=g, columns=g)

        dtype = df.values.dtype

        if not np.issubdtype(dtype, np.integer) and flat:
            raise ValueError("X should be a sign_array or DataFrame of integers")
        elif not np.issubdtype(dtype, np.floating) and not flat:
            raise ValueError("X should be an array or DataFrame of float p values")

        if not cmap and flat:
            cmap = ['1', '#fbd7d4', '#1a9641']
        
        elif not cmap and not flat:
            cmap = ['1', '#fbd7d4', '#005a32', '#238b45', '#a1d99b']

        if flat:
            np.fill_diagonal(df.values, -1)
            hax = heatmap(df, vmin=-1, vmax=1, cmap=ListedColormap(cmap),
                      cbar=False, ax=ax, **kwargs)
        if not labels:
            hax.set_xlabel('')
            hax.set_ylabel('')
            return hax
        
        else:
            df[(x < 0.001) & (x >= 0)] = 1
            df[(x < 0.01) & (x >= 0.001)] = 2
            df[(x < 0.05) & (x >= 0.01)] = 3
            df[(x >= 0.05)] = 0

            np.fill_diagonal(df.values, -1)

            if len(cmap) != 5:
                raise ValueError("Cmap list must contain 5 items")

            hax = heatmap(df, vmin=-1, vmax=3, cmap=ListedColormap(cmap), center=1,
                          cbar=False, ax=ax, **kwargs)
            if not labels:
                hax.sex_xlabel("")
                hax.sex_ylabel("")
          
            cbar_ax = hax.figure.add_axes(cbar_ax_bbox or [0.95, 0.35, 0.04, 0.3])
            cbar = ColorbarBase(cbar_ax, cmap=(ListedColormap(cmap[2:] + [cmap[1]])), norm=colors.NoNorm(),
                            boundaries=[0, 1, 2, 3, 4])
            cbar.set_ticks(np.linspace(0.3, 3, 4))
            cbar.set_ticklabels(['p < 0.001', 'p < 0.01', 'p < 0.05', 'NS'])
            cbar.outline.set_linewidth(1)
            cbar.outline.set_edgecolor('0.5')
            cbar.ax.tick_params(size=0)

            return hax, cbar
    

    def corrected_std(self,differences, n_train, n_test):
        kr = len(differences)
        corrected_var = np.var(differences, ddof=1) * (1 / kr + n_test / n_train)
        corrected_std = np.sqrt(corrected_var)
        return corrected_std


    def compute_corrected_ttest(self,differences, df, n_train, n_test):
        mean = np.mean(differences)
        std = self.corrected_std(differences, n_train, n_test)
        t_stat = mean / std
        p_val = t.sf(np.abs(t_stat), df)  # right-tailed t-test
        return t_stat, p_val
    
    def boxplot_visualize(self,data):
      #Vẽ biểu đồ boxplot thể hiện hiệu quả của từng model, từ đó chọn model phù hợp để tối ưu.
        if self.refinement == True:
            print("REFINEMENT DATA BEFORE MAKING BOXPLOT...")
            data = data[data>0]
        
        mean = list()
        for i in range(len(data.columns)):
            col = data.columns[i]
            x = data[col].mean().round(3)
            mean.append(x)
        df = np.array(mean)   
        ser = pd.Series(df, index =data.columns)
        print(ser)


        dict_columns = {'Mean':mean,'Method':data.columns,}
        df = pd.DataFrame(dict_columns)


        sns.set_style("whitegrid")
        plt.figure(figsize=(20,8))
        box_plot = sns.boxplot(data=data,showmeans=True ,meanprops={"marker":"d",
                           "markerfacecolor":"white", 
                           "markeredgecolor":"black",
                          "markersize":"10"})
        box_plot.axes.set_title("Data type comparison", fontsize=20, weight = 'semibold')
        #box_plot.set_xlabel("Method", fontsize=14)
        #box_plot.set_ylabel("Results", fontsize=14)
        vertical_offset = df["Mean"].median()*0.01

        for xtick in box_plot.get_xticks():
            box_plot.text(xtick,ser[xtick]+ vertical_offset,ser[xtick], 
            horizontalalignment='center',color='k',weight='semibold',fontsize = 12)

        #box_plot.get_xticks(range(len(data.columns)))
        box_plot.set_xticklabels(data.columns, rotation='horizontal', fontsize = 12)
        box_plot.set_ylabel(f"{self.scoring}".upper(), fontsize=16, weight='semibold')
        plt.savefig(f"{self.meta_folder}/{self.kind_analysis}_compare_visualize.png", dpi = 300)

    def visualize(self):
        self.posthoc()
        
        self.boxplot_visualize(data=self.meta_data)
        
        

class statistic_data(statistic_test):
    """
    - Make meta data and run statistic test

    Input:
    ------
    data_dir: path
        Directory contains data to run posthoc analysis
    save_data: bool (default = False)
        True if want to save the posthoc analysis data
    scoring: str
        if task_type = (C): 'f1', 'average_precision', 'recall'
        if task_type = (R): 'r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'
    posthoc_method: str ('Wilcoxon', 'Mannwhitney') 
        Method for posthoc analysis
    refinement: bool (default = False)
        True to remove results equal to zero in boxplot visualization
    kind_analysis: str ('Meta', 'Subgroup'):
        Kind of posthoc analysis
    
    Returns:
    --------
    meta_data: pandas.DataFrame
        Data for comparison
    pc: pandas.DataFrame
        Posthoc analysis data
    heatmap: fig
        Posthoc analysis visualization
    box_plot: fig
        Performance comparison among all data types
    """
    def __init__(self,data_dir, save_data=False, refinement =False,
                scoring = 'f1', posthoc_method ='Wilcoxon', kind_analysis ='Meta'):
        self.data_dir = data_dir
        self.save_data = save_data
        self.refinement = refinement
        self.scoring = scoring
        self.posthoc_method =posthoc_method
        self.kind_analysis = kind_analysis
        os.chdir(self.data_dir)
        self.data_name = []
        for i in sorted(glob.glob(self.data_dir+"/*.csv")):
            self.data_name.append(i[len(self.data_dir):-4])
            
    def meta_data(self, data_frame,data_name):
        # flatten list of lists
        flat_ls= data_frame.values.reshape(-1) 
        data = pd.DataFrame(flat_ls,columns = [data_name])
        return data
    
    def subgroup_data(self, data_frame,data_name, kind='best'):
        if kind == 'best':
            col = data_frame.mean().argmax()
        elif kind == 'worst':
            col = data_frame.mean().argmin()
        best_model = data_frame.iloc[:,col].values
        data = pd.DataFrame(best_model,columns = [data_name])
        return data
    
    def make_meta_data(self):
        self.meta_tab = pd.DataFrame()
        os.chdir(self.data_dir)
        for i in self.data_name:
            data = pd.read_csv(f"{i}.csv").drop(['Unnamed: 0'], axis =1)
            if data.shape[0]==30:
                if self.kind_analysis == 'Meta':
                    df = self.meta_data(data_frame = data,data_name = i[:-16])
                #df = df[df[i[:-16]] > 0]
                else:
                    df = self.subgroup_data(data_frame = data,data_name = i[:-16], kind='best')
                self.meta_tab = pd.concat([self.meta_tab, df], axis =1).reset_index(drop = True)
            else:
                pass

    def make_posthoc(self,meta_data, save_data, refinement,posthoc_method,scoring):
        compare = statistic_test(meta_data, save_data = save_data, refinement = refinement,
                                 posthoc_method =posthoc_method, scoring = scoring, kind_analysis=self.kind_analysis)
        compare.visualize()
        
    def fit(self):
        self.make_meta_data()
        self.make_posthoc(meta_data = self.meta_tab, save_data=self.save_data, scoring = self.scoring,
                         posthoc_method=self.posthoc_method, refinement=self.refinement)
