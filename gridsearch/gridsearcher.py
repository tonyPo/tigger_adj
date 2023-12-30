#%%
import time
import pickle
import optuna
from optuna.trial import TrialState
import numpy as np
import networkx as nx 
import pandas as pd
from datetime import datetime

if __name__ == "__main__":
    import os
    os.chdir('..')
    print(os.getcwd())
    import yaml
    from tigger_package.orchestrator import Orchestrator

from tigger_package.graphsage_unsup import TorchGeoGraphSageUnsup
from tigger_package.orchestrator import Orchestrator
from tigger_package.metrics.distribution_metrics import NodeDistributionMetrics, EdgeDistributionMetrics
from tigger_package.tools import plot_adj_matrix, plot_hist
from tigger_package.tab_ddpm.train_tab_ddpm import Tab_ddpm_controller
from tigger_package.inductive_controller import InductiveController
from tigger_package.mlp_edge_synthesizer import MLPEdgeSynthsizer
from tigger_package.bimlp_edge_synthesizer import BiMLPEdgeSynthesizer

#%%

class GridSearcher:
    def __init__(self, target_class, func_param, grid_file, loss_str):
        self.target_class = target_class
        self.func_param = func_param
        with open(grid_file, 'r') as file:
            config_dict = yaml.safe_load(file)
        self.config = config_dict
        self.loss_str = loss_str
        self.study = optuna.create_study(direction="minimize")
        self.test_params = []
        
        
    def apply_grid(self, visualization=False):
        # determin number of parallel job = cpu count - 2
        n_jobs = 2 #os.cpu_count() - 3
        self.study.optimize(self.objective, n_trials=self.config['n_trials'], timeout=None, n_jobs=n_jobs)
        complete_trials = self.study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
        
        # save trials
        out_path = self.config['out_path']
        if len(out_path) > 0:
            name = out_path + datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(name, 'wb') as handle:
                pickle.dump(self.study, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
        # visualizations
        if visualization:
            self.visualize_study()
            
        return complete_trials, self.study
    
    def parse_trail_dict(self, param_dict, target_dict, trial):
        for param, par_val in param_dict.items():
            if isinstance(par_val, dict):
                if "trial_meth" in list(par_val.keys()):
                    # this is parameter that is tested
                    par_val_cp = par_val.copy()
                    trial_meth = par_val_cp.pop('trial_meth')
                    target_dict[param] = getattr(trial, trial_meth)(param, **par_val_cp)
                    self.test_params.append(param)
                else:  # nested dict
                    sub_target_dict = {}
                    target_dict[param] = sub_target_dict
                    self.parse_trail_dict(par_val, sub_target_dict, trial)
            else:
                target_dict[param] = par_val
        
        
    def objective(self, trial):
        # parse the grid search parameters
        class_dict = {}
        self.parse_trail_dict(self.config['trial_info'], class_dict, trial)
            
        #correct for num neigtbors
        if 'num_neighbors' in class_dict.keys() :
            class_dict['num_neighbors'] = [class_dict['num_neighbors']]*class_dict['num_layers']
             
        class_instance = self.target_class(
            config_dict = class_dict,
            **self.func_param
        )
        
        metrics = class_instance.fit()
        loss = sum(metrics[self.loss_str][-5:]) / 5
        return loss
    
    
    def load_study(self, name):
        with open(name, 'rb') as handle:
            self.study = pickle.load(handle)
            
    def visualize_study(self):
        fig = optuna.visualization.plot_slice(self.study, params=self.test_params)
        fig.show()
        

def gridsearch_graphsage(folder):
    orchestrator = Orchestrator(folder)  
    grid_file = "gridsearch/graphsage_grid.yaml"
    func_param = {
        'nodes': orchestrator._load_nodes(),
        'edges': orchestrator._load_edges(),
        'config_path': "temp/"
    }
    loss_str = 'val_loss'
    
    searcher = GridSearcher(
        TorchGeoGraphSageUnsup,
        func_param, 
        grid_file, 
        loss_str)
    trials, study = searcher.apply_grid(visualization=True)
    # searcher.load_study('temp/gridsearch/graphsage_erdos20231218_090525')
    # searcher.visualize_study()
    return searcher

def gridsearch_ddpm(folder):
    orchestrator = Orchestrator(folder)  
    grid_file = "gridsearch/ddpm_grid.yaml"
    func_param = {
        'embed': orchestrator._load_normalized_embed(),
        'nodes': orchestrator._load_nodes(),
        'config_path': "temp/"
    }
    loss_str = 'val_loss'
    
    searcher = GridSearcher(
        Tab_ddpm_controller,
        func_param, 
        grid_file, 
        loss_str)
    trials, study = searcher.apply_grid(visualization=True)
    return searcher
    
  
def gridsearch_lstm(folder):
    orchestrator = Orchestrator(folder)  
    grid_file = "gridsearch/lstm_grid.yaml"
    func_param = {
        'nodes': orchestrator._load_nodes(),
        'edges':  orchestrator._load_edges(),
        'embed': orchestrator._load_normalized_embed(),
        'path': "temp/"        
    }
    loss_str = 'val_loss'
    
    searcher = GridSearcher(
        InductiveController,
        func_param, 
        grid_file, 
        loss_str)
    trials, study = searcher.apply_grid(visualization=True)
    return searcher

def gridsearch_mlp(folder):
    orchestrator = Orchestrator(folder)  
    grid_file = "gridsearch/mlp_grid.yaml"
    func_param = {
        'nodes': orchestrator._load_nodes(),
        'edges':  orchestrator._load_edges(),
        'embed': orchestrator._load_normalized_embed(),
        'path': "temp/"        
    }
    loss_str = 'val_loss'
    
    searcher = GridSearcher(
        MLPEdgeSynthsizer,
        func_param, 
        grid_file, 
        loss_str)
    trials, study = searcher.apply_grid(visualization=True)
    return searcher

def gridsearch_bimlp(folder):
    orchestrator = Orchestrator(folder)  
    grid_file = "gridsearch/bimlp_grid.yaml"
    func_param = {
        'nodes': orchestrator._load_nodes(),
        'edges':  orchestrator._load_edges(),
        'embed': orchestrator._load_normalized_embed(),
        'path': "temp/"        
    }
    loss_str = 'val_loss'
    
    searcher = GridSearcher(
        BiMLPEdgeSynthesizer,
        func_param, 
        grid_file, 
        loss_str)
    trials, study = searcher.apply_grid(visualization=True)
    return searcher
      
if __name__ == "__main__":
    folder = "data/erdos/"
    # gridsearch_graphsage(folder)
    # searcher = gridsearch_ddpm(folder)
    # searcher = gridsearch_lstm(folder)
    # searcher = gridsearch_mlp(folder)
    searcher = gridsearch_bimlp(folder)
    
# %%
