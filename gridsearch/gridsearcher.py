#%%
import time
import pickle
import numpy as np
import networkx as nx 
import pandas as pd

if __name__ == "__main__":
    import os
    os.chdir('..')
    print(os.getcwd())
    import yaml
    from tigger_package.orchestrator import Orchestrator


from tigger_package.orchestrator import Orchestrator
from tigger_package.metrics.distribution_metrics import NodeDistributionMetrics, EdgeDistributionMetrics
from tigger_package.tools import plot_adj_matrix, plot_hist
#%%


folder = "data/enron/"
grid_file = "data/enron/grid.yaml"
orchestrator = Orchestrator(folder)

class GridSearcher:
    def __init__(self, target_class, func_param, grid_file, loss_str):
        self.target_class = target_class
        self.func_param = func_param
        with open(grid_file, 'r') as file:
            config_dict = yaml.safe_load(file)
        self.config = config_dict
        self.loss_str = loss_str
        
    def objective(trial):
        class_dict = {}
        # loop through grid and suggest values for trial
        for group_name, group in self.config.item():
            if group_name[:5] == 'trail':  # group for trail suggest
                for k,v in group:
                    class_dict[k] = getattr(trial, group_name[7:])(k, *v)
        
        #add other values
        for k,v in self.config 
        
        
        
def grid_search(orchestrator, func, grid_file, loss_str):
    
    
    
    
    


ans = grid_search(orchestrator, orchestrator.create_graphsage_embedding, grid_file, "val_loss"