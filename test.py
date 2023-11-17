#%% 
import numpy as np
import pandas as pd
import pickle
import torch
import os
from tigger_package.orchestrator import Orchestrator
from tigger_package.tab_ddpm.train_tab_ddpm import train

#%%
enron_folder = "data/erdos/"
orchestrator = Orchestrator(enron_folder)
# %%

hist= train(config=orchestrator.config['tab_ddpm'], 
      path = orchestrator.config_path,
      nodes=orchestrator._load_nodes())
hist.loss.plot()
# %%
