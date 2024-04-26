# Databricks notebook source
# MAGIC %pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 
# MAGIC %pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
# MAGIC %pip install scikit-learn==1.2
# MAGIC %pip install threadpoolctl==3.2.0

# COMMAND ----------

# Set root dir
import os
os.chdir('..')
print(os.getcwd())

# COMMAND ----------

from gridsearch.gridsearcher import *
import yaml
from tigger_package.orchestrator import Orchestrator

# COMMAND ----------

import torch
torch.__version__

# COMMAND ----------

folder = "/Workspace/Repos/antonius.b.a.poppe@nl.abnamro.com_old/tigger_adj/data/reddit/"
# searcher = gridsearch_graphsage(folder)
# searcher = gridsearch_ddpm(folder)
searcher = gridsearch_lstm(folder)
# searcher = gridsearch_mlp(folder)
# searcher = gridsearch_bimlp(folder)
#147 in 5:20 -> 2,1 min per epoch

# COMMAND ----------

searcher.study.best_params

# COMMAND ----------

 Trial 1 finished with value: 5.520856475830078 and parameters: {'n_walks': 5885, 'num_clusters': 14, 'cluster_emb_dim': 9, 'l_w': 10, 'num_epochs': 4518, 'pca_components': 32, 'lr': 1.3415876376134754e-05, 'kl_weight': 0.002422183362569752, 'dropout': 0.05037192315433581, 'weight_decay': 0.002670319405414444}. Best is trial 0 with value: 4.043224477767945.
 [W 2024-02-04 13:04:35,993] Trial 2 failed with parameters: {'n_walks': 4988, 'num_clusters': 13, 'cluster_emb_dim': 9, 'l_w': 17, 'num_epochs': 2168, 'pca_components': 13, 'lr': 0.0006059381474972682, 'kl_weight': 0.013346618732457191, 'dropout': 0.013603934620045178, 'weight_decay': 2.689769045113198e-05} because of the following error: RuntimeError('Parent directory temp/lstm_model/models/ does not exist.').
Traceback (most recent call last):
