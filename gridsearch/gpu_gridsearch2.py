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
# searcher = gridsearch_lstm(folder)
# searcher = gridsearch_mlp(folder)
searcher = gridsearch_bimlp(folder)
#147 in 5:20 -> 2,1 min per epoch

# COMMAND ----------

# MAGIC %md
# MAGIC - dropout 0.05
# MAGIC - lr = 0.003
# MAGIC - timestep 2500
# MAGIC - step 50K (epochs?)
# MAGIC - weigh decay = 0.0001

# COMMAND ----------

searcher.study.best_params

# COMMAND ----------

[I 2024-01-29 16:16:28,493] Trial 0 finished with value: 4.5832727703827105 and parameters: {'num_clusters': 14, 'cluster_dim': 7, 'epochs': 284, 'lr': 2.9781698486625904e-05, 'weight_decay': 6.08649815139304e-05, 'kl_weight': 1.4129720612185052e-05}. Best is trial 0 with value: 4.5832727703827105.

[I 2024-01-30 03:09:32,971] Trial 1 finished with value: 3.637095402434435 and parameters: {'num_clusters': 9, 'cluster_dim': 5, 'epochs': 434, 'lr': 3.1896785188299403e-05, 'weight_decay': 3.958304262036784e-05, 'kl_weight': 1.4335158342485605e-07}. Best is trial 1 with value: 3.637095402434435.

[I 2024-01-30 09:27:24,318] Trial 2 finished with value: 4.176707463259203 and parameters: {'num_clusters': 15, 'cluster_dim': 10, 'epochs': 209, 'lr': 0.00028581541391794964, 'weight_decay': 2.5212124897264015e-05, 'kl_weight': 1.2792001652348734e-06}. Best is trial 1 with value: 3.637095402434435.
