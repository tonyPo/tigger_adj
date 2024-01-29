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


