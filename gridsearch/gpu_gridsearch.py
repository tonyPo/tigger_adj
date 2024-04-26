# Databricks notebook source
# MAGIC %pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 
# MAGIC %pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
# MAGIC %pip install scikit-learn==1.2
# MAGIC %pip install threadpoolctl==3.2.0
# MAGIC %pip install torch_geometric==2.2.0

# COMMAND ----------

import torch_geometric
torch_geometric.__version__

# COMMAND ----------


dbutils.library.restartPython()

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

folder = "/Workspace/Repos/antonius.b.a.poppe@nl.abnamro.com_old/tigger_adj/data/10k_trxn/"
orchestrator = Orchestrator(folder)
nodes = orchestrator._load_nodes()
edges = orchestrator._load_edges()
display(edges)

# COMMAND ----------

folder = "/Workspace/Repos/antonius.b.a.poppe@nl.abnamro.com_old/tigger_adj/data/10k_trxn/"
searcher = gridsearch_graphsage(folder)
# searcher = gridsearch_ddpm(folder)
# searcher = gridsearch_lstm(folder)
# searcher = gridsearch_mlp(folder)
# searcher = gridsearch_bimlp(folder)

# COMMAND ----------

searcher.study.best_params

# COMMAND ----------

4*2048



# COMMAND ----------


