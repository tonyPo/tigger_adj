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

import importlib
import tigger_package.orchestrator
importlib.reload(tigger_package.orchestrator)
from tigger_package.orchestrator import Orchestrator
from tigger_package.graphsage_unsup import TorchGeoGraphSageUnsup

# COMMAND ----------

# graphsage

folder = "/Workspace/Repos/antonius.b.a.poppe@nl.abnamro.com_old/tigger_adj/data/10k_trxn/"
orchestrator = Orchestrator(folder)
nodes = orchestrator._load_nodes()
edges =  orchestrator._load_edges()

# COMMAND ----------

edges['start'] = edges['start'].astype('int64')
edges['end'] = edges['end'].astype('int64')
edges.describe()

# COMMAND ----------

orchestrator.graphsage = TorchGeoGraphSageUnsup(
    config_dict = orchestrator.config['torch_geo_graphsage'],
    config_path=orchestrator.config_path,
    nodes=nodes, 
    edges=edges,
)

# COMMAND ----------

train_metrics = orchestrator.graphsage.fit()

# COMMAND ----------

folder = "/Workspace/Repos/antonius.b.a.poppe@nl.abnamro.com_old/tigger_adj/data/10k_trxn/"
orchestrator = Orchestrator(folder)
orchestrator.create_graphsage_embedding()
# orchestrator.init_graphsynthesizer('MLP', seed=1)
# orchestrator.train_graphsyntesizer()

# COMMAND ----------

orchestrator.create_synthetic_walks(
            synthesizer=orchestrator.graphsynthesizer, 
            target_cnt=1000
        )

# COMMAND ----------

# import torch
# from torch.utils.data import DataLoader, random_split
# folder = "/Workspace/Repos/antonius.b.a.poppe@nl.abnamro.com_old/tigger_adj/data/reddit/"
# orchestrator = Orchestrator(folder)
# orchestrator.init_graphsynthesizer('mlp', seed=1)
# mlp = orchestrator.graphsynthesizer
# generator1 = torch.Generator().manual_seed(1)
# train_dataset, test_dataset = random_split(
#     mlp.edges,
#     [1-mlp.test_fraction, mlp.test_fraction],
#     generator=generator1
# )

# train_loader = DataLoader(train_dataset, batch_size=mlp.batch_size, shuffle=True)
# val_loader = DataLoader(test_dataset, batch_size=mlp.batch_size, shuffle=False)
# batch = next(iter(train_loader))
# input_batch, output_batch = mlp.prep_batch(batch)

# COMMAND ----------


