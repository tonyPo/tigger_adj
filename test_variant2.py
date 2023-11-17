#%%
import torch
from tigger_package.orchestrator import Orchestrator
import numpy as np

#%%
enron_folder = "data/enron/"
orchestrator = Orchestrator(enron_folder)


nodes = orchestrator._load_nodes()
embed = orchestrator._load_embed()
edges = orchestrator._load_edges()
embed_nodes = np.concatenate([embed, nodes], axis=1)

# %%

ids = edges['end'].values
ids = ids[:5]
ids

# %%
input = embed_nodes[ids]
input
# input_cluster = self.get_cluster(input)
# %%
from sklearn.cluster import KMeans
kmeans = (KMeans(n_clusters=5, random_state=0,max_iter=10000)
                  .fit(embed_nodes)
        )
# %%
from torch.utils.data import DataLoader, random_split
generator1 = torch.Generator().manual_seed(42)
train_dataset, test_dataset = torch.utils.data.random_split(edges.values, [0.7, 0.3], generator=generator1)

batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for batch in val_loader:
    print(batch)
# %%
import yaml
adj_df = pd.DataFrame([(i, i+1, i/10, (i+1)/10) for i in range(9)],
            columns=['start', 'end', 'edge_attr1', 'edge_attr2'])
adj_df = pd.concat([adj_df]*500)
node_df = pd.DataFrame([(i/10, (i+1)/10) for i in range(10)],
            columns=['attr1','attr2'])
embed_df = pd.DataFrame([(i/10, (i+1)/10) for i in range(10)],
            columns=['emb1','emb2'])
config_dict = yaml.safe_load('''
  synth2_path: None
  test_fraction: 0.3
  batch_size: 128
  num_clusters: 10
  cluster_dim: 4
  epochs: 500
  z_dim: 16
  activation_function_str: 'relu'
  lr: 0.005
  weight_decay: 0.0001
  verbose: 2
''')
graphSynthesizer2 = GraphSynthesizer2(node_df, embed_df, adj_df, "", config_dict)
loss_dict, epoch_loss, val_loss = graphSynthesizer2.fit()
# %%


import torch
import torch.nn.functional as F

class LayerNormalization(nn.Module):
    def __init__(self, features, eps=1e-5):
        super(LayerNormalization, self).__init__()
        self.features = features
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

# Usage:
ln = LayerNormalization(64)  # You can specify the number of features
x = torch.randn(32, 64)  # Sample input tensor
normalized_x = ln(x)

# %%
import os
import pickle
import networkx as nx 
os.chdir('../..')
from tigger_package.variant2 import GraphSynthesizer2
import pandas as pd
import yaml

adj_df = pd.DataFrame([(i, i+1, i/10, (i+1)/10) for i in range(9)],
            columns=['start', 'end', 'edge_attr1', 'edge_attr2'])
adj_df = pd.concat([adj_df]*500)
node_df = pd.DataFrame([(i/10, (i+1)/10) for i in range(10)],
            columns=['attr1','attr2'])
embed_df = pd.DataFrame([(i/10, (i+1)/10) for i in range(10)],
            columns=['emb1','emb2'])
config_dict = yaml.safe_load('''
    synth2_path: None
    test_fraction: 0.3
    batch_size: 128
    num_clusters: 10
    cluster_dim: 4
    epochs: 500
    z_dim: 8
    activation_function_str: 'relu'
    lr: 0.005
    weight_decay: 0.0001
    verbose: 2
''')
graphSynthesizer2 = GraphSynthesizer2(node_df, embed_df, adj_df, "", config_dict)
loss_dict, epoch_loss, val_loss = graphSynthesizer2.fit()