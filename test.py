#%% 
import numpy as np
import pandas as pd
import pickle
import torch
import os
from tigger_package.orchestrator import Orchestrator
from tigger_package.tab_ddpm.train_tab_ddpm import Tab_ddpm_controller

#%%
folder = "data/erdos/"
orchestrator = Orchestrator(folder)
# #%%
# orchestrator.train_node_synthesizer()
# #%%
# orchestrator.sample_node_synthesizer()

# %% TESTDATA
size = 100
node_df = pd.DataFrame([(i/10, (i+1)/10) for i in range(10)]*size,
            columns=['attr1','attr2'])
node_df = orchestrator._load_nodes()
# embed_df = pd.DataFrame([(i/10, (i+1)/10) for i in range(10)]*size,
#             columns=['emb1','emb2'])
# embed_df = pd.DataFrame([(np.random.uniform(), np.random.normal()) for i in range(10*size)],
#             columns=['emb1','emb2'])
embed_df = orchestrator._load_embed()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
embed_norm = pd.DataFrame(scaler.fit_transform(embed_df)[:,:20])
print(f"node shape: {node_df.shape}, embed shape: {embed_norm.shape}")

# %%
ddpm = Tab_ddpm_controller(orchestrator.config_path, orchestrator.config['tab_ddpm'])
hist = ddpm.train(
      embed_norm,
      node_df,
      )
# %%
name = orchestrator.config_path + orchestrator.config['synth_nodes']
ddpm.sample_model(num_samples = 1000, name=name)
# %%
from tigger_package.tools import plot_hist
synth_nodes = pd.read_parquet(folder + 'synth_nodes.parquet')
synth_nodes.name = "ddpm"
x_data = embed_norm.join(node_df, how='inner')
plot_hist(x_data, synth_nodes)


