#%%
import pandas as pd
import numpy as np
import pickle
import os 
os.chdir('../..')
os.getcwd()
#%% edgelist
output_path = 'data/test_graph/'
edge_attr_cnt = 3
node_attr_cnt = 5
node_cnt = 11
emb_dim = 16

#%% edgelist
edge_dict = {'start': range(node_cnt-1), 'end': range(1,node_cnt)}
for i in range(edge_attr_cnt):
    if i%2==0:
        edge_dict['attr'+str(i)] = range(node_cnt-1)
    else:
        edge_dict['attr'+str(i)] = range(1, node_cnt)
edge_list = pd.DataFrame(edge_dict)
edge_list.to_parquet(output_path + "test_edge_list.parquet")
edge_list
# %%
node_dict = {'id': range(node_cnt)}
for i in range(node_attr_cnt):
    node_dict['attr'+str(i)] = range(node_cnt)

node_df = pd.DataFrame(node_dict)
node_df.to_parquet(output_path + 'test_node_attr.parquet')
# %%
embedding_dict = {}
for n in range(node_cnt):
      embedding_dict[n] = np.ones(16, dtype=float) * n * 2 / node_cnt
      
with open(output_path + 'test_embedding.pickle', 'wb') as handle:
    pickle.dump(embedding_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %%
from inductive_controller import InductiveController

if __name__ == "__main__":
    node_feature_path = output_path + 'test_node_attr.parquet'
    edge_list_path = output_path + "test_edge_list.parquet"
    graphsage_embeddings_path = output_path + 'test_embedding.pickle'
    n_walks=10
    inductiveController = InductiveController(
        node_feature_path=node_feature_path,
        edge_list_path=edge_list_path,
        graphsage_embeddings_path=graphsage_embeddings_path,
        n_walks=n_walks,
        batch_size = 6,
        num_clusters = 5,
        l_w = 7
    )
    seqs = inductiveController.sample_random_Walks()
    seqs = inductiveController.get_X_Y_from_sequences(seqs)
    seqs = inductiveController.data_shuffle(seqs)
    seqs = inductiveController.get_batch(0, 6, seqs)
    
    epoch_wise_loss, loss_dict = inductiveController.train_model()

# %%