#%%
import os
import time
import pickle

import torch
import torch.nn.functional as nnf
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import GraphSAGE
import torch_geometric as tg
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class TorchGeoGraphSageUnsup():
    '''Torch geometric implementation of GraphSage unsupervised.
    '''

    def __init__(self, config_dict, path, nodes, edges) -> None:
        self.config_path = path + config_dict['embed_path']
        self.patience = 300
        for key, val in config_dict.items():
            setattr(self, key, val)
            
        
        os.makedirs(self.config_path, exist_ok=True)
        self.model_path = self.config_path + 'model/'
        os.makedirs(self.model_path, exist_ok=True)
    
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.init_model(nodes, edges)
        
    def init_model(self, nodes, edges):    
        data = self.init_dataset(nodes, edges)
        transform = tg.transforms.RandomLinkSplit(
            num_val=0, num_test=self.num_test, is_undirected=False, 
            add_negative_train_samples=False, neg_sampling_ratio=0
        )
        train_data, _, test_data = transform(data)
    
        self.train_loader = LinkNeighborLoader(
                            train_data,
                            batch_size=self.batch_size,
                            shuffle=True,
                            neg_sampling_ratio=1.0,
                            num_neighbors=self.num_neighbors,
                        )
    
        self.test_loader = LinkNeighborLoader(
                            test_data,
                            batch_size=self.batch_size,
                            shuffle=True,
                            neg_sampling_ratio=1.0,
                            num_neighbors=self.num_neighbors,
                        )
        
        self.model = GraphSAGE(
            data.num_node_features,
            hidden_channels=self.embedding_dim,
            num_layers=self.num_layers,
            dropout = self.dropout
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def init_dataset(self, nodes, edges):
        '''Create torch dataset from nodes and edges'''
        edge_index = torch.tensor(edges[['start', 'end']].values)
        data = tg.data.Data(x = torch.FloatTensor(nodes.values), 
            edge_index=torch.transpose(edge_index, 0, 1))
        
        data = data.to(self.device, 'x', 'edge_index')
        return data
    
    def dataset_epoch(self, loader):
        '''Single loop through the dataset'''
        total_loss = 0
        total_num_nodes = 0
        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            h = self.model(batch.x, batch.edge_index)
            h_src = h[batch.edge_label_index[0]]
            h_dst = h[batch.edge_label_index[1]]
            pred = (h_src * h_dst).sum(dim=-1)
            loss = nnf.binary_cross_entropy_with_logits(pred, batch.edge_label)
            if self.model.train:
                loss.backward()
                self.optimizer.step()
            total_loss += float(loss) * pred.size(0)
            total_num_nodes += pred.size(0)

        return total_loss / total_num_nodes

    def fit(self):
        start = time.time()
        losses = []
        val_losses = []
        best_validation_loss = float('inf')
        early_stopping_counter = 0
        for epoch in range(self.epoch):
            
            self.model.train()
            loss = self.dataset_epoch(self.train_loader)
            
            self.model.eval()
            test_loss = self.dataset_epoch(self.test_loader)
            
            print(f'\rEpoch: {epoch:03d}, Loss: {loss:.4f}, '
                f' Test: {test_loss:.4f}', end='')
            
            losses.append(loss)
            val_losses.append(test_loss)
            
            #check for early stopping

            if test_loss < best_validation_loss:
                best_validation_loss = test_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= self.patience:
                print(f'Early stopping after {epoch + 1} epochs with no improvement in validation loss.')
                break
            
        end = time.time()
        
        train_metrics = {'train_loss': losses, 'val_loss': val_losses}
        if self.verbose > 1:
            print(f"Total training time {end-start:.4f}s")
            self.print_metrics(train_metrics)
            
        return train_metrics
    
    @torch.no_grad()
    def get_embedding(self, nodes, edges):
        data = self.init_dataset(nodes, edges)
        # might need to add a loader for big dataset
        embed = self.model(data.x, data.edge_index)
        df = pd.DataFrame(embed.numpy())
        df = df.reset_index(names='id')
        df.to_parquet(self.config_path + 'embedding.parquet')
        
    def lin_grid_search(self, grid_dict, nodes, edges):
        grid_param = list(grid_dict.keys())[0]
        vals = grid_dict[grid_param]
        res = {}
        
        for val in vals:
            run = {}
            setattr(self, grid_param, val)
            self.init_model(nodes, edges)
            
            train_metrics = self.fit()
            run['grid_param'] = grid_param
            run['grid_value'] = val
            run['loss'] = np.mean(train_metrics['train_loss'][-10:])
            run['val_loss'] = np.mean(train_metrics['val_loss'][-10:])
            run['train_metrics'] = train_metrics
            
            val_name = val[0] if type(val)==list else val
            res[val_name]=run
                   
        pickle.dump(res, open(self.model_path + "grid_search_" + grid_param +".pickle" , "wb"))
        
        if self.verbose>=2:
            self.plot_grid(res)
            
        return res

    def plot_grid(self, res):
        losses = []
        val_losses = []
        for param_val, run in res.items():
            losses.append(run['loss'])
            val_losses.append(run['val_loss'])

        fig, (ax1, ax2) = plt.subplots(1, 2)
        keys = [str(k) for k in res.keys()]
        X_axis = np.arange(len(keys))
        ax1.bar(X_axis+0.2, losses, width=0.3, label='loss')
        ax1.bar(X_axis-0.2, val_losses, width=0.3, label='val_loss')
        ax1.set_xticks(X_axis, keys)

        for k, v in res.items():
            ax2.plot(v['train_metrics']['val_loss'], label=str(k))
        ax2.legend(bbox_to_anchor=(1.25, 1.0))
        print(f"loss: {losses}")
        print(f"val loss: {val_losses}")
    
    def print_metrics(self, train_metrics):

        for name, metric in train_metrics.items():
            plt.plot(metric, label=name )

        plt.legend(bbox_to_anchor=(1.10, 1))
        plt.show()
# %%


