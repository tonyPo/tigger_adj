
#%%
import torch
import time
import torch.nn as nn
import torch.nn.functional as nnf
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import BallTree
if __name__ == "__main__":
    import os
    os.chdir('..')
    print(os.getcwd())
    import yaml
    from tigger_package.orchestrator import Orchestrator
#%%

class GraphSynthesizer2(nn.Module):
    
    def __init__(self, nodes, embed, edges, config_path, config_dict):
        super(GraphSynthesizer2, self).__init__()
        self.config_path = config_path + config_dict['synth2_path']
        for key, val in config_dict.items():
            setattr(self, key, val)
        self.device = 'cpu'    
        self.nodes = nodes
        self.embed = embed
        self.embed_nodes = torch.tensor(np.concatenate([embed, nodes], axis=1)).float()
        self.edges = torch.tensor(GraphSynthesizer2.format_edges(edges).values).float().to(self.device)
        self.act_funct = self.set_activation_function(self.activation_function_str)
        self.seed = 4
        
        
        self.init_model()
        self.optimizer = torch.optim.Adagrad(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.cluster_model = self.init_cluster_model()
        self.cluster_labels = torch.tensor(self.cluster_model.labels_).type(torch.long).to(self.device)
        
        #set data to device
        self.embed_nodes = self.embed_nodes.to(self.device)
    @staticmethod  
    def format_edges(edges):
        """ensure that start and end columns are the first two columns"""
        start_cols = ['start', 'end']
        cols = [c for c in edges.columns if c not in start_cols]
        return edges[start_cols+cols]
    
    def prep_batch(self, edge_batch):
        """transforms list of ids to the required input or output format.
        output is numpy array of source embed node attr
        + np array of corresponding cluster ids        
        """
        
        input_ids = edge_batch[:,0].int()
        input = self.embed_nodes[input_ids]
        input_cluster = self.cluster_labels[input_ids]
        
        output_ids = edge_batch[:,1].int()
        output = self.embed_nodes[output_ids]
        output_cluster = self.cluster_labels[output_ids]
        
        edge_attr = edge_batch[:, 2:]
        
        return (input, input_cluster), (output, output_cluster, edge_attr)
    
        
    def fit(self):
        """Main training loop"""
        generator1 = torch.Generator().manual_seed(self.seed)
        train_dataset, test_dataset = random_split(
            self.edges, 
            [1-self.test_fraction, self.test_fraction], 
            generator=generator1
        )

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        loss_epoch = []
        vall_loss_epoch = []
        loss_dict = {}
        for epoch in range(self.epochs):
            loss_epoch.append([])
            vall_loss_epoch.append([])
            
            for batch in train_loader:
                self.train()  # set training flag
                input_batch, output_batch = self.prep_batch(batch)
                #forward pass
                output_hat = self.forward(*input_batch, output_batch[1])
                loss, log_dict = self.calculate_loss(output_batch, output_hat)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 0.1)
                self.optimizer.step()
                
                loss_epoch[-1].append(log_dict['loss'])
                self.update_dict(loss_dict, log_dict)

            for batch in val_loader:   
                self.eval()  # set to evaluation mode
                output_hat = self.forward(*input_batch, output_batch[1])
                loss, log_dict = self.calculate_loss(output_batch, output_hat)
                
                vall_loss_epoch[-1].append(log_dict['loss'])
                self.update_dict(loss_dict, log_dict, prefix='val_')
                
            print(f"\r epoch {epoch+1}/{self.epochs} train loss {np.mean(loss_epoch[-1])}", 
                  f"val loss: {np.mean(vall_loss_epoch[-1])}",end="")
            
        loss_epoch = np.mean(loss_epoch, axis=1)
        vall_loss_epoch = np.mean(vall_loss_epoch, axis=1)
        if self.verbose > 1 :
            self.plot_loss(loss_dict, loss_epoch, vall_loss_epoch)
        
        return (loss_dict, loss_epoch, vall_loss_epoch)
        
    def update_dict(self, target_dict, increment_dict, prefix=""):
        for k in increment_dict.keys():
            if not target_dict.get(prefix + k, False):
                target_dict[prefix + k] = []
            target_dict[prefix + k].append(increment_dict[k]) 
        
        
    def forward(self, input, input_cluster, output_cluster=None):
        """single forward pass to predict the next node and edge"""  
        batch_size = input.shape[0]
        cluster_embed = self.cluster_embeddings(input_cluster)  # retrieve embedding for cluster id
        input_ = torch.cat((input, cluster_embed), -1)
        z = (self.act_funct(self.fc_input1(input_)))
        z = self.act_funct(self.fc_input2(z))
        # z = (self.act_funct(self.fc_input3(z)))
        
        # cluster_logits = self.act_funct(self.fc_clust_distr1(z))
        cluster_logits = self.bn_cluster1(z)
        cluster_logits = self.act_funct(self.fc_clust_distr2(cluster_logits))
        # cluster_logits = nnf.softmax(cluster_logits)
        
        z_mu = self.act_funct(self.fc_z_mu1(z))
        # z_mu = self.fc_z_mu2(z_mu)
        z_mu = z_mu.view(batch_size, self.z_dim, self.num_clusters)
        
        z_std_logits = self.act_funct(self.fc_z_sigma1(z))
        # z_std_logits = self.fc_z_sigma2(z_std_logits)
        z_std_logits = z_std_logits.view(batch_size, self.z_dim, self.num_clusters)
        
        #select cluster
        if self.training:  # select true cluster during training?
            y_clusterid_sampled = output_cluster
        else:
            clust_dist = nnf.softmax(cluster_logits, dim=-1)
            # clust_dist = clust_dist.view(-1,cluster_logits.shape[-1])
            y_clusterid_sampled = torch.multinomial(
                clust_dist, 1, replacement=True,
                generator=torch.Generator(device=self.device).manual_seed(self.seed)
                ).squeeze()
            
        # sample z_out
        y_clusterid_sampled = y_clusterid_sampled.unsqueeze(-1).repeat(1,self.z_dim).unsqueeze(2)  #add dim for cluster_id
        # y_clusterid_sampled = y_clusterid_sampled.type(torch.int64)
        mu = torch.gather(z_mu, 2, y_clusterid_sampled).squeeze(2) 
        std_logits = torch.gather(z_std_logits, 2, y_clusterid_sampled).squeeze(2)
        std_logits = torch.maximum(std_logits, torch.tensor(-20).to(self.device))  # avoid too small number that resolve to zero.
        
        std = torch.exp(std_logits)  # determine std for hidden layer z to gnn
        q = torch.distributions.Normal(mu, std)  # create distribution layer
        z_out = q.rsample()  # sample z for reconstruction of gnn embedding + edge atributes
        
        #reconstruct embed_node
        embed_node_out = self.act_funct(self.fc_output1(z_out))
        embed_node_out = self.act_funct(self.fc_output2(embed_node_out))
        
        # reconstruct cluster embed
        cluster_embed_out = self.act_funct(self.fc_output_cluster1(z_out))
        cluster_embed_out = self.act_funct(self.fc_output_cluster2(cluster_embed_out))
        
        # reconstruct edge_attr
        edge_out = self.act_funct(self.fc_output_edge1(z_out))
        edge_out = self.act_funct(self.fc_output_edge2(edge_out))
        
        return (embed_node_out, cluster_embed_out, edge_out, cluster_logits)
                                               
    def calculate_loss(self, output_batch, output_hat):
        embed_node_out = output_hat[0]
        cluster_embed_out = output_hat[1]
        edge_out = output_hat[2]
        cluster_logits_out = output_hat[3]
        embed_node = output_batch[0]
        cluster_id = output_batch[1]
        edge = output_batch[2]
        
        embed_node_loss = nnf.mse_loss(embed_node_out, embed_node)
        cluster_loss = nnf.cross_entropy(cluster_logits_out, cluster_id)
        edge_loss = nnf.mse_loss(edge_out, edge)
        loss = embed_node_loss + cluster_loss + edge_loss
        
        # logit to zero penalty
        mean_logit = torch.mean(cluster_logits_out, dim=1)
        zero_logit_loss = 0.0001 / mean_logit
        zero_logit_loss = torch.mean(zero_logit_loss)
        loss = loss + zero_logit_loss
        
        loss_dict = {
            'embed_node': embed_node_loss.item(),
            'cluster_loss': cluster_loss.item(),
            'edge_loss': edge_loss.item(),
            'zero_logit_loss': zero_logit_loss.item(),
            'loss': loss.item()
        }
        return (loss, loss_dict)
        
    
    def init_model(self):
        """ initiate model layers"""
        self.cluster_embeddings = nn.Embedding(
            num_embeddings=self.num_clusters,
            embedding_dim=self.cluster_dim,
            max_norm = 1
        ).to(self.device)
        input_dim = self.embed_nodes.shape[1] + self.cluster_dim
        self.fc_input1 = nn.Linear(input_dim, self.z_dim).to(self.device)
        self.bn_fc_input1 = nn.BatchNorm1d(self.z_dim).to(self.device)
        self.fc_input2 = nn.Linear(self.z_dim, self.z_dim).to(self.device)
        self.bn_fc_input2 = nn.BatchNorm1d(self.z_dim).to(self.device)
        self.fc_input3 = nn.Linear(self.z_dim, self.z_dim).to(self.device)
        self.fc_clust_distr1 = nn.Linear(self.z_dim, self.z_dim).to(self.device)
        self.fc_clust_distr2 = nn.Linear(self.z_dim, self.num_clusters).to(self.device)
        self.bn_cluster1 = nn.LayerNorm(self.z_dim).to(self.device)
        self.fc_z_mu1 = nn.Linear(self.z_dim, self.z_dim*self.num_clusters).to(self.device)
        self.fc_z_mu2 = nn.Linear(self.z_dim*self.num_clusters, self.z_dim*self.num_clusters).to(self.device)
        self.fc_z_sigma1 = nn.Linear(self.z_dim, self.z_dim*self.num_clusters).to(self.device)
        self.fc_z_sigma2 = nn.Linear(self.z_dim*self.num_clusters, self.z_dim*self.num_clusters).to(self.device)
        self.fc_output1 = nn.Linear(self.z_dim, self.z_dim).to(self.device)
        self.fc_output2 = nn.Linear(self.z_dim, self.embed_nodes.shape[1]).to(self.device)
        self.fc_output_cluster1 = nn.Linear(self.z_dim, self.z_dim).to(self.device)
        self.fc_output_cluster2 = nn.Linear(self.z_dim, self.cluster_dim).to(self.device)
 
        edge_dim = self.edges.shape[1] - 2  # excl start and end columns
        self.fc_output_edge1 = nn.Linear(self.z_dim, self.z_dim).to(self.device)
        self.fc_output_edge2 = nn.Linear(self.z_dim, edge_dim).to(self.device)
  

    def init_cluster_model(self): 
        """train kmeans and determine cluster labels"""
        kmeans = (KMeans(n_clusters=self.num_clusters, random_state=0,max_iter=10000)
                  .fit(self.embed_nodes)
        )
        return kmeans

    def set_activation_function(self, activation_function_str):
        activation_functions = {
            'relu': nnf.relu,
            'sigmoid': nnf.sigmoid,
            'tanh': nnf.tanh,
            'softmax': nnf.softmax
        }
        return activation_functions[activation_function_str]
    
    def plot_loss(self, loss_dict, loss_epoch, vall_loss_epoch):
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax1.plot(loss_epoch, label='loss')
        ax1.plot(vall_loss_epoch, label='val_loss')
        ax1.set_yscale("log")
        ax1.legend()
        
        for k,v in loss_dict.items():
            if k.startswith("val_"):
                ax2.plot(v, label=k)
        ax2.legend(bbox_to_anchor=(1.5, 1.))
        ax2.set_yscale("log")
        plt.show()
       
    def create_synthetic_walks(self, synth_nodes_df, target_cnt, map_real_time=False):
        """creates a list of tuples. Every tuple has (start_id, end_id, list(edge_attr))"""
        node_ids = torch.tensor(synth_nodes_df.index.values).int().to(self.device)
        synth_nodes = torch.tensor(synth_nodes_df.values).float().to(self.device)
        searcher = BallTree(synth_nodes, leaf_size=40)
        node_loader = DataLoader(node_ids, batch_size=self.batch_size, shuffle=True)
        self.eval()
        res = []
        
        while len(res) < target_cnt:
            for node_batch in node_loader:
                embed_node = synth_nodes[node_batch,:]
                clusters = torch.tensor(self.cluster_model.predict(embed_node)).to(self.device)
                output = self.forward(embed_node, clusters)
                end = self.map_to_nodes(searcher, output[0])
                edge_attr = output[2].detach().numpy()
                batch_res = [(s,e,list(a)) for s,e,a in zip(node_batch.detach().numpy(), end, edge_attr)]
                res = res + batch_res
                
        return res
                
    def map_to_nodes(self, searcher, embed_node):
        """ maps the infered embed_node vector to the closter synth node"""   
        _, mapped_id = searcher.query(torch.squeeze(embed_node, 1).tolist(), k=1)
        return np.squeeze(mapped_id, 1).tolist()        


if __name__ == "__main__":
    # folder = "data/enron/"
    # orchestrator = Orchestrator(folder)
    # nodes = orchestrator._load_nodes()
    # embed = orchestrator._load_embed()
    # edges = orchestrator._load_edges() 
    # config_dict = orchestrator.config['GraphSynthesizer2']
    # graphSynthesizer2 = GraphSynthesizer2(nodes, embed, edges, folder, config_dict)
    # start = time.time()
    # loss_dict, epoch_loss, val_loss = graphSynthesizer2.fit()
    # print(f"time: {time.time()-start:.2f}")
    # graphSynthesizer2.create_synthetic_walks(orchestrator._load_synthetic_nodes(), 1000)

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
        epochs: 5000
        z_dim: 8
        activation_function_str: 'relu'
        lr: 0.005
        weight_decay: 0.0001
        verbose: 2
    ''')
    graphSynthesizer2 = GraphSynthesizer2(node_df, embed_df, adj_df, "", config_dict)
    loss_dict, epoch_loss, val_loss = graphSynthesizer2.fit()
    
    synth_nodes = embed_df.merge(node_df,left_index=True, right_index=True )
    
    synth_edges = graphSynthesizer2.create_synthetic_walks(synth_nodes, 30)
    synth_edges
    
# %%