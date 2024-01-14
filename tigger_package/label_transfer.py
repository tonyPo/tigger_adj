#%%
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
if __name__ == "__main__":
    import os
    os.chdir('..')
    print(os.getcwd())
    import yaml
    from tigger_package.orchestrator import Orchestrator
    from pyspark import SparkContext, SparkConf
    from pyspark.sql import SparkSession
    conf = SparkConf().setAppName('appName').setMaster('local')
    sc = SparkContext(conf=conf)
    spark = SparkSession(sc)
from GAE.graph_case_controller import GraphAutoEncoder
from GAE.data_feeder_graphframes import DataFeederGraphFrames
from GAE.position_manager import PositionManager
from sklearn.neighbors import BallTree
#%%

class LabelTransferrer:
    def __init__(self, nodes, edges, synth_nodes, synth_edges, config_dict, config_path, spark=None):
        self.nodes = nodes
        self.edges = edges
        self.synth_nodes = synth_nodes
        self.synth_edges = synth_edges
        self.config_path = config_path
        for key, val in config_dict.items():
            setattr(self, key, val)
        self.spark = spark
        self.node_attr = [c for c in nodes.columns if c not in ['id', self.label_col]]  # node attributes
        
    
    def transfer(self):
        # set default label of synth nodes to false
        self.synth_nodes[self.label_col] = 0
        
        #train and retrieve graphcase embedding
        if self.spark is not None:
            res = self.train_and_calculate_graphcase_embedding_spark()
            gae = res['gae']
            
        else:
            raise NotImplementedError("Only spark variant is implemented")

        if self.verbose>1:
            self.show_plot(res['hist'])
            
        #select embedding for positive labels + add noise
        pos_embed = self.get_positive_embed(res['gc_embed'])
        
    
        #get synthetic embeding
        synth_embed = self.get_synthetic_embed(gae)
        
        # find closest embedding in synthetic graph
        matched_ids = self.get_closests_synth_embed(pos_embed, synth_embed)
        
        # loop per positive embedding
        for orig_node_id, match_id in matched_ids.items():
            orig_node_attr, orig_hub1_in, orig_hub1_out = self.get_hub1_info(orig_node_id, self.edges, self.nodes)
            
            # update 0th layer node.
            self.update_root_node(orig_node_attr, match_id)
        
            # map recon hub1 to nodes and update nodes
            processed_synth_nodes = set()  # keep track of updated synth nodes to avoid double process (once for incoming and once for outgoing)
            self.update_hub1(orig_hub1_in, orig_hub1_out, match_id, processed_synth_nodes)
      
        nodes_path, edges_path = self.save_nodes_and_edges()
        
        return nodes_path, edges_path 

    def save_nodes_and_edges(self):
        nodes_path = self.config_path + 'adj_synth_nodes.parquet'
        edges_path = self.config_path + 'adj_synth_edges.parquet'
        
        self.synth_nodes.to_parquet(nodes_path)
        self.synth_edges.rename(columns={'src': 'start', 'dst': 'end'}).to_parquet(edges_path)
        return nodes_path, edges_path 
    
    def update_root_node(self, orig_node_attr, match_id):
        salt = np.random.normal(loc=0, scale=self.noise_sd, size=orig_node_attr[self.node_attr].shape)
        self.synth_nodes.loc[match_id, self.node_attr] = orig_node_attr[self.node_attr] + salt 
        self.synth_nodes.loc[match_id, self.label_col] = 1
        
         
    def train_and_calculate_graphcase_embedding_spark(self):
        spark_nodes = self.spark.createDataFrame(self.nodes.reset_index())  
        spark_egdes = (self.spark.createDataFrame(self.edges)
            .withColumnRenamed("start", 'src')
            .withColumnRenamed("end", "dst")
            .withColumnRenamed(self.weight_col, "weight")
        )
        graph = (spark_nodes, spark_egdes)
        
        #init graphcase
        gc_verbose = True if self.verbose > 1 else False
        gae = GraphAutoEncoder(
            graph, support_size=self.support_size, 
            dims=self.dims, 
            batch_size=self.batch_size, 
            hub0_feature_with_neighb_dim=self.hub0_dim,
            useBN=self.usebn, 
            verbose=gc_verbose, 
            seed=self.seed, 
            learning_rate=self.learning_rate, 
            act=tf.nn.relu, 
            encoder_labels=self.node_attr,
            data_feeder_cls=DataFeederGraphFrames,
            pos_enc_cls=PositionManager
        )
        hist = gae.fit(epochs=self.epochs, layer_wise=False)
        gc_embed = gae.calculate_embeddings(graph)
        
        return {"gc_embed": gc_embed,
                "hist": hist,
                "gae": gae}
    
    def show_plot(self, hist):
        plt.plot(hist[None].history['loss'], label='loss')
        plt.plot(hist[None].history['val_loss'], label='val_loss')
        plt.legend()
        plt.show()
        
    def get_positive_embed(self, gc_embed):
        # trim embed
        gc_embed = gc_embed[:self.nodes.shape[0]]
        pos_idx = self.nodes[self.nodes[self.label_col]==1].index
        pos_embed = gc_embed[np.isin(gc_embed[:,0], pos_idx)]
        return pos_embed
        
    def get_synthetic_embed(self, gae):
        spark__synth_nodes = self.spark.createDataFrame(self.synth_nodes.reset_index())  
        spark_synth_egdes = (self.spark.createDataFrame(self.synth_edges)
            .withColumnRenamed("start", 'src')
            .withColumnRenamed("end", "dst")
            .withColumnRenamed(self.weight_col, "weight")
        )
        synth_graph = (spark__synth_nodes, spark_synth_egdes)
        synth_embed = gae.calculate_embeddings(synth_graph)
        synth_embed = synth_embed[:self.synth_nodes.shape[0]]
        return synth_embed
        
    def get_closests_synth_embed(self, pos_embed, synth_embed): 
        # init searcher
        searcher = BallTree(synth_embed[:,1:], leaf_size=40)
        dist, matches = searcher.query(pos_embed[:,1:], k=pos_embed.shape[0], 
                                       return_distance=True)
        
        # get the clostes match in an eager manner
        matched = {}
        for i in range(pos_embed.shape[0]):
            lowest_dist_loc = np.where(dist == np.min(dist))  # get matric coordinates of the lowest values
            selected_pos_embed = lowest_dist_loc[0][0]  # the pos embedding with the lowest distance
            matched_node = matches[lowest_dist_loc][0]  # get the embed row number with the lowest distance
            matched_id = synth_embed[matched_node,0]  # get the node id from column 0
            pos_id = pos_embed[selected_pos_embed,0]  # get the node id from column 0
            
            matched[pos_id] = matched_id  # ad results to dict
            
            dist[matches==matched_node] = np.finfo(np.float32).max  # set distance value of the matched node to high to avoid it is selected twice
            dist[selected_pos_embed] = np.finfo(np.float32).max  # set all distances of pos embed to high to avoid mapping the same pos node twice
        
        return matched

    
    def get_hub1_info(self, node_id, edges_df, nodes_df):
        # retrieve the node attribute of the node in the original graph
        orig_node_attr = nodes_df.loc[node_id]
        
        # retrieve the incoming edges and adjacent nodes in the original graph
        in_edges = edges_df[edges_df['end']==node_id]
        in_neighbors = nodes_df[nodes_df.index.isin(in_edges['start'])]
        orig_hub1_in = in_edges.merge(in_neighbors, right_index=True, left_on='start', how='inner')
        orig_hub1_in['id'] = orig_hub1_in['start']
        
        out_edges = edges_df[edges_df['start']==node_id]
        out_neighbors = nodes_df[nodes_df.index.isin(out_edges['end'])]
        orig_hub1_out = out_edges.merge(out_neighbors, right_index=True, left_on='end', how='inner')
        orig_hub1_out['id'] = orig_hub1_out['end']
        
        return orig_node_attr, orig_hub1_in, orig_hub1_out


    def update_hub1(self, orig_hub1_in, orig_hub1_out, match_id, processed_synth_nodes):
        # collect synth hub1 neighborhood 
        _, synth_hub1_in, synth_hub1_out = self.get_hub1_info(match_id, self.synth_edges, self.synth_nodes)       
        edge_cols = [c for c in self.synth_edges.columns if c not in ['start', 'end']]
        
        # match orig on synth edges
        for synth_hub1,  orig_hub1 in zip([synth_hub1_in, synth_hub1_out], [orig_hub1_in, orig_hub1_out]):
            #matches[pos orig node] = pos synth node
            matched = self.get_closests_adjacent_node(orig_hub1.drop(columns=['start', 'end', 'label']), 
                                                      synth_hub1.drop(columns=['start', 'end', 'label']))
        
            for orig_edge_id, synth_edge_id in matched.items():
                # overwrite the synth edge attributes with the original edge attributes
                orig_edge = orig_hub1.loc[orig_edge_id, edge_cols]
                salt = np.random.normal(loc=0, scale=self.noise_sd, size=orig_edge.shape)
                self.synth_edges.loc[synth_edge_id, edge_cols] = orig_edge.values + salt

                # average the node attributes
                orig_node_attr = orig_hub1.loc[orig_edge_id][self.node_attr]
                synth_node_id = synth_hub1.loc[synth_edge_id]['id']
                if synth_node_id in processed_synth_nodes:
                    continue  # node is already updated
                else:
                    processed_synth_nodes.add(synth_node_id)
                    self.synth_nodes.loc[synth_node_id, self.node_attr] = \
                        (orig_node_attr.values + self.synth_nodes.loc[synth_node_id, self.node_attr]) / 2

            #delete unmatched edges
            unmatched_idx = [i for i in synth_hub1.index if i not in matched.values()]
            self.synth_edges = self.synth_edges.drop(index=unmatched_idx)
        
    def get_closests_adjacent_node(self, orig_hub, synth_hub): 
        '''matches the edge+node combinations in the orig_hub to the closest combination in the synth hub
        Returns a dict with key the orig edge id and value the closest synthetic edge id'''
        matched = {}
        number_of_matches = min(orig_hub.shape[0], synth_hub.shape[0])  # determine the maximum number of possible matches
        if number_of_matches==0:  # One of the dataset has degree 0
            return matched
        
        # init searcher
        searcher = BallTree(synth_hub, leaf_size=1)
        dist, matches = searcher.query(orig_hub, k=number_of_matches, 
                                       return_distance=True)
        
        # get the clostes match in an eager manner
        for i in range(number_of_matches):
            lowest_dist_loc = np.where(dist == np.min(dist))  # get matric coordinates of the closesth distance
            selected_orig_edge_pos = lowest_dist_loc[0][0]  # the orig node pos with the lowest distance
            selected_orig_edge_id = orig_hub.iloc[selected_orig_edge_pos].name  # the orig node id with the lowest distance
            matched_edge_pos = matches[lowest_dist_loc][0]  # get the synth node pos with the lowest distance
            matched_edge_id = synth_hub.iloc[matched_edge_pos].name # get the synth node id with the lowest distance
            
            matched[selected_orig_edge_id] = matched_edge_id  # ad results to dict
            
            dist[matches==matched_edge_pos] = np.finfo(np.float32).max  # set distance value of the selected synth node to high to avoid it is selected twice
            dist[selected_orig_edge_pos] = np.finfo(np.float32).max  # set all distances of orig node to high to avoid mapping the same pos node twice
        
        return matched
        
        
        
if __name__ == "__main__":
  
    config_dict = yaml.safe_load('''
        weight_col: 'weight'
        label_col: 'label'
        support_size: [10, 10]
        dims: [2, 24,24,24]
        batch_size: 30
        hub0_dim: 24
        seed: 1
        learning_rate: 0.0002
        usebn: true
        verbose: 2
        epochs: 10
        noise_sd: 0.01
    ''')
    folder = "data/enron/"
    orchestrator = Orchestrator(folder)
    nodes = orchestrator._load_nodes(incl_label_col=True)
    edges = orchestrator._load_edges()

    # add random noise to the synthetic nodes
    synth_nodes = nodes.copy()
    np.random.seed(1)
    synth_nodes = synth_nodes + np.random.normal(loc=0, scale=0.05, size=synth_nodes.shape)
    synth_nodes = synth_nodes.clip(upper=1, lower=0)
    
    # add random noice to the synthetic edge
    synth_edges = edges.copy()
    edge_col = [c for c in edges.columns if c not in ['start', 'end']]
    synth_edges[edge_col] = synth_edges[edge_col] + np.random.normal(loc=0, scale=0.05, size=edges[edge_col].shape)
    synth_edges[edge_col] = synth_edges[edge_col].clip(upper=1, lower=0)
    
    # remove some edges from the positive labels
    pos_nodes = nodes[nodes['label']==1].index
    for pos_node in pos_nodes:
        condition = edges[edges['start']==pos_node].nlargest(3, 'weight').index
        edges = edges.drop(condition)
      
    lt = LabelTransferrer(
        nodes=nodes, 
        edges=edges, 
        synth_nodes=synth_nodes, 
        synth_edges=edges,
        config_dict=config_dict,
        config_path=folder,
        spark=spark)
    res = lt.transfer()
    
# %% prep test data
# path = "unittest/test_data/enron/"
# # save gae
# gae.save_weights(path + "gae")

# # save embedding
# import pickle
# pickle.dump(gc_embed, open(path + "gc_embed", 'wb'))

# # save synth_embed
# pickle.dump(synth_embed, open(path + "synth_embed", 'wb'))





# check if pos node are mapped to synth nodes correctly.

# check if synth nodes do get pos node labels
# check if adjecent nodes are mapped correctly to the synth adjacent nodes
# check if synth incident edge get the pos incident edges.
# check if adjacent nodes get avg values.