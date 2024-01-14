#%%
import pickle
import os
import pandas as pd
import numpy as np
import yaml
import tensorflow as tf
if __name__ == "__main__":
    import os
    os.chdir('..')
    print(os.getcwd())
    import yaml
from sklearn.preprocessing import MinMaxScaler
from tigger_package.graphsage_unsup import TorchGeoGraphSageUnsup
from tigger_package.graph_generator import GraphGenerator
from tigger_package.flownet import FlowNet
from tigger_package.inductive_controller import InductiveController
from tigger_package.mlp_edge_synthesizer import MLPEdgeSynthsizer
from tigger_package.bimlp_edge_synthesizer import BiMLPEdgeSynthesizer
from tigger_package.tab_ddpm.train_tab_ddpm import Tab_ddpm_controller
from tigger_package.label_transfer import LabelTransferrer
#%%

class Orchestrator():
    def __init__(self, config_path):
        with open(config_path + "config.yaml", 'r') as file:
            config_dict = yaml.safe_load(file)
        self.config = config_dict
        self.config_path = config_path
        if self.config['node_synthesizer_class'] == 'tab_ddpm':
            self.node_synthesizer_class = Tab_ddpm_controller
        else:
            self.node_synthesizer_class = FlowNet
        
        self.node_synthesizer = None
        self.graphsage = None
        self.lstm_controller = None
        self.graphsynthesizer = None
        self.spark = None
        
    
    def train_node_synthesizer(self):
        with tf.device('/CPU:0'):
            node = self._load_nodes()
            embed = self._load_normalized_embed()
            self.node_synthesizer = self.node_synthesizer_class(
                embed = embed,
                nodes = node,
                config_path=self.config_path,
                config_dict=self.config[self.config['node_synthesizer_class']])
            hist = self.node_synthesizer.fit()
        return hist
    
    def sample_node_synthesizer(self, model_name=None):
        name = self.config_path + self.config['synth_nodes']
        if model_name:
            self.node_synthesizer = self.node_synthesizer_class(
                config_path=self.config_path,
                config_dict=self.config[self.config['node_synthesizer_class']])
            self.node_synthesizer.load_model(model_name)
        self.node_synthesizer.sample_model(self.config['target_node_count'], name)
        
    def lin_grid_search_flownet(self, grid_dict):
        with tf.device('/CPU:0'):
            node = self._load_nodes()
            embed = self._load_normalized_embed()
            if not self.flownet:
                self.flownet = FlowNet(
                    config_path=self.config_path,
                    config_dict=self.config[self.config['node_synthesizer_class']])
            res = self.flownet.lin_grid_search(grid_dict, embed, node)
        return res
    
    def create_graphsage_embedding(self):
        nodes = self._load_nodes()
        edges =  self._load_edges()
        self.graphsage = TorchGeoGraphSageUnsup(
            config_dict = self.config['torch_geo_graphsage'],
            config_path=self.config_path,
            nodes=nodes, 
            edges=edges,
        )
        train_metrics = self.graphsage.fit()
        self.graphsage.get_embedding(nodes, edges)  
        return train_metrics
        
        
    def lin_grid_search_graphsage(self, grid_dict):
        nodes = self._load_nodes()
        edges =  self._load_edges()
        if not self.graphsage:
            self.graphsage = TorchGeoGraphSageUnsup(
                config_dict = self.config['torch_geo_graphsage'],
                path=self.config_path,
                nodes=nodes, 
                edges=edges,
        )   
        res = self.graphsage.lin_grid_search(grid_dict, nodes, edges)
        return res
          
    def train_lstm(self):
        if not self.lstm_controller:
            self.init_lstm()
        loss_dict = self.lstm_controller.fit()
        return (loss_dict)
    
    def train_graphsyntesizer(self):
        if not self.graphsynthesizer:
            self.init_graphsynthesizer()
        loss_dict = self.graphsynthesizer.fit()
        return loss_dict
    
    def init_graphsynthesizer(self, edge_synthesizer= 'MLP', seed=None):
        if edge_synthesizer == 'MLP':
            edge_synthesizer_class = MLPEdgeSynthsizer
            config_dict = self.config['MLPEdgeSynthesizer']
        elif edge_synthesizer == 'LSTM':
            edge_synthesizer_class = InductiveController
            config_dict = self.config['lstm']
        else:
            edge_synthesizer_class = BiMLPEdgeSynthesizer
            config_dict = self.config['BIMLPEdgeSynthesizer']
        
        #set seed
        if seed is not None:
            config_dict['seed'] = seed
            
        self.graphsynthesizer = edge_synthesizer_class(
            nodes=self._load_nodes(),
            embed=self._load_normalized_embed(),
            edges=self._load_edges(),
            path="",
            config_dict=config_dict
        )
       
    def lin_grid_search_lstm(self, grid_dict):
        if not self.lstm_controller:
            self.init_lstm()
        res = self.lstm_controller.lin_grid_search(grid_dict)
        return res 
    
    def create_synthetic_walks(self, synthesizer, target_cnt, synth_node_file_name=None, map_real_time=True):
        generated_nodes = self._load_synthetic_nodes(synth_node_file_name)
        self.synth_walks = synthesizer.create_synthetic_walks(generated_nodes, target_cnt=target_cnt, map_real_time=map_real_time)
        pickle.dump(self.synth_walks, open(self.config_path + self.config['synth_walks'], "wb"))
     
    def generate_synth_graph(self, synth_nodes_name=None):
        results_dir = self.config_path + self.config['synth_graph_dir']
        edge_cols = [c for c in self._load_edges().columns if c not in ['start', 'end']]         
        node_cols = self._load_nodes().columns
        
        graph_generator = GraphGenerator(
            results_dir = results_dir , 
            node_cols = node_cols,
            edge_cols = edge_cols
        )
        
        graph_generator.generate_graph(
            nodes=self._load_synthetic_nodes(synth_nodes_name),
            edges=self._load_synth_walks(),
            target_edge_count=self._load_edges().shape[0]
        )
           
    def transfer_labels(self, spark=None):
        if spark is None:
            self.spark = self._init_spark()
            
        #update dict
        self.config['transfer']['label_col'] = self.config['label_col']
        synth_edges = (self._load_synth_graph_edges()
                       .rename(columns={'src': 'start', 'dst': 'end'})
        )
        
        lt = LabelTransferrer(
            nodes=self._load_nodes(incl_label_col=True), 
            edges=self._load_edges(), 
            synth_nodes=self._load_synth_graph_nodes(),
            synth_edges=synth_edges,
            config_dict=self.config['transfer'],
            config_path=self.config_path,
            spark=self.spark)
        nodes_path, edges_path = lt.transfer()
        self.spark.stop()
        return (nodes_path, edges_path)
                                                
    # -- private methodes
    
    def _init_spark(self):
        from pyspark import SparkContext, SparkConf
        from pyspark.sql import SparkSession
        conf = SparkConf().setAppName('appName').setMaster('local')
        sc = SparkContext(conf=conf)
        spark = SparkSession(sc)
        return spark
    
    def _load_edges(self):
        return pd.read_parquet(self.config_path + self.config['edges_path'])  

    def _load_nodes(self, incl_label_col=False):
        # assume id column
        nodes = pd.read_parquet(self.config_path + self.config['nodes_path'])  
        nodes = nodes.sort_values('id').set_index('id')
        
        if not incl_label_col and self.config.get('label_col') is not None:
            nodes = nodes.drop(columns=self.config.get('label_col'))
        return nodes
    
    def _load_normalized_embed(self):
        # normalize the embedding based on overall min and mx value to keep the relative distance.
        embed = self._load_embed()
        
        #normalised between 0 and 1
        max_val = embed.max().max()
        min_val = embed.min().min()
        
        embed_norm = embed - min_val
        embed_norm = embed_norm / (max_val - min_val)

        return embed_norm
    
    def _load_embed(self):
        embed_path = self.config_path + self.config['embed_path']
        try:  # incase embed is stored as dict
            node_embeddings = pickle.load(open(embed_path,"rb"))                   
            node_embedding_df = pd.DataFrame.from_dict(node_embeddings, orient='index')
        except:
            node_embedding_df = pd.read_parquet(embed_path)
            node_embedding_df = node_embedding_df.sort_values('id').set_index('id')
        
        return node_embedding_df
        
    def _load_synthetic_nodes(self, name=None):
        """loads the synth node embed_ + attrib from flownet"""
        if name is None:
            path = self.config_path + self.config['synth_nodes']
        else:
            path = name
        
        synth_nodes = pd.read_parquet(path)
        return synth_nodes
    
    def _load_synth_walks(self):
        path = self.config_path + self.config['synth_walks']
        synth_walk = pickle.load(open(path, ("rb"))) 
        return synth_walk   
                
    
    def _load_synthetic_graph_nodes(self, name=None):
        path = self.config_path + self.config['synth_graph_dir'] + 'node_attributes.parquet'
        return pd.read_parquet(path)
    
    def _load_synth_graph_edges(self):
        path = self.config_path + self.config['synth_graph_dir'] + 'adjacency.parquet'  
        synth_edges = pd.read_parquet(path)
        float32_cols = list(synth_edges.select_dtypes(include='float32'))
        synth_edges[float32_cols] = synth_edges[float32_cols].astype('float64')
        return synth_edges
    
    def _load_synth_graph_nodes(self):
        path = self.config_path + self.config['synth_graph_dir'] + 'node_attributes.parquet'  
        synth_nodes = pd.read_parquet(path)
        synth_nodes.index.rename("id", inplace=True)
        float32_cols = list(synth_nodes.select_dtypes(include='float32'))
        synth_nodes[float32_cols] = synth_nodes[float32_cols].astype('float64')
        return synth_nodes
        
    
 
if __name__ == "__main__":
    folder = "data/enron/"
    orchestrator = Orchestrator(folder)
    # nodes=orchestrator._load_nodes(incl_label_col=True)
    # synth_nodes=orchestrator._load_synth_graph_nodes()
    orchestrator.transfer_labels()
    
#%%       