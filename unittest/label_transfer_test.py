import unittest
import os
import sys
import pickle
import pandas as pd
import numpy as np
import yaml
print(f"current idr: {os.getcwd()}")
sys.path.append(os.getcwd())

from tigger_package.orchestrator import Orchestrator
from tigger_package.label_transfer import LabelTransferrer



class LabelTransferrerTest(unittest.TestCase):
    spark = None
    
    @classmethod
    def setUpClass(cls):
        # init spark
        from pyspark import SparkContext, SparkConf
        from pyspark.sql import SparkSession
        conf = SparkConf().setAppName('appName').setMaster('local')
        sc = SparkContext(conf=conf)
        LabelTransferrerTest.spark = SparkSession(sc)
        
    
    def setUp(self):        
        # load node and edge data
        self.base_folder = 'unittest/test_data/enron/'
        nodes = pd.read_parquet(self.base_folder + 'enron_nodes.parquet')  
        self.nodes = nodes.sort_values('id').set_index('id')
        self.edges = pd.read_parquet(self.base_folder + 'enron_edges.parquet')  
        
        # add random noise to the synthetic nodes
        synth_nodes = self.nodes.copy()
        np.random.seed(1)
        synth_nodes = synth_nodes + np.random.normal(loc=0, scale=0.01, size=synth_nodes.shape)
        self.synth_nodes = synth_nodes.clip(upper=1, lower=0)
        
        # add random noice to the synthetic edge
        self.synth_edges = self.edges.copy()
        edge_col = [c for c in self.edges.columns if c not in ['start', 'end']]
        self.synth_edges[edge_col] = self.synth_edges[edge_col] + np.random.normal(loc=0, scale=0.05, size=self.edges[edge_col].shape)
        self.synth_edges[edge_col] = self.synth_edges[edge_col].clip(upper=1, lower=0)
        
        # remove some edges from the positive labels
        pos_nodes = self.nodes[self.nodes['label']==1].index
        for pos_node in pos_nodes:
            condition = self.edges[self.edges['start']==pos_node].nlargest(3, 'weight').index
            self.edges = self.edges.drop(condition)
            
        # config data
        config_dict = yaml.safe_load('''
            weight_col: 'weight'
            label_col: 'label'
            support_size: [4, 4]
            dims: [2, 6, 6, 6]
            batch_size: 15
            hub0_dim: 6
            seed: 1
            learning_rate: 0.001
            usebn: true
            verbose: 2
            epochs: 10
        ''')
        
        # init lavel transferrer
        self.labelTransferrer = LabelTransferrer(
            nodes=self.nodes, 
            edges=self.edges, 
            synth_nodes=self.synth_nodes, 
            synth_edges=self.synth_edges,
            config_dict=config_dict,
            spark=LabelTransferrerTest.spark)
        
    
    def test_embed(self):
        # train and get graphcase embedding
        res = self.labelTransferrer.train_and_calculate_graphcase_embedding_spark()
        gc_embed = res['gc_embed']
        
        #check that embed has the same number of rows as the original nodes matrix
        self.assertEqual(self.nodes.shape[0], gc_embed.shape[0], 
                         msg=f"The number of rows in the embed matrixs {gc_embed.shape[0]} deviates from the number of nodes {self.nodes.shape[0]} ")
        
    def test_pos_embed(self):
        gc_embed = pickle.load(open(self.base_folder + 'gc_embed', 'rb'))
        #retrieve the positive nodes (8, 78, 121)
        pos_embed = self.labelTransferrer.get_positive_embed(gc_embed)
        
        # check that the nubmer of positive nodes
        self.assertEqual(pos_embed.shape[0], 3, 
                         msg=f"The number of rows in the pos embed matrixs {pos_embed.shape[0]} deviates from the expected number (3)")
        
        pos_idx = pos_embed[:,0]
        false_matches = [p for p in pos_idx if p not in [8,78,121]]
        self.assertEqual(len(false_matches), 0, f"There are some false positive ids {false_matches}.")
       
    
    def test_synth_embed(self):
        res = self.labelTransferrer.train_and_calculate_graphcase_embedding_spark()
        gae = res['gae']
        synth_embed = self.labelTransferrer.get_synthetic_embed(gae)
            #check that embed has the same number of rows as the original nodes matrix
        self.assertEqual(self.nodes.shape[0], synth_embed.shape[0], 
                         msg=f"The number of rows in the embed matrixs {synth_embed.shape[0]} deviates from the number of nodes {self.synth_nodes.shape[0]} ")

      
    def test_get_closests_synth_embed(self):
        gc_embed = pickle.load(open(self.base_folder + 'gc_embed', 'rb'))
        pos_embed = self.labelTransferrer.get_positive_embed(gc_embed)
        synth_embed = pickle.load(open(self.base_folder + 'synth_embed', 'rb'))
        matched_ids = self.labelTransferrer.get_closests_synth_embed(pos_embed, synth_embed)  
       
       # check the matched ids count
        self.assertEqual(len(matched_ids), 3, 
                         msg=f"The number of matches {len(matched_ids)} deviates from the number of positive nodes 3")
        
        # check the matched keys
        keys_correct = [k for k in matched_ids.keys() if k not in [8,78,121]]
        self.assertEqual(len(keys_correct), 0, 
                         msg=f"The matches dict has invalid keys {keys_correct}")
        
        # check values (node 121 is mapped to 146)
        values_incorrect = [v for v in matched_ids.values() if v not in [8,78,146]]
        self.assertEqual(len(values_incorrect), 0, 
                         msg=f"The matches dict has invalid values {values_incorrect}")
        
    def test_get_hub1_info(self):
        orig_node_attr, orig_hub1_in, orig_hub1_out = self.labelTransferrer.get_hub1_info(8, self.edges, self.nodes)
        
        node8_attr = self.nodes.loc[8, ['attr_received_size', 'attr_cnt_to', 'attr_cnt_cc', 'attr_sent_size', 'attr_cnt_send', 'label']]
         
        # check node attributes 
        delta = np.sum(orig_node_attr - node8_attr) 
        self.assertAlmostEqual(delta, 0, msg=f"The node attributes are unequal")
        
        # check incoming edges
        inc_edges = self.edges[self.edges['end']==8]
        edge_cols = self.edges.columns
        edge_check_sum_soll = inc_edges[edge_cols].sum().sum()
        edge_check_sum_ist = orig_hub1_in[edge_cols].sum().sum()
        self.assertAlmostEqual(edge_check_sum_soll, edge_check_sum_ist, msg=f"The inedge attributes are unequal")
        self.assertAlmostEqual(inc_edges.shape[0], orig_hub1_in.shape[0], 
                               msg=f"The in count of edge { orig_hub1_in.shape[0]} devaited from expected {inc_edges.shape[0]}")
        
        # check incming adjacent nodes
        node_idx = self.edges[self.edges['end']==8]['start']
        adj_in_nodes = self.nodes.loc[node_idx].drop(columns='label')
        node_check_sum_soll = adj_in_nodes.sum().sum()
        node_cols = [c for c in orig_hub1_in.columns if c not in list(edge_cols) + ['label', 'id']]
        node_check_sum_ist = orig_hub1_in[node_cols].sum().sum()
        self.assertAlmostEqual(node_check_sum_soll, node_check_sum_ist, msg=f"The adjacent in nodes attributes are unequal")
        self.assertAlmostEqual(adj_in_nodes.shape[0], orig_hub1_in.shape[0], 
                               msg=f"The in count of nodes { orig_hub1_in.shape[0]} devaited from expected {adj_in_nodes.shape[0]}")
        
        
         # check outgoing edges
        outg_edges = self.edges[self.edges['start']==8]
        edge_check_sum_soll = outg_edges[edge_cols].sum().sum()
        edge_check_sum_ist = orig_hub1_out[edge_cols].sum().sum()
        self.assertAlmostEqual(edge_check_sum_soll, edge_check_sum_ist, msg=f"The outedge attributes are unequal")
        self.assertAlmostEqual(outg_edges.shape[0], orig_hub1_out.shape[0], 
                               msg=f"The count of outgoing edges { orig_hub1_out.shape[0]} devaites from expected {outg_edges.shape[0]}")
        
        
        # check outgoing adjacent nodes
        node_idx = self.edges[self.edges['start']==8]['end']
        adj_out_nodes = self.nodes.loc[node_idx].drop(columns='label')
        node_check_sum_soll = adj_out_nodes.sum().sum()
        node_check_sum_ist = orig_hub1_out[node_cols].sum().sum()
        self.assertAlmostEqual(node_check_sum_soll, node_check_sum_ist, msg=f"The adjacent outgoing nodes attributes are unequal")
        self.assertAlmostEqual(adj_out_nodes.shape[0], orig_hub1_out.shape[0], 
                               msg=f"The count of adjacent outgoing nodes { orig_hub1_out.shape[0]} devaites from expected {adj_out_nodes.shape[0]}")
        
        
    def test_update_root_node(self):
        orig_node_attr, orig_hub1_in, orig_hub1_out = self.labelTransferrer.get_hub1_info(8, self.edges, self.nodes)
        self.labelTransferrer.update_root_node(orig_node_attr, 8)
        
        # Check if synth node index 8 has label 1
        self.assertEqual(self.synth_nodes.loc[8, 'label'], 1, 
                         msg=f"The matched synth node has not label = 1")
        
        #check node attributes
        check_sum_soll = self.nodes.loc[8].sum()
        check_sum_ist = self.synth_nodes.loc[8].sum()
        self.assertAlmostEqual(check_sum_soll, check_sum_ist, 
                         msg=f"The matched synth node has not the same attribute values")
        
    def test_get_closests_adjacent_node(self):
        orig_node_attr, orig_hub1_in, orig_hub1_out = self.labelTransferrer.get_hub1_info(121, self.edges, self.nodes)
        synth_node_attr, synth_hub1_in, synth_hub1_out = self.labelTransferrer.get_hub1_info(146, self.synth_edges, self.synth_nodes)
        
        matched = self.labelTransferrer.get_closests_adjacent_node(orig_hub1_in.drop(columns=['start', 'end', 'label']), 
                                                      synth_hub1_in.drop(columns=['start', 'end', 'label']))
        print(matched)
        
        #check if size of dict is equal to the min number of row of either synth or orig.
        expected_count = min(orig_hub1_in.shape[0], synth_hub1_in.shape[0])
        self.assertEqual(len(matched), expected_count, 
                         msg=f"There are {len(matched)} matches, expected {expected_count}")
        
        #check matches
        org = orig_hub1_in.drop(columns=['start', 'end', 'label'])
        synths = synth_hub1_in.drop(columns=['start', 'end', 'label'])
                                                  
        for orig_id, match_id in matched.items():
            min_delta = min(np.mean((org.loc[orig_id] - synths)**2, axis=1))
            delta = np.mean((org.loc[orig_id] - synths.loc[match_id])**2)
            self.assertAlmostEqual(min_delta, delta, 
                         msg=f"Not the closted edge is matched for {org.loc[orig_id].name}")
            synths.loc[match_id] = np.finfo(np.float32).max
            
            
            
    def test_update_hub1(self):
        _, orig_hub1_in, orig_hub1_out = self.labelTransferrer.get_hub1_info(121, self.edges, self.nodes)
        _, synth_hub1_in, synth_hub1_out = self.labelTransferrer.get_hub1_info(146, self.synth_edges, self.synth_nodes)
        in_matched = self.labelTransferrer.get_closests_adjacent_node(orig_hub1_in.drop(columns=['start', 'end', 'label']), 
                                                      synth_hub1_in.drop(columns=['start', 'end', 'label']))
        
        before_synt_nodes = self.synth_nodes.copy()
        processed_synth_nodes = set()
        self.labelTransferrer.update_hub1(orig_hub1_in, orig_hub1_out, 146, processed_synth_nodes)
        
        # check the in matched nodes
        cols = self.labelTransferrer.node_attr
        edge_cols = [c for c in self.edges.columns if c not in ['start', 'end']]
        for orig_edge_id, synth_edge_id in in_matched.items():
            synth_node_id = synth_hub1_in.loc[synth_edge_id]['start']
            orig_node_id = orig_hub1_in.loc[orig_edge_id]['start']
            expected = sum((self.nodes.loc[orig_node_id, cols] + before_synt_nodes.loc[synth_node_id, cols]) / 2)
            observed = sum(self.synth_nodes.loc[synth_node_id, cols])
            self.assertAlmostEqual(expected, observed, 
                         msg=f"Attribute values for synth node {synth_node_id} deviate")
            
        # check the matched in edges
        for orig_edge_id, synth_edge_id in in_matched.items():
            expected = sum(self.edges.loc[orig_edge_id, edge_cols])
            observed = sum(self.synth_edges.loc[synth_edge_id, edge_cols])
            self.assertAlmostEqual(expected, observed, 
                         msg=f"Attribute values for edge {orig_edge_id} deviate")
            
        # check if unnecessary incoming edges are removed 
        expected = orig_hub1_in.shape[0]  #in degree
        observed = self.labelTransferrer.synth_edges[self.synth_edges['end']==146].shape[0]
        self.assertAlmostEqual(expected, observed, 
                         msg=f"indegree for synth node 146 is larger {observed} than expected {expected}")
            
        # check the out nodes
        out_matched = self.labelTransferrer.get_closests_adjacent_node(orig_hub1_out.drop(columns=['start', 'end', 'label']), 
                                                      synth_hub1_out.drop(columns=['start', 'end', 'label']))
        processed_synth_nodes = [int(n) for n in list(self.synth_edges.loc[list(in_matched.values())]['start'])]
        for orig_edge_id, synth_edge_id in out_matched.items():
            synth_node_id = synth_hub1_out.loc[synth_edge_id]['end']
            if int(synth_node_id) in processed_synth_nodes:
                continue  # skip node that are alrady set in the matched incoming edges
            orig_node_id = orig_hub1_out.loc[orig_edge_id]['end']
            expected = sum((self.nodes.loc[orig_node_id, cols] + before_synt_nodes.loc[synth_node_id, cols]) / 2)
            observed = sum(self.synth_nodes.loc[synth_node_id, cols])
            self.assertAlmostEqual(expected, observed, 
                         msg=f"Attribute values for synth node {synth_node_id} deviate")
            
        # check the matched out edges
        for orig_edge_id, synth_edge_id in out_matched.items():
            expected = sum(self.edges.loc[orig_edge_id, edge_cols])
            observed = sum(self.labelTransferrer.synth_edges.loc[synth_edge_id, edge_cols])
            self.assertAlmostEqual(expected, observed, 
                         msg=f"Attribute values for edge {orig_edge_id} deviate")
            
        # check if outgoing edges need to be remove for
        expected = orig_hub1_out.shape[0]  #out degree
        observed = self.labelTransferrer.synth_edges[self.synth_edges['start']==146].shape[0]
        self.assertAlmostEqual(expected, observed, 
                         msg=f"outdegree for synth node 146 is larger {observed} than expected {expected}")
        

            
        
        
        
        
        
        
    
    @classmethod
    def tearDownClass(cls):
        LabelTransferrerTest.spark.stop()

 