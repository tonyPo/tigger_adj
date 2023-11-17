import unittest
import os
import sys
import pandas as pd
import numpy as np
print(f"current idr: {os.getcwd()}")
sys.path.append(os.getcwd())

from tigger_package.variant2 import GraphSynthesizer2
from tigger_package.orchestrator import Orchestrator

class Variant2(unittest.TestCase):
    BASE_FOLDER = 'unittest/test_data/test_graph/'
    
    def setUp(self):
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
        # loss_dict, epoch_loss, val_loss = graphSynthesizer2.fit()