#%%
import pickle
import pandas as pd
if __name__ == "__main__":
    import os
    os.chdir('..')
    print(os.getcwd())
    import yaml


#%%

class EdgeCategoricalTransformer:
    ''' transforms the floating values of the synthetic edges in the synth walks into binairy and categorical values
    with edge value int.
    '''
    
    def __init__(self, synth_edges, config_dict, config_path, spark=None):
        self.synth_edges = synth_edges
        self.config_path = config_path
        for key, val in config_dict.items():
            setattr(self, key, val)
            
    def transform(self):
        self.transform_binairy()
        self.transform_categorical()
        self.synth_edges.to_parquet(self.config_path + "adjacency.parquet")
        return self.synth_edges

    def transform_binairy(self):
        for c in self.bin_cols:
            self.synth_edges[c] = (self.synth_edges[c] >= 0.5).astype('int')
            one_hot_encoded = pd.get_dummies(max_cols, prefix='category').astype('int')

    def transform_categorical(self):
        for cs in self.cat_cols:
            max_cols = self.synth_edges[cs].idxmax(axis=1)  # determine col name having the max value
            one_hot_encoded = pd.get_dummies(max_cols, prefix='cat_').astype('int')  # create new cols with a 1 in the max value col.
            self.synth_edges = (
                pd.concat([self.synth_edges, one_hot_encoded], axis=1)  # merge new cols with dataframe
                .drop(cs, axis=1)  # remove the old columns
                .rename(columns={"cat_"+c:c for c in cs})  # rename the new cols
            )
        
if __name__ == "__main__":
    path = "data/erdos/synth_graph/adjacency.parquet"
    synth_edges = pd.read_parquet(path)
    float32_cols = list(synth_edges.select_dtypes(include='float32'))
    synth_edges[float32_cols] = synth_edges[float32_cols].astype('float64')
    
    config_dict = yaml.safe_load('''
        bin_cols: ['edge_att0', 'edge_att1']
        cat_cols: [['edge_att2', 'edge_att3', 'edge_att4'], ['edge_att5', 'edge_att6']]
    ''')
    config_path = 'temp/'
    
    edge_cat_transformer = Edge_cat_transformer(synth_edges, config_dict, config_path)
    res = edge_cat_transformer.transform()
# %%
