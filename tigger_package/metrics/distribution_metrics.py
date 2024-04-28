import math
import scipy 
import subprocess
import shlex
import os
import pandas as pd
import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    pass
from collections import defaultdict
from tigger_package.tools import plot_hist 

class NodeDistributionMetrics:
    def __init__(self, nodes, synth_nodes, gtrie_dir=None, temp_dir=None):
        self.nodes = nodes
        self.synth_nodes = synth_nodes
        self.gtrie_dir = gtrie_dir
        self.temp_dir = temp_dir
        
    def calculate_wasserstein_distance(self):
        cols = self.nodes.columns
        
        ws_dist = {}
        
        for col in cols:
            ws = scipy.stats.wasserstein_distance(
                self.nodes[col].values,
                self.synth_nodes[col].values
            )
            ws_dist[col] = ws
            
        ws_df = pd.DataFrame.from_dict(ws_dist, orient='index', columns=['value'] )
        ws_df['type'] = 'node_attributes'
        ws_df['metric'] = 'Wasserstein_distance'
        
        return ws_df
            
    def plot_hist(self):
        cols = self.nodes.columns
        
        cnt = len(cols)
        horz_plots = 4
        vert_plots = math.ceil(cnt/horz_plots)
        
        fig = plt.figure(figsize=(10,10))
        for i, col in enumerate(cols):
            ax = fig.add_subplot(vert_plots, horz_plots, i+1)
            ax.hist(self.nodes[col].values, bins=20, color='green', label='orig', alpha=0.5)
            if self.synth_nodes is not None:
                ax.hist(self.synth_nodes[col].values, bins=20, color='red', label='synth', alpha=0.5)
            if i==cnt-1:
                ax.legend(bbox_to_anchor=(1.3, 1.))
                
class EdgeDistributionMetrics:
    def __init__(self, edges, synth_edges, temp_dir='temp/', gtrie_dir='~/Downloads/gtrieScanner_src_01/'):
        self.edges = edges
        self.edges_degree = None
        self.synth_edges = synth_edges
        self.synth_edges_degree = None
        self.temp_dir = temp_dir
        self.gtrie_dir = gtrie_dir
        
        cols = list(self.edges.columns)
        cols.remove('end')
        cols.remove('start')
        self.cols = cols
        
        
    def calculate_wasserstein_distance(self):        
        ws_dist = {}
        
        for col in self.cols:
            ws = scipy.stats.wasserstein_distance(
                self.edges[col].values,
                self.synth_edges[col].values
            )
            ws_dist[col] = ws
            
        ws_df = pd.DataFrame.from_dict(ws_dist, orient='index', columns=['value'] )
        ws_df['type'] = 'edge_attributes'
        ws_df['metric'] = 'Wasserstein_distance'
        
        return ws_df
            
    def plot_hist(self):
        if self.synth_edges is None:
            plot_hist(self.edges.drop(['start', 'end'], axis=1), None)
        else:
            plot_hist(self.edges.drop(['start', 'end'], axis=1), self.synth_edges.drop(['src', 'dst'], axis=1))
                
    def get_degrees_dist(self):
        datasets = [('edges', {"src": 'start', 'dst': 'end'})]
        if self.synth_edges is not None:
            datasets.append(('synth_edges', {"src": 'src', 'dst': 'dst'}))

        if not self.edges_degree:
            for edge_name, name_dict in datasets:
                edges = getattr(self, edge_name)
                out_degree = edges[name_dict['src']].value_counts(sort=False).value_counts(sort=False)
                in_degree = edges[name_dict['dst']].value_counts(sort=False).value_counts(sort=False)
                setattr(self, edge_name+"_degree", {'out_degree': out_degree, 'in_degree': in_degree})
                
    def get_degree_wasserstein_distance(self):
        self.get_degrees_dist()
        ws_dist = {}
        for direction in ['in_degree', 'out_degree']:
            df = pd.concat([self.edges_degree[direction], self.synth_edges_degree[direction]], axis=1)
            df = df.fillna(0)
            ws = scipy.stats.wasserstein_distance(
                df.iloc[:,0],
                df.iloc[:,1]
            )
            ws_dist[direction] = ws
            
        ws_df = pd.DataFrame.from_dict(ws_dist, orient='index', columns=['value'] )
        ws_df['type'] = 'edge_degree'
        ws_df['metric'] = 'Wasserstein_distance'
        
        return ws_df
    
    def plot_degree_dst(self):
        self.get_degrees_dist()
        orig_in = self.edges_degree['in_degree']
        orig_out= self.edges_degree['out_degree']
        if self.synth_edges is not None:
            synth_in = self.synth_edges_degree['in_degree']
            synth_out = self.synth_edges_degree['out_degree']
        
        # calculate bins
        if self.synth_edges is not None:
            out_max = max(np.max(orig_out.keys()), np.max(synth_out.keys()))
            in_max =max(np.max(orig_in.keys()), np.max(synth_in.keys()))
        else:
            out_max = np.max(orig_out.keys())
            in_max = np.max(orig_in.keys())

        out_bins = [i* math.ceil(out_max/20) for i in range(20)]
        
        in_bins = [i* math.ceil(in_max/20) for i in range(20)]
        
        plt.style.use('seaborn')
        fig, (ax1, ax2) = plt.subplots(1, 2)
        
        ax1.hist(orig_in.index, weights=orig_in.values, bins=in_bins, alpha=0.5, label='orig', density=True, edgecolor='black')
        ax2.hist(orig_out.index, weights=orig_out.values, bins=out_bins, alpha=0.5, label='orig', density=True, edgecolor='black')
        if self.synth_edges is not None:
            ax1.hist(synth_in.index, weights=synth_in.values, bins=in_bins, alpha=0.5, label='synth', density=True, edgecolor='black')
            ax2.hist(synth_out.index, weights=synth_out.values, bins=out_bins, alpha=0.5, label='synth', density=True, edgecolor='black')
        
        ax1.legend()
        ax1.set_title("in degree dist")
        ax2.legend()
        ax2.set_title("out degree dist")
        fig.show()
        
    def widgets_distr(self, show_plot=True):
        assert self.temp_dir is not None, "temp dir is not set"
        assert self.gtrie_dir is not None, "gtrie dir is not set"
        dfs = {}
        if self.synth_edges is None:
            names = ['edges']
        else:
            names = ['edges', 'synth_edges']
        
        for name in names:
            input_file = self.adj_to_csv(name)
            
            gtrie_cmd = f"{self.gtrie_dir}gtrieScanner -s 3 -m gtrie {self.gtrie_dir}gtries/dir3.gt -d -t html "
            input = f"-g {input_file} "
            output = f"-o {self.temp_dir}dir3_{name}.html "
   
            with open(self.temp_dir+"console_output_"+name, 'w') as fp:
                print(gtrie_cmd + input + output)
                proc_res = subprocess.run(gtrie_cmd + input + output, shell=True, stdout=fp, stderr=fp)
                if proc_res.returncode < 0:
                    raise Exception("Terminal process did not exit succesfully")
            
            df = pd.read_html(f"{self.temp_dir}dir3_{name}.html")
            df =df[0][['Subgraph.1', 'Org. Frequency']]
            df[name+"_frac"] = df['Org. Frequency'] / df['Org. Frequency'].sum()
            df = df.rename({'Org. Frequency': name+"_freq"}, axis=1)
            dfs[name] = df

        if self.synth_edges is None:
            return dfs['edges']
            
        df = dfs['edges'].merge(dfs['synth_edges'], on='Subgraph.1', how='outer')
        
        #calculate % difference
        df['delta'] = np.absolute(df['edges_frac'] - df['synth_edges_frac'])
        
        #plot results
        if show_plot:
            plt.style.use('seaborn-v0_8')
            fig, ax = plt.subplots(1, 1)
            
            x_axis = np.arange(df.shape[0])
                            
            ax.bar(x_axis - 0.2, df.edges_frac, width=0.4, label='orig', )
            ax.bar(x_axis + 0.2, df.synth_edges_frac, width=0.4, label='synth')
            ax.legend()
            ax.set_xticks(np.arange(0, df.shape[0]))
            ax.set_xticklabels(df['Subgraph.1'], rotation=90)
            
            fig.show()

        return (df, df.delta.mean())
    
    def widgets_distr_table(self):
        df, _mean_delta = self.widgets_distr(show_plot=False)
        df_orig = df[['Subgraph.1', 'edges_frac']].rename(columns={'edges_frac': 'value'})
        df_orig.index = "ORIGINAL_" + df_orig['Subgraph.1']

        df_synth = df[['Subgraph.1', 'synth_edges_frac']].rename(columns={'synth_edges_frac': 'value'})
        df_synth.index = "SYNTH_" + df_synth['Subgraph.1']

        df_delta = df[['Subgraph.1', 'delta']].rename(columns={'delta': 'value'})
        df_delta.index = "DELTA_" + df_delta['Subgraph.1']


        df = pd.concat([df_orig, df_synth, df_delta], axis=0)
        df.drop("Subgraph.1", axis=1, inplace=True)
        df.loc["mean_delta_widget", "value"] = _mean_delta
        df['type'] = "widget_count"
        df['metric'] = "fraction"
        return df

        
    def clustering_coef_undirected(self):
        """calculates the global clustering coef"""
        assert self.temp_dir is not None, "temp dir is not set"
        assert self.gtrie_dir is not None, "gtrie dir is not set"
        cc = {}
        names = ['edges', 'synth_edges']
        
        for name in names:
            input_file = self.adj_to_csv(name)
   
            gtrie_cmd = f"{self.gtrie_dir}gtrieScanner -s 3 -m gtrie {self.gtrie_dir}gtries/undir3.gt -t html "
            input = f"-g {input_file} "
            output = f"-o {self.temp_dir}undir3_{name}.html "
   
            with open(self.temp_dir+"console_output_undir_"+name, 'w') as fp:
                proc_res = subprocess.run(gtrie_cmd + input + output, shell=True, stdout=fp, stderr=fp)
                if proc_res.returncode < 0:
                    raise Exception("Terminal process did not exit succesfully")
            
            df = pd.read_html(f"{self.temp_dir}undir3_{name}.html")
            df =df[0][['Subgraph.1', 'Org. Frequency']]
            # df[name+"_frac"] = df['Org. Frequency'] / df['Org. Frequency'].sum()
            # df = df.rename({'Org. Frequency': name+"_freq"}, axis=1)
            
            # calculate the cluster coef
            triangles =  df.loc[df['Subgraph.1']=='011 101 110','Org. Frequency']
            total = df['Org. Frequency'].sum() + 2 * triangles
            cluster_coef = triangles * 3 / total

            cc[name] = cluster_coef.to_numpy()[0]  
            
        # convert to datafraome
        cc['dif_cluster_coef'] = cc['edges'] - cc['synth_edges']
        cc_df = pd.DataFrame.from_dict(cc, orient='index', columns=['value']) 
        cc_df['type'] = 'cluster_coef'
        cc_df['metric'] = 'cluster_coefficient'
        
        return cc_df

    def get_undirected_degrees(self):
        degree_dict = {}
        for edge_name, name_dict in [('edges', {"src": 'start', 'dst': 'end'}), ('synth_edges', {"src": 'src', 'dst': 'dst'})]:
            edges = getattr(self, edge_name)
            neighbors_dict = defaultdict(set)
            
            df = edges.loc[:,[name_dict['src'],name_dict['dst']]]
            for id, row in df.iterrows():
                start_id = row[name_dict['src']]
                end_id = row[name_dict['dst']]
                neighbors_dict[start_id].add(end_id)
                neighbors_dict[end_id].add(start_id)
                            
            degrees = [len(v) for v in neighbors_dict.values()]
            degree_dict[edge_name] = degrees
        return degree_dict
            
                    
    def adj_to_csv(self, name):
        input_file = self.temp_dir + name + "_adj.csv"
        df = getattr(self, name)
        if name == 'edges':
            df = df.rename({'start': 'src', 'end': 'dst'}, axis=1)
           
        df = df[['src', 'dst']].copy()
        df['new_src'] = df['src'] + 1
        df['new_dst'] = df['dst'] + 1
        df['weight'] = 1
        
        # to_csv return a carriage return at the end which needs to be removed.
        adj_str = df[['new_src', 'new_dst', 'weight']].to_csv(header=False, index=False, sep=" ", compression=None)[:-1]
        
        with open(input_file, "w") as text_file:
            text_file.write(adj_str)
        return input_file
        
def compare_metrics(nodes, edges, synth_nodes, synth_edges, name, temp_dir='temp/', gtrie_dir='~/Downloads/gtrieScanner_src_01/'):
    ndm = NodeDistributionMetrics(nodes, synth_nodes)
    node_metric_df = ndm.calculate_wasserstein_distance()
    
    # edge atributes
    edm = EdgeDistributionMetrics(edges, synth_edges, temp_dir=temp_dir, gtrie_dir=gtrie_dir)
    edge_metrec_df = edm.calculate_wasserstein_distance()
    
    degree_metric_df = edm.get_degree_wasserstein_distance()
    widget_df = edm.widgets_distr_table()
    cc_df = edm.clustering_coef_undirected()
    
    results = pd.concat([node_metric_df,edge_metrec_df,degree_metric_df, cc_df, widget_df], axis=0)
    results.loc["edge_count",:] = {'value': synth_edges.shape[0], 'type': 'edge_cnt', 'metric': 'count'}
    results.loc["orig_edge_count",:] = {'value': edges.shape[0], 'type': 'edge_cnt', 'metric': 'count'}
    results.loc["node_count",:] = {'value': synth_nodes.shape[0], 'type': 'edge_cnt', 'metric': 'count'}
    results.loc["orig_node_count",:] = {'value': nodes.shape[0], 'type': 'edge_cnt', 'metric': 'count'}
    
    results = results.reset_index(names='name')
    results = results.rename(columns={"value": name})
    
    
    return results

