import math
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def nx_to_df(graph):
    edges = nx.to_pandas_edgelist(graph)
    nodes = pd.DataFrame.from_dict(dict(graph.nodes(data=True)), orient='index')
    nodes["id"] = nodes.index
    return (nodes, edges)

def plot_adj_matrix(adj_df):
    
    attr = list(adj_df.columns)
    attr.remove('dst')
    attr.remove('src')
    G = nx.from_pandas_edgelist(adj_df, source='src', target='dst', create_using=nx.DiGraph)
    
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G)
    nx.draw_networkx_edges(G, pos, width=0.1, arrows=False, alpha=0.4)
    nx.draw_networkx_nodes(
        G, pos,
        node_size=10,
        node_color='black',
    )
    plt.axis("off")
    plt.show()

def plot_hist(x_data, embed, start=0, end=-1, min_x=None, max_x=None):
    if embed is None:
        show_embed = False
        embed = x_data
    else:
        show_embed = True

    fig = plt.figure(figsize=(20,20))
    dim = x_data.shape[1]
    rows = math.ceil(len(range(dim)[start:end])/2)
    
    #set labels
    if hasattr(x_data, 'name'):
        x_label = x_data.name
    else:
        x_label = 'orig'

  
    if hasattr(embed, 'name'):
        embed_label = embed.name
    else:
        embed_label = 'synth'
    
    for i in range(dim)[start:end]:
        if min_x is None:
            min_val = min(embed.iloc[:, i].min(),x_data.iloc[:, i].min())
        else:
            min_val= min_x
            
        if max_x is None:
            max_val = max(embed.iloc[:, i].max(),x_data.iloc[:, i].max())
        else:
            max_val = max_x
        bins = np.linspace(min_val, max_val, 20)
        ax = fig.add_subplot(rows,2, i+1-start)
        if show_embed:
            ax.hist(embed.iloc[:, i], bins=bins, alpha=0.5, color='g', label=embed_label)
        ax.hist(x_data.iloc[:, i], bins=bins, alpha=0.5, color='r', label=x_label)

    plt.legend()
    plt.show()
    