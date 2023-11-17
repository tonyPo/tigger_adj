import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


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
