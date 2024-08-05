# graph_based_model.py
import networkx as nx
import pandas as pd


class GraphBasedModel:
    def __init__(self):
        self.graph = nx.Graph()

    def fit(self, edges_df):
        self.graph.add_edges_from(edges_df.values)

    def recommend(self, node, top_n=10):
        neighbors = list(self.graph.neighbors(node))
        return neighbors[:top_n]


def load_edges(file_path):
    return pd.read_csv(file_path)
