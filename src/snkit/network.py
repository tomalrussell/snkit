"""Network representation
"""
from geopandas import GeoDataFrame
import pandas


class Network():
    """A Network is composed of nodes (points in space) and edges (lines)

    Arguments
    ---------
    nodes : geopandas.geodataframe.GeoDataFrame, optional
    edges : geopandas.geodataframe.GeoDataFrame, optional

    Attributes
    ----------
    nodes : geopandas.geodataframe.GeoDataFrame
    edges : geopandas.geodataframe.GeoDataFrame

    """
    def __init__(self, nodes=None, edges=None):
        """
        """
        if nodes is None:
            nodes = GeoDataFrame()
        self.nodes = nodes

        if edges is None:
            edges = GeoDataFrame()
        self.edges = edges


def add_ids(network, id_col='id', edge_prefix='edge', node_prefix='node'):
    """Add an id column with ascending ids
    """
    node_ids = pandas.DataFrame(
        ['{}_{}'.format(node_prefix, i) for i in range(len(network.nodes))],
        columns=[id_col]
    )
    edge_ids = pandas.DataFrame(
        ['{}_{}'.format(edge_prefix, i) for i in range(len(network.edges))],
        columns=[id_col]
    )
    return Network(
        nodes=pandas.concat([network.nodes, node_ids], axis=1),
        edges=pandas.concat([network.edges, edge_ids], axis=1)
    )
