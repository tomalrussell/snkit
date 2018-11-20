"""Network representation and utilities
"""
from geopandas import GeoDataFrame
import pandas
from shapely.geometry import Point


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


def get_endpoints(network):
    """Get nodes for each edge endpoint
    """
    endpoints = []
    for edge in network.edges.itertuples():
        start = Point(edge.geometry.coords[0])
        end = Point(edge.geometry.coords[-1])
        endpoints.append(start)
        endpoints.append(end)

    # create dataframe to match the nodes geometry column name
    try:
        geom_col = network.nodes.geometry.name
    except AttributeError:
        geom_col = 'geometry'
    return drop_duplicate_geometries(GeoDataFrame(endpoints, columns=[geom_col]))


def add_endpoints(network):
    """Add nodes at line endpoints
    """
    endpoints = get_endpoints(network)
    nodes = drop_duplicate_geometries(pandas.concat([network.nodes, endpoints], axis=0))

    return Network(
        nodes=nodes,
        edges=network.edges
    )


def drop_duplicate_geometries(gdf, keep='first'):
    """Drop duplicate geometries from a dataframe
    """
    # convert to wkb so drop_duplicates will work
    # discussed in https://github.com/geopandas/geopandas/issues/521
    mask = gdf.geometry.apply(lambda geom: geom.wkb)
    # use dropped duplicates index to drop from actual dataframe
    return gdf.loc[mask.drop_duplicates(keep).index]
