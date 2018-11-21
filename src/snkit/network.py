"""Network representation and utilities
"""
import os

import pandas
from geopandas import GeoDataFrame
from shapely.geometry import Point, MultiPoint
from shapely.ops import split

# optional progress bars
if 'SNKIT_PROGRESS' in os.environ and os.environ['SNKIT_PROGRESS'] in ('1', 'TRUE'):
    try:
        from tqdm import tqdm
    except ImportError:
        from snkit.utils import tqdm_standin as tqdm
else:
    from snkit.utils import tqdm_standin as tqdm


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
    for edge in tqdm(network.edges.itertuples(), desc="endpoints", max=len(network.edges)):
        if edge.geometry is None:
            continue
        if edge.geometry.geometryType() == 'MultiLineString':
            for line in edge.geometry.geoms:
                start = Point(line.coords[0])
                end = Point(line.coords[-1])
                endpoints.append(start)
                endpoints.append(end)
        else:
            start = Point(edge.geometry.coords[0])
            end = Point(edge.geometry.coords[-1])
            endpoints.append(start)
            endpoints.append(end)

    # create dataframe to match the nodes geometry column name
    geom_col = geometry_column_name(network.nodes)
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


def snap_nodes(network, threshold=None):
    """Move nodes (within threshold) to edges
    """
    def snap_node(node):
        snap = nearest_point_on_edges(node.geometry, network.edges)
        distance = snap.distance(node.geometry)
        if threshold is not None and distance > threshold:
            snap = node.geometry
        return snap

    snapped_geoms = network.nodes.apply(snap_node, axis=1)
    geom_col = geometry_column_name(network.nodes)
    nodes = pandas.concat([
        network.nodes.drop(geom_col, axis=1),
        GeoDataFrame(snapped_geoms, columns=[geom_col])
    ], axis=1)

    return Network(
        nodes=nodes,
        edges=network.edges
    )


def split_edges_at_nodes(network):
    """Split network edges where they intersect node geometries
    """
    split_edges = []
    for i, edge in tqdm(
            enumerate(network.edges.itertuples(index=False)), desc="split", max=len(network.edges)):
        hits = nodes_intersecting(edge.geometry, network.nodes)
        split_points = MultiPoint([hit.geometry for hit in hits.itertuples()])

        # potentially split to multiple edges
        edges = split_edge_at_points(edge, split_points)
        split_edges.append(edges)

    # combine dfs
    edges = pandas.concat(split_edges, axis=0)
    # reset index and drop
    edges = edges.reset_index().drop('index', axis=1)
    # return new network with split edges
    return Network(
        nodes=network.nodes,
        edges=edges
    )


def geometry_column_name(gdf):
    """Get geometry column name, fall back to 'geometry'
    """
    try:
        geom_col = gdf.geometry.name
    except AttributeError:
        geom_col = 'geometry'
    return geom_col


def drop_duplicate_geometries(gdf, keep='first'):
    """Drop duplicate geometries from a dataframe
    """
    # convert to wkb so drop_duplicates will work
    # discussed in https://github.com/geopandas/geopandas/issues/521
    mask = gdf.geometry.apply(lambda geom: geom.wkb)
    # use dropped duplicates index to drop from actual dataframe
    return gdf.loc[mask.drop_duplicates(keep).index]


def nearest_point_on_edges(point, edges):
    """Find nearest point on edges to a point
    """
    edge = nearest_edge(point, edges)
    snap = nearest_point_on_line(point, edge.geometry)
    return snap


def nearest_edge(point, edges):
    """Find nearest edge to a point
    """
    query = (point.x, point.y, point.x, point.y)
    matches_idx = list(edges.sindex.nearest(query))
    assert len(matches_idx) == 1
    for m in matches_idx:
        match = edges.iloc[m]
    return match


def edges_within(point, edges, distance):
    """Find edges within a distance of point
    """
    pass

def nodes_intersecting(line, nodes):
    """Find nodes intersecting line
    """
    bounds = line.bounds
    candidate_idxs = list(nodes.sindex.intersection(bounds))
    candidates = nodes.iloc[candidate_idxs]
    return candidates[candidates.intersects(line)]


def split_edge_at_points(edge, points):
    """Split edge at point/multipoint
    """
    segments = list(split(edge.geometry, points))
    edges = GeoDataFrame([edge] * len(segments))
    edges.geometry = segments
    return edges


def nearest_point_on_line(point, line):
    """Return the nearest point on a line
    """
    return line.interpolate(line.project(point))
