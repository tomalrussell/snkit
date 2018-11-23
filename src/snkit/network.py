"""Network representation and utilities
"""
import os

import pandas
from geopandas import GeoDataFrame
from shapely.geometry import Point, MultiPoint, LineString
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


def add_topology(network, id_col='id'):
    """Add from_id, to_id to edges
    """
    from_ids = []
    to_ids = []
    for edge in tqdm(network.edges.itertuples(), desc="endpoints"):
        start, end = line_endpoints(edge.geometry)

        start_node = nearest_node(start, network.nodes)
        from_ids.append(start_node[id_col])

        end_node = nearest_node(end, network.nodes)
        to_ids.append(end_node[id_col])

    ids = pandas.DataFrame(data={
        'from_id': from_ids,
        'to_id': to_ids
    })

    return Network(
        nodes=network.nodes,
        edges=pandas.concat([network.edges, ids], axis=1)
    )



def get_endpoints(network, process=None):
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
    return matching_gdf_from_geoms(network.nodes, endpoints)


def add_endpoints(network):
    """Add nodes at line endpoints
    """
    endpoints = get_endpoints(network)
    nodes = concat_dedup([network.nodes, endpoints])

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
    for edge in tqdm(
            network.edges.itertuples(index=False), desc="split", max=len(network.edges)):
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


def link_nodes_to_edges_within(network, distance):
    """Link nodes to all edges within some distance
    """
    new_node_geoms = []
    new_edge_geoms = []
    for node in tqdm(
            network.nodes.itertuples(index=False), desc="link", max=len(network.nodes)):
        # for each node, find edges within
        edges = edges_within(node.geometry, network.edges, distance)
        for edge in edges.itertuples():
            # add nodes at points-nearest
            point = nearest_point_on_line(node.geometry, edge.geometry)
            if point != node.geometry:
                new_node_geoms.append(point)
                # add edges linking
                line = LineString([node.geometry, point])
                new_edge_geoms.append(line)

    new_nodes = matching_gdf_from_geoms(network.nodes, new_node_geoms)
    all_nodes = concat_dedup([network.nodes, new_nodes])

    new_edges = matching_gdf_from_geoms(network.edges, new_edge_geoms)
    all_edges = concat_dedup([network.edges, new_edges])

    # split edges as necessary after new node creation
    unsplit = Network(
        nodes=all_nodes,
        edges=all_edges
    )
    return split_edges_at_nodes(unsplit)


def geometry_column_name(gdf):
    """Get geometry column name, fall back to 'geometry'
    """
    try:
        geom_col = gdf.geometry.name
    except AttributeError:
        geom_col = 'geometry'
    return geom_col


def matching_gdf_from_geoms(gdf, geoms):
    """Create a geometry-only GeoDataFrame with column name to match an existing GeoDataFrame
    """
    geom_col = geometry_column_name(gdf)
    return GeoDataFrame(geoms, columns=[geom_col])


def concat_dedup(dfs):
    """Concatenate a list of GeoDataFrames, dropping duplicate geometries
    - note: repeatedly drops indexes for deduplication to work
    """
    cat = pandas.concat(dfs, axis=0)
    cat.reset_index(drop=True, inplace=True)
    cat_dedup = drop_duplicate_geometries(cat)
    cat_dedup.reset_index(drop=True, inplace=True)
    return cat_dedup


def drop_duplicate_geometries(gdf, keep='first'):
    """Drop duplicate geometries from a dataframe
    """
    # convert to wkb so drop_duplicates will work
    # discussed in https://github.com/geopandas/geopandas/issues/521
    mask = gdf.geometry.apply(lambda geom: geom.wkb)
    # use dropped duplicates index to drop from actual dataframe
    return gdf.iloc[mask.drop_duplicates(keep).index]


def nearest_point_on_edges(point, edges):
    """Find nearest point on edges to a point
    """
    edge = nearest_edge(point, edges)
    snap = nearest_point_on_line(point, edge.geometry)
    return snap


def nearest_node(point, nodes):
    """Find nearest node to a point
    """
    return nearest(point, nodes)


def nearest_edge(point, edges):
    """Find nearest edge to a point
    """
    return nearest(point, edges)


def nearest(geom, gdf):
    """Find the element of a GeoDataFrame nearest a shapely geometry
    """
    matches_idx = gdf.sindex.nearest(geom.bounds)
    nearest = min(
        [gdf.iloc[match_idx] for match_idx in matches_idx],
        key=lambda match: geom.distance(match.geometry)
    )
    return nearest


def edges_within(point, edges, distance):
    """Find edges within a distance of point
    """
    return d_within(point, edges, distance)


def nodes_intersecting(line, nodes):
    """Find nodes intersecting line
    """
    return intersects(line, nodes)


def d_within(geom, gdf, distance):
    """Find the subset of a GeoDataFrame within some distance of a shapely geometry
    """
    buf = geom.buffer(distance)
    return intersects(buf, gdf)


def intersects(geom, gdf):
    """Find the subset of a GeoDataFrame intersecting with a shapely geometry
    """
    candidate_idxs = list(gdf.sindex.intersection(geom.bounds))
    candidates = gdf.iloc[candidate_idxs]
    return candidates[candidates.intersects(geom)]


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
