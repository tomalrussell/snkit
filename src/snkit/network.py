"""Network representation and utilities
"""
import os

import numpy as np
import pandas
import shapely.errors

from geopandas import GeoDataFrame
from shapely.geometry import Point, MultiPoint, LineString, GeometryCollection, shape, mapping
from shapely.ops import split, linemerge

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

    Parameters
    ----------
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

    def set_crs(self, crs=None, epsg=None):
        """Set network (node and edge) crs

        Parameters
        ----------
        crs : dict or str
            Projection parameters as PROJ4 string or in dictionary form.
        epsg : int
            EPSG code specifying output projection

        """
        if crs is None and epsg is None:
            raise ValueError("Either crs or epsg must be provided to Network.set_crs")

        if epsg is not None:
            crs = {'init': 'epsg:{}'.format(epsg)}

        self.edges.crs = crs
        self.nodes.crs = crs

    def to_crs(self, crs=None, epsg=None):
        """Set network (node and edge) crs

        Parameters
        ----------
        crs : dict or str
            Projection parameters as PROJ4 string or in dictionary form.
        epsg : int
            EPSG code specifying output projection

        """
        if crs is None and epsg is None:
            raise ValueError("Either crs or epsg must be provided to Network.set_crs")

        if epsg is not None:
            crs = {'init': 'epsg:{}'.format(epsg)}

        self.edges.to_crs(crs, inplace=True)
        self.nodes.to_crs(crs, inplace=True)


def add_ids(network, id_col='id', edge_prefix='edge', node_prefix='node'):
    """Add or replace an id column with ascending ids
    """
    nodes = network.nodes.copy()
    if not nodes.empty:
        nodes = nodes.reset_index(drop=True)

    edges = network.edges.copy()
    if not edges.empty:
        edges = edges.reset_index(drop=True)

    nodes[id_col] = ['{}_{}'.format(node_prefix, i) for i in range(len(nodes))]
    edges[id_col] = ['{}_{}'.format(edge_prefix, i) for i in range(len(edges))]

    return Network(
        nodes=nodes,
        edges=edges
    )


def add_topology(network, id_col='id'):
    """Add or replace from_id, to_id to edges
    """
    from_ids = []
    to_ids = []

    for edge in tqdm(network.edges.itertuples(), desc="topology", total=len(network.edges)):
        start, end = line_endpoints(edge.geometry)

        start_node = nearest_node(start, network.nodes)
        from_ids.append(start_node[id_col])

        end_node = nearest_node(end, network.nodes)
        to_ids.append(end_node[id_col])

    edges = network.edges.copy()
    edges['from_id'] = from_ids
    edges['to_id'] = to_ids

    return Network(
        nodes=network.nodes,
        edges=edges
    )


def get_endpoints(network):
    """Get nodes for each edge endpoint
    """
    endpoints = []
    for edge in tqdm(network.edges.itertuples(), desc="endpoints", total=len(network.edges)):
        if edge.geometry is None:
            continue
        if edge.geometry.geometryType() == 'MultiLineString':
            for line in edge.geometry.geoms:
                start, end = line_endpoints(line)
                endpoints.append(start)
                endpoints.append(end)
        else:
            start, end = line_endpoints(edge.geometry)
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


def round_geometries(network, precision=3):
    """Round coordinates of all node points and vertices of edge linestrings to some precision
    """
    def _set_precision(geom):
        return set_precision(geom, precision)
    network.nodes.geometry = network.nodes.geometry.apply(_set_precision)
    network.edges.geometry = network.edges.geometry.apply(_set_precision)
    return network


def split_multilinestrings(network):
    """Create multiple edges from any MultiLineString edge

    Ensures that edge geometries are all LineStrings, duplicates attributes over any
    created multi-edges.
    """
    simple_edge_attrs = []
    simple_edge_geoms = []
    edges = network.edges
    for edge in tqdm(edges.itertuples(index=False), desc="split_multi", total=len(edges)):
        if edge.geometry.geom_type == 'MultiLineString':
            edge_parts = list(edge.geometry)
        else:
            edge_parts = [edge.geometry]

        for part in edge_parts:
            simple_edge_geoms.append(part)

        attrs = GeoDataFrame([edge] * len(edge_parts))
        simple_edge_attrs.append(attrs)

    simple_edge_geoms = GeoDataFrame(simple_edge_geoms, columns=['geometry'])
    edges = pandas.concat(simple_edge_attrs, axis=0).reset_index(drop=True).drop('geometry', axis=1)
    edges = pandas.concat([edges, simple_edge_geoms], axis=1)

    return Network(
        nodes=network.nodes,
        edges=edges
    )

def merge_multilinestring(geom):
    """ Merge a MultiLineString to LineString
    """
    try:
        if geom.geom_type == 'MultiLineString':
            geom_inb = linemerge(geom)
            if geom_inb.is_ring:
                return geom
# In case of linestring merge issues, we could add this to the script again
#            from centerline.main import Centerline
#            if geom_inb.geom_type == 'MultiLineString':
#                return linemerge(Centerline(geom.buffer(0.5)))
            else:
                return geom_inb
        else:
            return geom
    except:
        return GeometryCollection()

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


def split_edges_at_nodes(network, tolerance=1e-9):
    """Split network edges where they intersect node geometries
    """
    split_edges = []
    for edge in tqdm(network.edges.itertuples(index=False), desc="split", total=len(network.edges)):
        hits = nodes_intersecting(edge.geometry, network.nodes, tolerance)
        split_points = MultiPoint([hit.geometry for hit in hits.itertuples()])

        # potentially split to multiple edges
        edges = split_edge_at_points(edge, split_points, tolerance)
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


def link_nodes_to_edges_within(network, distance, condition=None, tolerance=1e-9):
    """Link nodes to all edges within some distance
    """
    new_node_geoms = []
    new_edge_geoms = []
    for node in tqdm(network.nodes.itertuples(index=False), desc="link", total=len(network.nodes)):
        # for each node, find edges within
        edges = edges_within(node.geometry, network.edges, distance)
        for edge in edges.itertuples():
            if condition is not None and not condition(node, edge):
                continue
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
    return split_edges_at_nodes(unsplit, tolerance)


def link_nodes_to_nearest_edge(network, condition=None):
    """Link nodes to all edges within some distance
    """
    new_node_geoms = []
    new_edge_geoms = []
    for node in tqdm(network.nodes.itertuples(index=False), desc="link", total=len(network.nodes)):
        # for each node, find edges within
        edge = nearest_edge(node.geometry, network.edges)
        if condition is not None and not condition(node, edge):
            continue
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

def merge_edges(network):
    """ Merge edges that share a node with a connectivity degree of 2
    """
    if 'degree' not in network.nodes.columns:
        network.nodes['degree'] = network.nodes.id.apply(lambda x:
                                                 node_connectivity_degree(x,network))

    degree2 = list(network.nodes.id.loc[network.nodes.degree == 2])
    d2_set = set(degree2)
    node_paths = []
    edge_paths = []

    while d2_set:
        popped_node = d2_set.pop()
        node_path = [popped_node]
        candidates = set([popped_node])
        while candidates:
            popped_cand = candidates.pop()
            matches = list(np.unique(network.edges[['from_id','to_id']].loc[(
                    (network.edges.from_id.isin([popped_cand])) |
                    (network.edges.to_id.isin([popped_cand])))].values))
            matches.remove(popped_cand)
            for match in matches:
                if match in node_path:
                    continue

                if match in degree2:
                    candidates.add(match)
                    node_path.append(match)
                    d2_set.remove(match)
                else:
                    node_path.append(match)
        if len(node_path) > 2:
            node_paths.append(node_path)
            edge_paths.append(network.edges.loc[(
                    (network.edges.from_id.isin(node_path)) &
                    (network.edges.to_id.isin(node_path)))])

    concat_edge_paths = []
    unique_edge_ids = set()
    for edge_path in edge_paths:
        unique_edge_ids.update(list(edge_path.id))
        if edge_path.bridge.isnull().any():
            edge_path = edge_path.copy()
            edge_path['bridge'] = 'yes'
        concat_edge_paths.append(edge_path.dissolve(by=['infra_type'], aggfunc='first'))

    edges_new = network.edges.copy()
    edges_new = edges_new.loc[~(edges_new.id.isin(list(unique_edge_ids)))]
    edges_new.geometry = edges_new.geometry.apply(merge_multilinestring)
    network.edges = pandas.concat([edges_new,pandas.concat(concat_edge_paths).reset_index()],sort=False)

    nodes_new = network.nodes.copy()
    network.nodes = nodes_new.loc[~(nodes_new.id.isin(list(degree2)))]

    return Network(
        nodes=network.nodes,
        edges=network.edges
    )

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
    cat = pandas.concat(dfs, axis=0, sort=False)
    cat.reset_index(drop=True, inplace=True)
    cat_dedup = drop_duplicate_geometries(cat)
    cat_dedup.reset_index(drop=True, inplace=True)
    return cat_dedup

def node_connectivity_degree(node, network):
    return len(
            network.edges[
                (network.edges.from_id == node) | (network.edges.to_id == node)
            ]
    )

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
    nearest_geom = min(
        [gdf.iloc[match_idx] for match_idx in matches_idx],
        key=lambda match: geom.distance(match.geometry)
    )
    return nearest_geom


def edges_within(point, edges, distance):
    """Find edges within a distance of point
    """
    return d_within(point, edges, distance)


def nodes_intersecting(line, nodes, tolerance=1e-9):
    """Find nodes intersecting line
    """
    return intersects(line, nodes, tolerance)


def intersects(geom, gdf, tolerance=1e-9):
    """Find the subset of a GeoDataFrame intersecting with a shapely geometry
    """
    return _intersects(geom, gdf, tolerance)


def d_within(geom, gdf, distance):
    """Find the subset of a GeoDataFrame within some distance of a shapely geometry
    """
    return _intersects(geom, gdf, distance)


def _intersects(geom, gdf, tolerance=1e-9):
    buf = geom.buffer(tolerance)
    if buf.is_empty:
        # can have an empty buffer with too small a tolerance, fallback to original geom
        buf = geom
    try:
        return _intersects_gdf(buf, gdf)
    except shapely.errors.TopologicalError:
        # can exceptionally buffer to an invalid geometry, so try re-buffering
        buf = buf.buffer(0)
        return _intersects_gdf(buf, gdf)

def _intersects_gdf(geom, gdf):
    candidate_idxs = list(gdf.sindex.intersection(geom.bounds))
    candidates = gdf.iloc[candidate_idxs]
    return candidates[candidates.intersects(geom)]


def line_endpoints(line):
    """Return points at first and last vertex of a line
    """
    start = Point(line.coords[0])
    end = Point(line.coords[-1])
    return start, end


def split_edge_at_points(edge, points, tolerance=1e-9):
    """Split edge at point/multipoint
    """
    try:
        segments = split_line(edge.geometry, points, tolerance)
    except ValueError:
        # if splitting fails, e.g. becuase points is empty GeometryCollection
        segments = [edge.geometry]
    edges = GeoDataFrame([edge] * len(segments))
    edges.geometry = segments
    return edges


def split_line(line, points, tolerance=1e-9):
    """Split line at point or multipoint, within some tolerance
    """
    to_split = snap_line(line, points, tolerance)
    return list(split(to_split, points))


def snap_line(line, points, tolerance=1e-9):
    """Snap a line to points within tolerance, inserting vertices as necessary
    """
    if points.geom_type == 'Point':
        if points.distance(line) < tolerance:
            line = add_vertex(line, points)
    elif points.geom_type == 'MultiPoint':
        points = [point for point in points if point.distance(line) < tolerance]
        for point in points:
            line = add_vertex(line, point)
    return line


def add_vertex(line, point):
    """Add a vertex to a line at a point
    """
    v_idx = nearest_vertex_idx_on_line(point, line)
    point_coords = tuple(point.coords[0])

    if point_coords == line.coords[v_idx]:
        # nearest vertex could be identical to point, so return unchanged
        return line

    insert_before_idx = None
    if v_idx == 0:
        # nearest vertex could be start, so insert just after (or could extend)
        insert_before_idx = 1
    elif v_idx == len(line.coords) - 1:
        # nearest vertex could be end, so insert just before (or could extend)
        insert_before_idx = v_idx
    else:
        # otherwise insert in between vertices of nearest segment
        segment_before = LineString([line.coords[v_idx], line.coords[v_idx - 1]])
        segment_after = LineString([line.coords[v_idx], line.coords[v_idx + 1]])
        if point.distance(segment_before) < point.distance(segment_after):
            insert_before_idx = v_idx
        else:
            insert_before_idx = v_idx + 1
    # insert point coords before index, return new linestring
    new_coords = list(line.coords)
    new_coords.insert(insert_before_idx, point_coords)
    return LineString(new_coords)


def nearest_vertex_idx_on_line(point, line):
    """Return the index of nearest vertex to a point on a line
    """
    # distance to all points is calculated here - and this is called once per splitting point
    # any way to avoid this m x n behaviour?
    nearest_idx, _ = min(
        [(idx, point.distance(Point(coords))) for idx, coords in enumerate(line.coords)],
        key=lambda item: item[1]
    )
    return nearest_idx


def nearest_point_on_line(point, line):
    """Return the nearest point on a line
    """
    return line.interpolate(line.project(point))


def set_precision(geom, precision):
    """Set geometry precision
    """
    geom_mapping = mapping(geom)
    geom_mapping['coordinates'] = np.round(np.array(geom_mapping['coordinates']), precision)
    return shape(geom_mapping)
