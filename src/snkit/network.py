"""Network representation and utilities
"""
import os
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import geopandas

import numpy as np
import pandas
import shapely.errors

try:
    import networkx as nx

    USE_NX = True
except ImportError:
    USE_NX = False

from geopandas import GeoDataFrame
from shapely.geometry import (
    Point,
    MultiPoint,
    LineString,
    GeometryCollection,
    shape,
    mapping,
)
from shapely.ops import split, linemerge, unary_union

from collections import Counter

# optional progress bars
if "SNKIT_PROGRESS" in os.environ and os.environ["SNKIT_PROGRESS"] in ("1", "TRUE"):
    try:
        from tqdm import tqdm
    except ImportError:
        from snkit.utils import tqdm_standin as tqdm
else:
    from snkit.utils import tqdm_standin as tqdm


class Network:
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
        """ """
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
            crs = {"init": "epsg:{}".format(epsg)}

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
            crs = {"init": "epsg:{}".format(epsg)}

        self.edges.to_crs(crs, inplace=True)
        self.nodes.to_crs(crs, inplace=True)


def add_ids(network, id_col="id", edge_prefix="edge", node_prefix="node"):
    """Add or replace an id column with ascending ids"""
    nodes = network.nodes.copy()
    if not nodes.empty:
        nodes = nodes.reset_index(drop=True)

    edges = network.edges.copy()
    if not edges.empty:
        edges = edges.reset_index(drop=True)

    nodes[id_col] = ["{}_{}".format(node_prefix, i) for i in range(len(nodes))]
    edges[id_col] = ["{}_{}".format(edge_prefix, i) for i in range(len(edges))]

    return Network(nodes=nodes, edges=edges)


def add_topology(network, id_col="id"):
    """Add or replace from_id, to_id to edges"""
    from_ids = []
    to_ids = []

    for edge in tqdm(
        network.edges.itertuples(), desc="topology", total=len(network.edges)
    ):
        start, end = line_endpoints(edge.geometry)

        start_node = nearest_node(start, network.nodes)
        from_ids.append(start_node[id_col])

        end_node = nearest_node(end, network.nodes)
        to_ids.append(end_node[id_col])

    edges = network.edges.copy()
    edges["from_id"] = from_ids
    edges["to_id"] = to_ids

    return Network(nodes=network.nodes, edges=edges)


def get_endpoints(network):
    """Get nodes for each edge endpoint"""
    endpoints = []
    for edge in tqdm(
        network.edges.itertuples(), desc="endpoints", total=len(network.edges)
    ):
        if edge.geometry is None:
            continue
        if edge.geometry.geometryType() == "MultiLineString":
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
    """Add nodes at line endpoints"""
    endpoints = get_endpoints(network)
    nodes = concat_dedup([network.nodes, endpoints])

    return Network(nodes=nodes, edges=network.edges)


def round_geometries(network, precision=3):
    """Round coordinates of all node points and vertices of edge linestrings to some precision"""

    def _set_precision(geom):
        return set_precision(geom, precision)

    network.nodes.geometry = network.nodes.geometry.apply(_set_precision)
    network.edges.geometry = network.edges.geometry.apply(_set_precision)
    return network


def split_multilinestrings(network):
    """
    Create multiple edges from any MultiLineString edge

    Ensures that edge geometries are all LineStrings, duplicates attributes
    over any created multi-edges.
    """

    edges = network.edges
    geom_col: str = geometry_column_name(edges)
    split_edges = edges.explode(column=geom_col, ignore_index=True)

    geo_types = set(split_edges.geom_type)
    if geo_types != {'LineString'}:
        raise ValueError(
            f"exploded edges are of type(s) {geo_types} but should only be LineString"
        )

    return Network(nodes=network.nodes, edges=split_edges)


def merge_multilinestring(geom):
    """Merge a MultiLineString to LineString"""
    try:
        if geom.geom_type == "MultiLineString":
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
    """Move nodes (within threshold) to edges"""

    def snap_node(geom):
        snap = nearest_point_on_edges(geom, network.edges)
        distance = snap.distance(geom)
        if threshold is not None and distance > threshold:
            snap = geom
        return snap

    geom_col = geometry_column_name(network.nodes)
    snapped_geoms = network.nodes[geom_col].apply(snap_node)
    nodes = GeoDataFrame(
        pandas.concat(
            [
                network.nodes.drop(geom_col, axis=1),
                GeoDataFrame(snapped_geoms, columns=[geom_col]),
            ],
            axis=1,
        ),
        crs=network.nodes.crs
    )

    return Network(nodes=nodes, edges=network.edges)


def split_edges_at_nodes(network, tolerance=1e-9):
    """Split network edges where they intersect node geometries"""
    split_edges = []
    for edge in tqdm(
        network.edges.itertuples(index=False), desc="split", total=len(network.edges)
    ):
        hits = nodes_intersecting(edge.geometry, network.nodes, tolerance)
        split_points = MultiPoint([hit.geometry for hit in hits.itertuples()])

        # potentially split to multiple edges
        edges = split_edge_at_points(edge, split_points, tolerance)
        split_edges.append(edges)

    # combine dfs
    edges = pandas.concat(split_edges, axis=0)
    edges = edges.reset_index().drop("index", axis=1)

    return Network(nodes=network.nodes, edges=edges)


def split_edges_at_intersections(network, tolerance=1e-9):
    """Split network edges where they intersect line geometries"""
    split_edges = []
    split_points = []
    for edge in tqdm(
        network.edges.itertuples(index=False), desc="split", total=len(network.edges)
    ):
        # note: the symmetry of intersection is not exploited here.
        # (If A intersects B, then B intersects A)
        # since edges are not modified within the loop, this has just
        # potential performance consequences.

        hits_points = edges_intersecting_points(edge.geometry, network.edges, tolerance)

        # store the split edges and intersection points
        split_points.extend(hits_points)
        hits_points = MultiPoint(hits_points)
        edges = split_edge_at_points(edge, hits_points, tolerance)
        split_edges.append(edges)

    # add the (potentially) split edges
    edges = pandas.concat(split_edges, axis=0)
    edges = edges.reset_index().drop("index", axis=1)

    # combine the original nodes with the new intersection nodes
    # dropping the duplicates.
    # note: there are at least duplicates from above since intersections
    # are checked twice
    # note: intersection nodes are appended, and if any duplicates, the
    # original counterparts are kept.
    nodes = GeoDataFrame(geometry=split_points)
    nodes = pandas.concat([network.nodes, nodes], axis=0).drop_duplicates()
    nodes = nodes.reset_index().drop("index", axis=1)

    return Network(nodes=nodes, edges=edges)


def link_nodes_to_edges_within(network, distance, condition=None, tolerance=1e-9):
    """Link nodes to all edges within some distance"""
    new_node_geoms = []
    new_edge_geoms = []
    for node in tqdm(
        network.nodes.itertuples(index=False), desc="link", total=len(network.nodes)
    ):
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
    unsplit = Network(nodes=all_nodes, edges=all_edges)
    return split_edges_at_nodes(unsplit, tolerance)


def link_nodes_to_nearest_edge(network, condition=None):
    """Link nodes to all edges within some distance"""
    new_node_geoms = []
    new_edge_geoms = []
    for node in tqdm(
        network.nodes.itertuples(index=False), desc="link", total=len(network.nodes)
    ):
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
    unsplit = Network(nodes=all_nodes, edges=all_edges)
    return split_edges_at_nodes(unsplit)


def merge_edges(network, id_col="id", by=None):
    """Merge edges that share a node with a connectivity degree of 2

    Parameters
    ----------
    network : snkit.network.Network
    id_col : string
    by : List[string], optional
      list of columns to use when merging an edge path - will not merge if
      edges have different values.
    """
    if "degree" not in network.nodes.columns:
        network.nodes["degree"] = network.nodes[id_col].apply(
            lambda x: node_connectivity_degree(x, network)
        )

    degree2 = list(network.nodes[id_col].loc[network.nodes.degree == 2])
    d2_set = set(degree2)
    edge_paths = []

    while d2_set:
        if len(d2_set) % 1000 == 0:
            print(len(d2_set))
        popped_node = d2_set.pop()
        node_path = set([popped_node])
        candidates = set([popped_node])
        while candidates:
            popped_cand = candidates.pop()
            matches = set(
                np.unique(
                    network.edges[["from_id", "to_id"]]
                    .loc[
                        (network.edges.from_id == popped_cand)
                        | (network.edges.to_id == popped_cand)
                    ]
                    .values
                )
            )
            matches.remove(popped_cand)
            matches = matches - node_path
            for match in matches:
                if match in degree2:
                    candidates.add(match)
                    node_path.add(match)
                    d2_set.remove(match)
                else:
                    node_path.add(match)
        if len(node_path) > 2:
            edge_paths.append(
                network.edges.loc[
                    (network.edges.from_id.isin(node_path))
                    & (network.edges.to_id.isin(node_path))
                ]
            )

    concat_edge_paths = []
    unique_edge_ids = set()
    new_node_ids = set(network.nodes[id_col]) - set(degree2)

    for edge_path in tqdm(edge_paths, desc="merge_edge_paths"):
        unique_edge_ids.update(list(edge_path[id_col]))
        edge_path = edge_path.dissolve(by=by)
        edge_path_dicts = []
        for edge in edge_path.itertuples(index=False):
            if edge.geometry.geom_type == "MultiLineString":
                edge_geom = linemerge(edge.geometry)
                if edge_geom.geom_type == "MultiLineString":
                    edge_geoms = list(edge_geom)
                else:
                    edge_geoms = [edge_geom]
            else:
                edge_geoms = [edge.geometry]
            for geom in edge_geoms:
                start, end = line_endpoints(geom)
                start = nearest_node(start, network.nodes)
                end = nearest_node(end, network.nodes)
                edge_path_dict = {
                    "from_id": start[id_col],
                    "to_id": end[id_col],
                    "geometry": geom,
                }
                for i, col in enumerate(edge_path.columns):
                    if col not in ("from_id", "to_id", "geometry"):
                        edge_path_dict[col] = edge[i]
                edge_path_dicts.append(edge_path_dict)

        concat_edge_paths.append(geopandas.GeoDataFrame(edge_path_dicts))
        new_node_ids.update(list(edge_path.from_id) + list(edge_path.to_id))

    edges_new = network.edges.copy()
    edges_new = edges_new.loc[~(edges_new.id.isin(list(unique_edge_ids)))]
    edges_new.geometry = edges_new.geometry.apply(merge_multilinestring)
    edges = pandas.concat(
        [edges_new, pandas.concat(concat_edge_paths).reset_index()], sort=False
    )

    nodes = network.nodes.set_index(id_col).loc[list(new_node_ids)].copy().reset_index()

    return Network(nodes=nodes, edges=edges)


def geometry_column_name(gdf):
    """Get geometry column name, fall back to 'geometry'"""
    try:
        geom_col = gdf.geometry.name
    except AttributeError:
        geom_col = "geometry"
    return geom_col


def matching_gdf_from_geoms(gdf, geoms):
    """Create a geometry-only GeoDataFrame with column name to match an existing GeoDataFrame"""
    geom_col = geometry_column_name(gdf)
    geom_arr = geoms_to_array(geoms)
    return GeoDataFrame(geom_arr, columns=[geom_col])


def geoms_to_array(geoms):
    geom_arr = np.empty(len(geoms), dtype="object")

    # Filter warnings until Shapely 2.0
    # see https://shapely.readthedocs.io/en/stable/migration.html
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=shapely.errors.ShapelyDeprecationWarning
        )
        geom_arr[:] = geoms

    return geom_arr


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
        network.edges[(network.edges.from_id == node) | (network.edges.to_id == node)]
    )


def drop_duplicate_geometries(gdf, keep="first"):
    """Drop duplicate geometries from a dataframe"""

    # convert to wkb so drop_duplicates will work
    # discussed in https://github.com/geopandas/geopandas/issues/521
    mask = gdf.geometry.apply(lambda geom: geom.wkb).drop_duplicates(keep=keep).index

    # use mask to drop from actual dataframe
    return gdf.loc[mask]


def nearest_point_on_edges(point, edges):
    """Find nearest point on edges to a point"""
    edge = nearest_edge(point, edges)
    snap = nearest_point_on_line(point, edge.geometry)
    return snap


def nearest_node(point, nodes):
    """Find nearest node to a point"""
    return nearest(point, nodes)


def nearest_edge(point, edges):
    """Find nearest edge to a point"""
    return nearest(point, edges)


def nearest(geom, gdf):
    """Find the element of a GeoDataFrame nearest a shapely geometry"""
    match_idx = gdf.sindex.nearest(geom, return_all=False)[1][0]
    return gdf.loc[match_idx]


def edges_within(point, edges, distance):
    """Find edges within a distance of point"""
    return d_within(point, edges, distance)


def edges_intersecting_points(line, edges, tolerance=1e-9):
    """Return intersection points of intersecting edges"""
    hits = edges_intersecting(line, edges, tolerance)

    hits_points = []
    for hit in hits.geometry:
        # first extract the actual intersections from the hits
        # for being new geometrical objects, they are not in the sindex
        intersection = line.intersection(hit)
        # if the line is not simple, there is a self-crossing point
        # (note that it will always interact with itself)
        # note that __eq__ is used on purpose instead of equals()
        # this is stricter: for geometries constructed in the same way
        # it makes sense since the sindex is used here
        if line == hit and not line.is_simple:
            # there is not built-in way to find self-crossing points
            # duplicated points after unary_union are the intersections
            intersection = unary_union(line)
            segments_coordinates = []
            for seg in intersection.geoms:
                segments_coordinates.extend(list(seg.coords))
            intersection = [
                Point(p) for p, c in Counter(segments_coordinates).items() if c > 1
            ]
            intersection = MultiPoint(intersection)

        # then extract the intersection points
        hits_points = intersection_endpoints(intersection, hits_points)

    return hits_points


def edges_intersecting(line, edges, tolerance=1e-9):
    """Find edges intersecting line"""
    return intersects(line, edges, tolerance)


def nodes_intersecting(line, nodes, tolerance=1e-9):
    """Find nodes intersecting line"""
    return intersects(line, nodes, tolerance)


def intersects(geom, gdf, tolerance=1e-9):
    """Find the subset of a GeoDataFrame intersecting with a shapely geometry"""
    return _intersects(geom, gdf, tolerance)


def d_within(geom, gdf, distance):
    """Find the subset of a GeoDataFrame within some distance of a shapely geometry"""
    return _intersects(geom, gdf, distance)


def _intersects(geom, gdf, tolerance=1e-9):
    if geom.is_empty:
        return geopandas.GeoDataFrame()
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
    """Return points at first and last vertex of a line"""
    try:
        coords = np.array(line.coords)
        start = Point(coords[0])
        end = Point(coords[-1])
    except NotImplementedError as e:
        print(line)
        raise e
    return start, end


def intersection_endpoints(geom, output=None):
    """Return the points from an intersection geometry

    It extracts the starting and ending points of intersection
    geometries recursively and appends them to `output`.
    This doesn't handle polygons or collections of polygons.
    """
    if output is None:
        output = []

    geom_type = geom.geom_type
    if geom.is_empty:
        pass
    elif geom_type == "Point":
        output.append(geom)
    elif geom_type == "LineString":
        start = Point(geom.coords[0])
        end = Point(geom.coords[-1])
        output.append(start)
        output.append(end)
    # recursively for collections of geometries
    # note that there is no shared inheritance relationship
    elif (
        geom_type == "MultiPoint"
        or geom_type == "MultiLineString"
        or geom_type == "GeometryCollection"
    ):
        for geom_ in geom.geoms:
            output = intersection_endpoints(geom_, output)

    return output


def split_edge_at_points(edge, points, tolerance=1e-9):
    """Split edge at point/multipoint"""
    try:
        segments = split_line(edge.geometry, points, tolerance)
    except ValueError:
        # if splitting fails, e.g. because points is empty GeometryCollection
        segments = [edge.geometry]
    edges = GeoDataFrame([edge] * len(segments))
    edges.geometry = segments
    return edges


def split_line(line, points, tolerance=1e-9):
    """Split line at point or multipoint, within some tolerance"""
    to_split = snap_line(line, points, tolerance)
    # when the splitter is a self-intersection point, shapely splits in
    # two parts only in a semi-arbitrary way, see the related question:
    # https://gis.stackexchange.com/questions/435879/python-shapely-split-a-complex-line-at-self-intersections?noredirect=1#comment711214_435879
    # checking only that the line is complex might not be enough
    # but the difference operation is useless in the worst case
    if not to_split.is_simple:
        to_split = to_split.difference(points)
    return list(split(to_split, points).geoms)


def snap_line(line, points, tolerance=1e-9):
    """Snap a line to points within tolerance, inserting vertices as necessary"""
    if points.geom_type == "Point":
        if points.distance(line) < tolerance:
            line = add_vertex(line, points)
    elif points.geom_type == "MultiPoint":
        points = [point for point in points.geoms if point.distance(line) < tolerance]
        for point in points:
            line = add_vertex(line, point)
    return line


def add_vertex(line, point):
    """Add a vertex to a line at a point"""
    v_idx = nearest_vertex_idx_on_line(point, line)
    point_coords = np.array(point.coords)[0]
    line_coords = np.array(line.coords)

    if (point_coords == line_coords[v_idx]).all():
        # nearest vertex could be identical to point, so return unchanged
        return line

    insert_before_idx = None
    if v_idx == 0:
        # nearest vertex could be start, so insert just after (or could extend)
        insert_before_idx = 1
    elif v_idx == len(line_coords) - 1:
        # nearest vertex could be end, so insert just before (or could extend)
        insert_before_idx = v_idx
    else:
        # otherwise insert in between vertices of nearest segment
        segment_before = LineString([line_coords[v_idx], line_coords[v_idx - 1]])
        segment_after = LineString([line_coords[v_idx], line_coords[v_idx + 1]])
        if point.distance(segment_before) < point.distance(segment_after):
            insert_before_idx = v_idx
        else:
            insert_before_idx = v_idx + 1
    # insert point coords before index, return new linestring
    new_coords = list(line_coords)
    new_coords.insert(insert_before_idx, point_coords)
    return LineString(new_coords)


def nearest_vertex_idx_on_line(point, line):
    """Return the index of nearest vertex to a point on a line"""
    # distance to all points is calculated here - and this is called once per splitting point
    # any way to avoid this m x n behaviour?
    # idea: put line vertices in an STRTree and query it repeatedly for nearest (with each
    # splitting point)
    line_coords = np.array(line.coords)
    nearest_idx, _ = min(
        [
            (idx, point.distance(Point(coords)))
            for idx, coords in enumerate(line_coords)
        ],
        key=lambda item: item[1],
    )
    return nearest_idx


def nearest_point_on_line(point, line):
    """Return the nearest point on a line"""
    return line.interpolate(line.project(point))


def set_precision(geom, precision):
    """Set geometry precision"""
    geom_mapping = mapping(geom)
    geom_mapping["coordinates"] = np.round(
        np.array(geom_mapping["coordinates"]), precision
    )
    return shape(geom_mapping)


def to_networkx(network, directed=False, weight_col=None):
    """Return a networkx graph"""
    if not USE_NX:
        raise ImportError("No module named networkx")
    else:
        # init graph
        if not directed:
            G = nx.Graph()
        else:
            G = nx.MultiDiGraph()
        # get nodes from network data
        G.add_nodes_from(network.nodes.id.to_list())

        # add nodal positions from geom
        for node_id, x, y in zip(
            network.nodes.id, network.nodes.geometry.x, network.nodes.geometry.y
        ):
            G.nodes[node_id]["pos"] = (x, y)

        # get edges from network data
        if weight_col is None:
            # default to geometry length
            edges_as_list = list(
                zip(
                    network.edges.from_id,
                    network.edges.to_id,
                    network.edges.geometry.length,
                )
            )
        else:
            edges_as_list = list(
                zip(
                    network.edges.from_id,
                    network.edges.to_id,
                    network.edges[weight_col],
                )
            )
        # add edges to graph
        G.add_weighted_edges_from(edges_as_list)
        return G


def get_connected_components(network):
    """Get connected components within network and id to each individual graph"""
    if not USE_NX:
        raise ImportError("No module named networkx")
    else:
        G = to_networkx(network)
        return sorted(nx.connected_components(G), key=len, reverse=True)


def add_component_ids(network, id_col="component_id"):
    """Add column of component IDs to network data"""
    # get connected components
    connected_parts = get_connected_components(network)
    # add unique id to each graph
    network.edges[id_col] = 0  # init id_col
    network.nodes[id_col] = 0  # init id_col
    for count, part in enumerate(connected_parts):
        # edges
        network.edges.loc[
            (network.edges.from_id.isin(list(part)))
            | (network.edges.to_id.isin(list(part))),
            id_col,
        ] = (
            count + 1
        )
        # nodes
        network.nodes.loc[(network.nodes.id.isin(list(part))), id_col] = count + 1
    # return
    return network
