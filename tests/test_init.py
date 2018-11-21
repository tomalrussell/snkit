"""Test core objects/concepts
"""
from geopandas import GeoDataFrame
from pandas import RangeIndex
from pandas.testing import assert_frame_equal
from pytest import fixture
from shapely.geometry import Point, LineString

import snkit
import snkit.network


@fixture
def edge_only():
    """Single edge:
        |
        |
        |
    """
    edge = LineString([(0, 0), (0, 2)])
    edges = GeoDataFrame([{'geometry': edge}])
    return snkit.Network(edges=edges)


@fixture
def nodes_only():
    """Two nodes:
        x

        x
    """
    point_a = Point((0, 0))
    point_b = Point((0, 2))
    nodes = GeoDataFrame([{'geometry': point_a}, {'geometry': point_b}])
    return snkit.Network(nodes=nodes)


@fixture
def connected():
    """Edge with nodes:
        b
        |
        a
    """
    edge = LineString([(0, 0), (0, 2)])
    point_a = Point((0, 0))
    point_b = Point((0, 2))
    edges = GeoDataFrame([{'geometry': edge}])
    nodes = GeoDataFrame([{'geometry': point_a}, {'geometry': point_b}])
    return snkit.Network(edges=edges, nodes=nodes)


@fixture
def misaligned():
    """Edge with nodes offset:
      b |
        |
        | a
    """
    edge = LineString([(0, 0), (0, 2)])
    point_a = Point((0.5, 0))
    point_b = Point((-0.5, 2))
    edges = GeoDataFrame([{'geometry': edge}])
    nodes = GeoDataFrame([{'geometry': point_a}, {'geometry': point_b}])
    return snkit.Network(edges=edges, nodes=nodes)


@fixture
def unsplit():
    """T-junction with nodes, long edge not split:
      b
      |
      |c--d
      |
      a
    """
    edge_ab = LineString([(0, 0), (0, 2)])
    edge_cd = LineString([(0, 1), (1, 1)])
    point_a = Point((0, 0))
    point_b = Point((0, 2))
    point_c = Point((0, 1))
    point_d = Point((1, 1))
    edges = GeoDataFrame([edge_ab, edge_cd], columns=['geometry'])
    nodes = GeoDataFrame([point_a, point_b, point_c, point_d], columns=['geometry'])
    return snkit.Network(edges=edges, nodes=nodes)


@fixture
def split():
    """T-junction with nodes, long edge split:
      b
      |
      c--d
      |
      a
    """
    edge_ac = LineString([(0, 0), (0, 1)])
    edge_cb = LineString([(0, 1), (0, 2)])
    edge_cd = LineString([(0, 1), (1, 1)])
    point_a = Point((0, 0))
    point_b = Point((0, 2))
    point_c = Point((0, 1))
    point_d = Point((1, 1))
    edges = GeoDataFrame([edge_ac, edge_cb, edge_cd], columns=['geometry'])
    nodes = GeoDataFrame([point_a, point_b, point_c, point_d], columns=['geometry'])
    return snkit.Network(edges=edges, nodes=nodes)


@fixture
def gap():
    """T-junction with nodes, edges not quite intersecting:
      b
      |
      | c--d
      |
      a
    """
    a = Point((0, 0))
    b = Point((0, 2))
    c = Point((0.1, 1))
    d = Point((1, 1))
    nodes = GeoDataFrame([a, b, c, d], columns=['geometry'])
    ab = LineString([a, b])
    cd = LineString([c, d])
    edges = GeoDataFrame([ab, cd], columns=['geometry'])
    return snkit.Network(edges=edges, nodes=nodes)


@fixture
def bridged():
    """T-junction with nodes, bridged by short edge:
      b
      |
      e-c--d
      |
      a
    """
    a = Point((0, 0))
    b = Point((0, 2))
    c = Point((0.1, 1))
    d = Point((1, 1))
    e = Point((0, 1))
    nodes = GeoDataFrame([a, b, c, d, e], columns=['geometry'])
    ae = LineString([a, e])
    eb = LineString([e, b])
    cd = LineString([c, d])
    ce = LineString([c, e])
    edges = GeoDataFrame([ae, eb, cd, ce], columns=['geometry'])
    return snkit.Network(edges=edges, nodes=nodes)


def test_init():
    """Should create an empty network
    """
    net = snkit.Network()
    assert len(net.nodes) == 0
    assert len(net.edges) == 0


def test_add_ids(edge_only, connected):
    """Should add ids to network nodes and columns
    """
    edge_with_ids = snkit.network.add_ids(edge_only)
    assert list(edge_with_ids.edges.id) == ['edge_0']
    assert list(edge_with_ids.nodes.id) == []

    net_with_ids = snkit.network.add_ids(connected)
    assert list(net_with_ids.edges.id) == ['edge_0']
    assert list(net_with_ids.nodes.id) == ['node_0', 'node_1']


def test_add_endpoints(edge_only, connected):
    """Should add nodes at edge endpoints
    """
    with_endpoints = snkit.network.add_endpoints(edge_only)
    assert_frame_equal(with_endpoints.nodes, connected.nodes)


def test_snap_nodes(misaligned, connected):
    """Should snap nodes to edges
    """
    snapped = snkit.network.snap_nodes(misaligned)
    assert_frame_equal(snapped.nodes, connected.nodes)

    # don't move if under threshold
    snapped = snkit.network.snap_nodes(misaligned, threshold=0.1)
    assert_frame_equal(snapped.nodes, misaligned.nodes)


def test_split_at_nodes(unsplit, split):
    """Should split edges at nodes, duplicating attributes if any
    """
    actual = snkit.network.split_edges_at_nodes(unsplit)
    assert_frame_equal(split.edges, actual.edges)

def test_link_nodes_to_edges_within(gap, bridged):
    actual = snkit.network.link_nodes_to_edges_within(gap, distance=0.1)
    assert_frame_equal(actual.nodes, bridged.nodes)
    assert_frame_equal(actual.edges, bridged.edges)
