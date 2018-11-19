"""Test core objects/concepts
"""
from geopandas import GeoDataFrame
from pytest import fixture
from shapely.geometry import Point, LineString

import snkit


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
        x
        |
        x
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
      x |
        |
        | x
    """
    edge = LineString([(0, 0), (0, 2)])
    point_a = Point((0.5, 0))
    point_b = Point((-0.5, 2))
    edges = GeoDataFrame([{'geometry': edge}])
    nodes = GeoDataFrame([{'geometry': point_a}, {'geometry': point_b}])
    return snkit.Network(edges=edges, nodes=nodes)


def test_init():
    """Create an empty network
    """
    net = snkit.Network()
    assert len(net.nodes) == 0
    assert len(net.edges) == 0
