from ast import expr_context
from cmath import exp
from lfricabrac import Polyline
import numpy
from functools import reduce
from operator import __add__

def test_equator():
    xy = numpy.array([(0., 0.), (360., 0.)])
    earth_radius = 6371e3
    pl = Polyline(xy, planet_radius=earth_radius)
    length = reduce(__add__, pl.get_lengths())
    expected_length = 2*numpy.pi*earth_radius
    assert(abs(length - expected_length) < 1.e-10)
