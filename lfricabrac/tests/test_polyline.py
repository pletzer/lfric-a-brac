from ast import expr_context
from cmath import exp
from functools import reduce
from operator import __add__
import numpy
from lfricabrac import Polyline


def test_equator():
    xy = numpy.array([(0., 0.), 
                      (90., 0.),
                      (180., 0.),
                      (270., 0.),
                      (360., 0.)])
    earth_radius = 6371e3
    # pl = Polyline(xy, planet_radius=earth_radius)
    # length = reduce(__add__, pl.get_lengths())
    # print(f'pl.get_lengths() = {pl.get_lengths()}')
    # print(f'length = {length}')
    # expected_length = 2*numpy.pi*earth_radius
    # assert(abs(length - expected_length) < 1.e-10)

