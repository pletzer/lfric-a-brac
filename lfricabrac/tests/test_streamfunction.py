from pathlib import Path
import numpy
import iris
from iris.experimental.ugrid import PARSE_UGRID_ON_LOAD
from numpy import cos, sin # used for eval

DATA_DIR = Path(__file__).absolute().parent.parent / Path('data')

def test_cs():

    filename = DATA_DIR / 'cs8_wind.nc'

    with PARSE_UGRID_ON_LOAD.context():
        # defined on unique edge Ids
        edge_integrals = iris.load_cube(filename,
            "edge_integrals")

    assert(edge_integrals.location == 'edge')

    # get the coordinates
    deg2rad = numpy.pi/180.
    xnode = edge_integrals.mesh.node_coords.node_x.points * deg2rad
    ynode = edge_integrals.mesh.node_coords.node_y.points * deg2rad

    # get the edge to node connectivity
    edge2node = numpy.array(edge_integrals.mesh.edge_node_connectivity.indices_by_location(),
                numpy.uint64)
    edge2node -= edge_integrals.mesh.edge_node_connectivity.start_index

    # get the start/end node indices for each edge
    ibeg, iend = edge2node[:, 0], edge2node[:, 1]

    # get the start/end node coordinates for each edge
    xbeg, ybeg = xnode[ibeg], ynode[ibeg]
    xend, yend = xnode[iend], ynode[iend]

    # evaluate the stream function difference on each edge
    stream_function_expr = edge_integrals.attributes['stream_function']

    x, y = xbeg, ybeg
    sbeg = eval(stream_function_expr)

    x, y = xend, yend
    send = eval(stream_function_expr)

    edge_integrals_exact = send - sbeg

    # check
    assert(numpy.fabs(edge_integrals_exact - edge_integrals.data).mean() < 1.e-10)


if __name__ == '__main__':
    test_cs()
