from curses import use_default_colors
from distutils.fancy_getopt import fancy_getopt
from pathlib import Path
import numpy
import iris
from iris.experimental.ugrid import PARSE_UGRID_ON_LOAD
from mint import printLogMessages
import mint

from lfricabrac import ExtensiveField, FunctionSpace

DATA_DIR = Path(__file__).absolute().parent.parent / Path('data')


def test_simple():

    # single cell grid
    """
    3 <- 2
    V    ^
    0 -> 1
    """
    points = numpy.array([
        [0., 0., 0.],
        [1., 0., 0.],
        [1., 1., 0.],
        [0., 1., 0.]
    ], numpy.float64
    )
    edge2nodes = numpy.array([
        [0, 1],
        [1, 2],
        [3, 2],
        [0, 3]
    ], numpy.uint64)
    face2nodes = numpy.array([
        [0, 1, 2, 3]
    ], numpy.uint64)

    grid = mint.Grid()
    grid.loadFromUgrid2DData(points, face2nodes, edge2nodes)

    stream_function_values = numpy.array([0, -1., 1., 4.])

    edge_values_from_stream_function = numpy.empty(mint.NUM_EDGES_PER_QUAD, numpy.float64)
    for ie in range(mint.NUM_EDGES_PER_QUAD):
        i0 = ie
        i1 = (i0 + 1) % mint.NUM_EDGES_PER_QUAD

        # this sign takes into account the orientation of the edge
        # when computing the line integral
        sign = -2*(ie // 2) + 1

        # this sign takes care of the cross product in evaluating the flux
        sign2 = -2*((ie + 1) % 2) + 1

        edge_values_from_stream_function[ie] = sign2*sign*(stream_function_values[i1] - stream_function_values[i0])

    edge_values_exact = numpy.array([1., 2., 3., 4.])
    print(f'edge values from streamfunction: {edge_values_from_stream_function}')
    assert( (numpy.fabs(edge_values_exact - edge_values_from_stream_function) < 1.e-10).all() )

def test_cs():

    filename = DATA_DIR / 'cs8_wind.nc'

    a_radius = 6371.e3

    ef = ExtensiveField(filename, planet_radius=a_radius)
    ef.build(u_std_name="eastward_wind_at_cell_faces",
             v_std_name="northward_wind_at_cell_faces")

    edge_integrals = ef.compute_edge_integrals(func_space=FunctionSpace.W2H)
    with PARSE_UGRID_ON_LOAD.context():
        # defined on unique edge Ids
        edge_integrals_exact = ef.get_from_unique_edge_data(
            iris.load_cube(filename,
            "wind_integrated_at_cell_faces").data
        )
    

    num_cells = ef.get_num_faces()
    edge_integrals = edge_integrals.reshape((num_cells, mint.NUM_EDGES_PER_QUAD))
    edge_integrals_exact = edge_integrals_exact.reshape((num_cells, mint.NUM_EDGES_PER_QUAD))

    # for W2H staggering, we need to change the sign of the extensive field on
    # edges 0 and 2 due to the cross product
    # edge_integrals_exact[:, (0,2)] *= -1

    grid = ef.get_grid()
    verts = grid.getPoints()

    # compute the loop integrals
    loop_integrals = edge_integrals[:, 0] + \
                     edge_integrals[:, 1] - \
                     edge_integrals[:, 2] - \
                     edge_integrals[:, 3]

    
    loop_integrals_exact = edge_integrals_exact[:, 0] + \
                           edge_integrals_exact[:, 1] + \
                           edge_integrals_exact[:, 2] + \
                           edge_integrals_exact[:, 3]


    print(f'loop integrals exact min={loop_integrals_exact.min():.3e} max={loop_integrals_exact.max():.3e}')
    print(f'loop integrals min={loop_integrals.min():.3e} max={loop_integrals.max():.3e}')
    
    print(f'**** loop_integrals_exact = {loop_integrals_exact}')
    grid.attach('loop_integrals_exact', loop_integrals_exact, copy=False)
    grid.attach('loop_integrals', loop_integrals, copy=False)
    grid.dump('cs8.vtk')

    #  check the line integrals for a few edges
    u = ef.getU()
    v = ef.getV()

    deg2rad = numpy.pi/180.

    icell = 10
    for ie in range(mint.NUM_EDGES_PER_QUAD):
        edgeId, edgeSign = grid.getEdgeId(icell, ie)
        # get the start/end verts of this edge
        p0 = verts[icell, ie, :]
        p1 = verts[icell, (ie + 1) % mint.NUM_EDGES_PER_QUAD, :]
        # estimate the flux integral
        dLambda = (p1[0] - p0[0])*deg2rad
        dTheta = (p1[1] - p0[1])*deg2rad
        theta = 0.5*(p1[1] + p0[1])*deg2rad
        # this sign takes care of the orientation of the edge
        sign = 1
        if ie >= 2:
            sign = -1
        # this sign ensures that our fluxes are positive in the logical directions
        sign2 = 1
        if ie % 2 == 0:
            sign2 = -1
        flux = sign2*sign*a_radius*(dTheta*u[edgeId] - dLambda*v[edgeId]*numpy.cos(theta))
        print(f'cell {icell} edge {ie} u={u[edgeId]} v={v[edgeId]} p0={p0} p1={p1} edge int={edge_integrals[icell,ie]} exact={edge_integrals_exact[icell,ie]} flux={flux}')

    print(f'edge_integrals_exact min={edge_integrals_exact.min():.3e} max={edge_integrals_exact.max():.3e}')
    print(f'edge_integrals min={edge_integrals.min():.3e} max={edge_integrals.max():.3e}')
    mean_diff = numpy.fabs(edge_integrals - edge_integrals_exact).mean()
    mean_flux = numpy.fabs(edge_integrals).mean()
    printLogMessages()
    print(numpy.fabs(edge_integrals - edge_integrals_exact))
    mean_rel_diff = mean_diff/mean_flux
    print(f'mean rel diff = {mean_rel_diff}')
    assert(mean_rel_diff < 1.e-3)

if __name__ == '__main__':
    test_simple()
    test_cs()
