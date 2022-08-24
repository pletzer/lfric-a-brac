from pathlib import Path
import numpy
import iris
from iris.experimental.ugrid import PARSE_UGRID_ON_LOAD

from lfricabrac import ExtensiveField, FunctionSpace

DATA_DIR = Path(__file__).absolute().parent.parent / Path('data')

def test_cs2():
    filename = DATA_DIR / 'cs2_wind.nc'
    ef = ExtensiveField(filename, planet_radius=6271e3)
    ef.build(u_std_name="eastward_wind_at_cell_faces",
             v_std_name="northward_wind_at_cell_faces")
    edge_integrals = ef.compute_edge_integrals(func_space=FunctionSpace.W2H)
    with PARSE_UGRID_ON_LOAD.context():
        edge_integrals_exact = iris.load_cube(filename,
                               long_name="wind_integrated_at_cell_facespyets")
    diff = numpy.fabs(edge_integrals - edge_integrals_exact)
    assert(diff < 1.e-3)
