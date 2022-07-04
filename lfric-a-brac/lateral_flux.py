#
# Compute the lateral flux across an irregular boundary
#
import defopt
import mint
from pathlib import Path
import xarray

INF = float('inf')\
MESH_NAME = 'Mesh2d'
U_NAME = 'u_in_w2h'
V_NAME = 'v_in_w2h'
A = 6.371e6 # Earth radius


class LateralFlux(object):

    def __init__(self, filename):
        self.filename = filename
        self.grid = mint.Grid()
        self.grid.loadFromUgrid2DFile(f'{filename}${MESH_NAME}')
        self.pli = mint.PolylineIntegral()
        self.pli.setGrid(self.grid)
        self.pli.build_locator(numCellsPerBucket=128., periodX=360., enableFolding=False)
        self.numEdges = 0

    def compute_edge_integrals(self):
        nc = xarray.open_dataset(self.filename)
        # from m/s to deg/s
        edge_lat_name = nc[MESH_NAME].edge_coordinates.split()[1]
        edge_lats = nc[edge_lat_name][:]
        # in radians
        edge_lats *= numpy.pi/180.
        a_cos_lat = A * numpy.cos(edge_lats)

        efc = mint.ExtensiveFieldConverter()
        efc.setGrid(self.grid)

        dims = nc.variables[U_NAME].shape
        self.numEdges = dims[-1] # assume last dimension is number of edges
        self.edge_integrated = numpy.empty(dims, numpy.float64)
        mai = mint.MultiArrayIter(dims[:-1]) # assume last dimension is number of edges
        mai.begin()
        for i in range(mai.getNumIters()):
            inds = mai.getIndices()
            slce = list(inds) + [slice(0, numEdges)]
            # read a slice
            u = nc.variables[U_NAME][slce]
            v = nc.variables[V_NAME][slce]
            # convert from m/s to deg/s
            u *= (180/numpy.pi) / a_cos_lat
            v *= (180/numpy.pi) / A
            self.edge_integrated[slce] = efc.getFaceData(u, v, placement=mint.UNIQUE_EDGE_DATA)
            mai.next()


    def set_target_line(self, xy):
        xyz = numpy.array([(p[0], p[1], 0.) for p in xy])
        self.pli.computeWeights(xyz)

    def set_lo_level(self, z):
        # TO DO
        pass

    def set_hi_level(self, z):
        pass

    def compute_flux(self):
        pass


############################################################################
def main(*, filename: Path='./lfric_diag.nc', path: string='', lo_level: float=-INF, hi_level=INF):

    lf = LateralFlux(filename=filename)
    xy = eval(path)
    lf.set_path(xy)
    lf.set_lo_level(z=lo_level)
    lf.set_hi_level(z=hi_level)
    flux = lf.compute_flux()
    print(f'integrated flux: {flux}')


if __name__ == '__main__':
    defopt.run(main)


