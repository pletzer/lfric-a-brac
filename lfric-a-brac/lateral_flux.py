#
# Compute the lateral flux across an irregular boundary
#
import defopt
import mint
from pathlib import Path
import xarray
import numpy

INF = float('inf')
MESH_NAME = 'Mesh2d'
U_NAME = 'u_in_w2h'
V_NAME = 'v_in_w2h'
ELEV_NAME = 'half_levels'
A = 6.371e6 # Earth radius


class LateralFlux(object):

    def __init__(self, filename):
        self.filename = filename
        self.grid = mint.Grid()
        self.grid.loadFromUgrid2DFile(f'{filename}${MESH_NAME}')
        self.pli = mint.PolylineIntegral()
        self.pli.setGrid(self.grid)
        self.pli.buildLocator(numCellsPerBucket=128, periodX=360., enableFolding=False)
        self.numEdges = 0

    def compute_edge_integrals(self):

        nc = xarray.open_dataset(self.filename)

        # elevations
        elvs = nc[ELEV_NAME][:]

        # latitudes at the mid edge point
        edge_lat_name = nc[MESH_NAME].edge_coordinates.split()[1]
        edge_lats = nc[edge_lat_name][:]
        # in radians
        edge_lats *= numpy.pi/180.

        a_cos_lat = A * numpy.cos(edge_lats)

        efc = mint.ExtensiveFieldConverter()
        efc.setGrid(self.grid)

        self.dims = nc.variables[U_NAME].shape
        self.numEdges = self.dims[-1] # assume last dimension is number of edges
        self.edge_integrated = numpy.empty(self.dims, numpy.float64)

        self.mai = mint.MultiArrayIter(self.dims[:-1]) # assume last dimension is number of edges
        self.mai.begin()
        for i in range(self.mai.getNumIters()):
            inds = self.mai.getIndices()
            slce = list(inds) + [slice(0, self.numEdges)]
            # read a slice
            u = nc.variables[U_NAME][slce]
            v = nc.variables[V_NAME][slce]
            # convert from m/s to deg/s
            u *= (180./numpy.pi) / a_cos_lat
            v *= (180./numpy.pi) / A
            self.edge_integrated[slce] = efc.getFaceData(u, v, placement=mint.UNIQUE_EDGE_DATA)
            # uncomment if want to multiply by dz
            # self.edge_integrated[slce] *= elevs[:]
            # want to multiply by density?
            self.mai.next()


    def set_target_line(self, xy):
        xyz = numpy.array([(p[0], p[1], 0.) for p in xy])
        self.pli.computeWeights(xyz)

    def compute_fluxes(self):
        self.fluxes = numpy.empty(self.dims[:-1], numpy.float64)
        self.mai.begin()
        for i in range(self.mai.getNumIters()):
            inds = self.mai.getIndices()
            slce = list(inds) + [slice(0, self.numEdges)]
            self.fluxes[inds] = self.pli.getIntegral(self.edge_integrated[slce], placement=mint.UNIQUE_EDGE_DATA)

    def get_fluxes(self):
        return self.fluxes


############################################################################
def main(*, filename: Path='./lfric_diag.nc', target_line: str='[(-180., -85.), (180., 85.)]'):

    lf = LateralFlux(filename=filename)
    xy = eval(target_line)
    lf.set_target_line(xy)
    lf.compute_edge_integrals()
    lf.compute_fluxes()
    fluxes = lf.get_fluxes()
    print(f'integrated flux: {flux}')


if __name__ == '__main__':
    defopt.run(main)


