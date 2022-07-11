#
# Compute the lateral flux across an irregular boundary
#
import defopt
import mint
from pathlib import Path
import iris
from iris.experimental.ugrid import PARSE_UGRID_ON_LOAD
import numpy

INF = float('inf')
A = 6.371e6 # Earth radius


class LateralFlux(object):

    def __init__(self, filename):

        self.filename = filename
        self.grid = mint.Grid()
        self.pli = mint.PolylineIntegral()
        self.numEdges = 0
        self.numPoints = 0
        self.numFaces = 0
        self.u = None
        self.v = None

    def build(self):

        # get the horizontal velocity components
        with PARSE_UGRID_ON_LOAD.context():
            self.u = iris.load_cube(self.filename, "eastward_wind_at_cell_faces")
            assert(self.u.location == 'edge')
            self.v = iris.load_cube(self.filename, "northward_wind_at_cell_faces")
            assert(self.v.location == 'edge')

        # get the mesh points
        x = self.u.mesh.node_coords.node_x.points
        y = self.u.mesh.node_coords.node_y.points
        self.numPoints = x.shape[0]
        self.points = numpy.zeros((self.numPoints, 3), numpy.float64)
        self.points[:, 0] = x
        self.points[:, 1] = y

        # get the face-node connectivity
        # note: need to promote 64 bit integers
        self.face2node = numpy.array(self.u.mesh.face_node_connectivity.indices_by_location(),
                                     numpy.uint64)
        self.numFaces = self.face2node.shape[0]
        
        # get the edge-node connectivity
        self.edge2node = numpy.array(self.u.mesh.edge_node_connectivity.indices_by_location(),
                                    numpy.uint64)
        self.numEdges = self.edge2node.shape[0]

        # build the mesh
        self.grid.loadFromUgrid2DData(self.points, self.face2node, self.edge2node)
        self.pli.setGrid(self.grid)
        self.pli.buildLocator(numCellsPerBucket=128, periodX=360., enableFolding=False)


    def compute_edge_integrals(self):

        # get the mid edge locations
        edge_y = self.u.mesh.edge_coords.edge_y.points

        # in radians
        edge_y *= numpy.pi/180.
        a_cos_lat = A * numpy.cos(edge_y)

        efc = mint.ExtensiveFieldConverter()
        efc.setGrid(self.grid)

        self.dims = self.u.shape
        # assume last dimension is number of edges
        # the extensive field is always cell by cell
        dms = self.dims[:-1] + (self.numFaces*mint.NUM_EDGES_PER_QUAD,)
        self.edge_integrated = numpy.empty(dms, numpy.float64)

        self.mai = mint.MultiArrayIter(self.dims[:-1]) # assume last dimension is number of edges
        self.mai.begin()
        for i in range(self.mai.getNumIters()):

            inds = tuple(self.mai.getIndices())

            # assume last dimension is number of edges
            slab = inds + (slice(0, self.numEdges),)

            # read a slab of data
            u = self.u[slab]
            v = self.v[slab]

            # convert from m/s to deg/s
            u *= (180./numpy.pi) / a_cos_lat
            v *= (180./numpy.pi) / A

            # compute the edge integrated field
            extensive_data = efc.getFaceData(u.data, v.data, placement=mint.UNIQUE_EDGE_DATA)
            slab_cell_by_cell = inds + (slice(0, self.numFaces*mint.NUM_EDGES_PER_QUAD),)
            self.edge_integrated[slab_cell_by_cell] = extensive_data

            # uncomment if want to multiply by dz
            # self.edge_integrated[slce] *= elevs[:]
            # want to multiply by density?
            self.mai.next()


    def set_target_line(self, xy):
        xyz = numpy.array([(p[0], p[1], 0.) for p in xy])
        self.pli.computeWeights(xyz)


    def compute_fluxes(self):

        # assume last dimension is number of edges
        self.fluxes = numpy.empty(self.dims[:-1], numpy.float64)

        self.mai.begin()
        for i in range(self.mai.getNumIters()):
            inds = tuple(self.mai.getIndices())
            slab = inds + (slice(0, self.numFaces*mint.NUM_EDGES_PER_QUAD),)
            self.fluxes[inds] = self.pli.getIntegral(self.edge_integrated[slab],
                                placement=mint.CELL_BY_CELL_DATA)

    def get_fluxes(self):
        return self.fluxes


############################################################################
def main(*, filename: Path='./lfric_diag.nc', target_line: str='[(-180., -85.), (180., 85.)]'):

    lf = LateralFlux(filename=filename)
    lf.build()

    xy = eval(target_line)
    lf.set_target_line(xy)

    lf.compute_edge_integrals()

    lf.compute_fluxes()
    fluxes = lf.get_fluxes()
    print(f'integrated fluxes: {fluxes}')


if __name__ == '__main__':
    defopt.run(main)


