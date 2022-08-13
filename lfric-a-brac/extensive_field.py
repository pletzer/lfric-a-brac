#
# Compute the lateral flux across an irregular boundary
#
import defopt
import mint
from pathlib import Path
import iris
from iris.experimental.ugrid import PARSE_UGRID_ON_LOAD
import numpy
from function_space import FunctionSpace

class ExtensiveField(object):

    def __init__(self, filename: Path, planet_radius: float=1.0):
        """
        Constructor
        :param filename: Ugrid2d NetCDF file name
        :param planet_radius: radius of the planet
        """
        self.filename = filename
        self.planet_radius = planet_radius
        self.grid = mint.Grid()
        self.numEdges = 0
        self.numPoints = 0
        self.numFaces = 0
        self.u = None
        self.v = None
        self.dims = []

    def build(self, 
            u_std_name: str="eastward_wind_at_cell_faces", 
            v_std_name: str="northward_wind_at_cell_faces"):
        """
        Build the field's mesh and connectivity
        :param u_std_name: standard name of the zonal component of the vector field
        :param v_std_name: standard name of the meridional component of the vector field
        """

        # must have at least one component defined
        assert(u_std_name or v_std_name)

        # get the horizontal velocity components
        with PARSE_UGRID_ON_LOAD.context():

            if u_std_name:
                self.u = iris.load_cube(self.filename, u_std_name)
                assert(self.u.location == 'edge')
                self.dims = self.u.shape
            
            if v_std_name:
                self.v = iris.load_cube(self.filename, v_std_name)
                assert(self.v.location == 'edge')
                self.dims = self.v.shape

        assert(len(self.dims) > 0)

        # set the undfined component to zero (if need be)
        if not self.u:
            self.u = numpy.zeros(self.dims, numpy.float64)
        if not self.v:
            self.v = numpy.zeros(self.dims, numpy.float64)


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


    def get_grid(self):
        """
        Get the grid (or mesh)
        :return grid object
        """
        return self.grid


    def get_dims(self):
        """
        Get the dimensions of the edge field
        :return array of sizes
        """
        return self.dims


    def get_num_faces(self):
        """
        Get the number of faces (cells)
        """
        return self.numFaces


    def get_num_edges(self):
        """
        Get the number of edges
        """
        return self.numEdges


    def get_num_points(self):
        """
        Get the number of points
        """
        return self.numPoints


    def compute_edge_integrals(self, space: FunctionSpace):
        """
        Compute the edge integrals and store the result
        :param space: function space, either FunctionSpace.w1 or FunctionSpace.w2h
        :return array of size num edges
        """

        # get the mid edge locations
        edge_y = self.u.mesh.edge_coords.edge_y.points

        # in radians
        edge_y *= numpy.pi/180.
        a_cos_lat = self.planet_radius * numpy.cos(edge_y)

        efc = mint.ExtensiveFieldConverter()
        efc.setGrid(self.grid)

        # assume last dimension is number of edges
        # the extensive field is always cell by cell
        dms = self.dims[:-1] + (self.numFaces*mint.NUM_EDGES_PER_QUAD,)
        self.edge_integrated = numpy.empty(dms, numpy.float64)

        deg2rad = numpy.pi/180

        # choose between W1 and W2h fields
        getData = efc.getFaceData
        if space == FunctionSpace.w1:
            getData = efc.getEdgeData
        
        mai = mint.MultiArrayIter(self.dims[:-1]) # assume last dimension is number of edges
        mai.begin()
        for _ in range(mai.getNumIters()):

            inds = tuple(mai.getIndices())

            # assume last dimension is number of edges
            slab = inds + (slice(0, self.numEdges),)

            # read a slab of data
            u = self.u[slab]
            v = self.v[slab]

            # conversion factors to get fluxes in m^2/s
            u *= self.planet_radius * deg2rad
            v *= a_cos_lat * deg2rad

            # compute the edge integrated field
            extensive_data = getData(u.data, v.data, placement=mint.UNIQUE_EDGE_DATA)
            slab_cell_by_cell = inds + (slice(0, self.numFaces*mint.NUM_EDGES_PER_QUAD),)
            self.edge_integrated[slab_cell_by_cell] = extensive_data

            # uncomment if want to multiply by dz
            # self.edge_integrated[slce] *= elevs[:]
            # want to multiply by density?
            mai.next()

        return self.edge_integrated


############################################################################
def main(*, filename: Path='./lfric_diag.nc',
            space: FunctionSpace=FunctionSpace.w2h):

    ef = ExtensiveField(filename=filename)
    ef.build()
    edge_integrals = ef.compute_edge_integrals(space)
    emin = edge_integrals.min()
    emax = edge_integrals.max()
    eavg = edge_integrals.mean()
    estd = edge_integrals.std()
    print(f'field edge integrals min, max, mean, std: {emin}, {emax}, {eavg}, {estd}')


if __name__ == '__main__':
    defopt.run(main)
