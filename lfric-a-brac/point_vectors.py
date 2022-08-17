from tarfile import ExtractError
from attr import field
import mint
from pathlib import Path
import vtk

from traitlets import Bool
from function_space import FunctionSpace
import numpy
import defopt
from functools import reduce
from operator import __add__

from extensive_field import ExtensiveField

class PointVectors(object):

    def __init__(self, ef: ExtensiveField):

        self.ef = ef
        self.vi = mint.VectorInterp()
        grid = ef.get_grid()
        self.vi.setGrid(grid)
        self.points = []
        self.vectors = []


    def set_points(self, points):
        assert(len(points.shape) == 2)
        assert(points.shape[1] == 3) # 3D
        self.points = points


    def build(self):
        self.vi.buildLocator(numCellsPerBucket=128, periodX=360.0, enableFolding=False)
        self.vi.findPoints(self.points)


    def save_vtk(self, filename):
        # create grid, points, point array and vector array
        grd = vtk.vtkPolyData()
        pts = vtk.vtkPoints()
        ptar = vtk.vtkDoubleArray()

        # connect
        grd.SetPoints(pts)
        num_points = self.points.shape[0]
        num_cells = num_points # the cells are verts
        not_used = 1
        grd.Allocate(num_cells, not_used)

        # build the connectivity
        ptIds = vtk.vtkIdList()
        ptIds.SetNumberOfIds(1) # one point per cell
        for i in range(num_cells):
            ptIds.SetId(0, i)
            grd.InsertNextCell(vtk.VTK_VERTEX, ptIds)

        ptar.SetNumberOfComponents(3)
        ptar.SetNumberOfTuples(num_points)
        ptar.SetName('points')
        save = 1
        ptar.SetVoidArray(self.points, num_points*3, save)
        pts.SetData(ptar)

        vcarrs = {}
        extra_dims = self.ef.get_dims()[:-1]
        dst_slab = (slice(0, num_points),)
        mai = mint.MultiArrayIter(extra_dims)
        mai.begin()
        for _ in range(mai.getNumIters()):

            inds = tuple(mai.getIndices())

            # name of the vector field
            varname = 'i_' + reduce(__add__, [f'{index:05d}_' for index in inds])
            varname += "vectors"

            all_inds = inds + dst_slab
            va = vtk.vtkDoubleArray()
            va.SetNumberOfComponents(3)
            va.SetNumberOfTuples(num_points)
            va.SetName(varname)

            save = 1
            va.SetVoidArray(self.vectors[all_inds], num_points*3, save)

            # attach to grid
            grd.GetPointData().AddArray(va)

            vcarrs[varname] = va # store

            # increment the iterator
            mai.next()

        # create the writer
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileVersion(42) # old format so we can read with using VisIt
        writer.SetFileName(filename)
        writer.SetInputData(grd)
        # write
        writer.Write()


    def get_vectors(self, space: FunctionSpace=FunctionSpace.w2h):

        # time, elevation, ... dimensions
        extra_dims = self.ef.get_dims()[:-1]

        # vectors are over time, elev, num_points, 3
        res = numpy.empty(extra_dims + self.points.shape, numpy.float64)
        num_points = self.points.shape[0]

        # select the interpolation method according to the function space
        getVectors = self.vi.getFaceVectors
        if space == FunctionSpace.w1:
            getVectors = self.vi.getEdgeVectors

        dst_layer_slab = (slice(0, num_points), slice(0, 3))
        # data is cell by cell
        num_edges = self.ef.get_num_faces()*mint.NUM_EDGES_PER_QUAD
        src_layer_slab = (slice(0, num_edges),)

        # iterate over time, elevation, ...
        mai = mint.MultiArrayIter(extra_dims) # assume last dimension is number of edges
        mai.begin()
        for _ in range(mai.getNumIters()):

            # get the index set for this iteration (e.g. time and elevation index)
            inds = tuple(mai.getIndices())

            # get the integrated edge values for this iteration
            slab = inds + src_layer_slab
            data = self.ef.edge_integrated[slab]

            # interpolate the vector field to cell centres
            slab2 = inds + dst_layer_slab
            res[slab2] = getVectors(data, placement=mint.CELL_BY_CELL_DATA)

            # increment the iterator
            mai.next()

        self.vectors = res
        return res

############################################################################
def main(*, filename: Path='./lfric_diag.nc',
            space: FunctionSpace=FunctionSpace.w2h,
            points: str='[(0., 10.), (20., 30.)]',
            output: str=''):

    from numpy import linspace

    ef = ExtensiveField(filename=filename)
    ef.build()
    ef.compute_edge_integrals(space)

    cv = PointVectors(ef)
    pts = numpy.array([(p[0], p[1], 0.0) for p in eval(points)])
    cv.set_points(pts)
    cv.build()
    vecs = cv.get_vectors(space)
    # print(vecs)

    if output:
        cv.save_vtk(output)

if __name__ == '__main__':
    defopt.run(main)
