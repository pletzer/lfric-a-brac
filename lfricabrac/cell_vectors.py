from attr import field
import mint
from pathlib import Path

from traitlets import Bool
from . import FunctionSpace
import numpy
import defopt
from functools import reduce
from operator import __add__

from . import ExtensiveField

class CellVectors(object):

    def __init__(self, ef: ExtensiveField):

        self.ef = ef
        self.vi = mint.VectorInterp()
        grid = ef.get_grid()
        self.vi.setGrid(grid)
        points = grid.getPoints()
        # compute the points in the middle of the cell
        self.cell_points = 0.25*points.sum(axis=1)
        self.vectors = []


    def build(self):
        self.vi.buildLocator(numCellsPerBucket=128, periodX=360.0, enableFolding=False)
        self.vi.findPoints(self.cell_points)


    def attach_vectors_to_grid(self, time_index=None, z_index=None, func_space=FunctionSpace.w2h):
        """
        Attach interpolated vector values to the grid
        :param time_index: time index, use None to attach all the time values
        :param z_index: elevation index, Use None to attach all the elevation values
        :param func_space: function space, e.g. FunctionSpace.w2h
        """

        # may want to get the vectors
        if len(self.vectors) == 0:
            self.vectors = self.get_vectors(func_space=func_space)
    
        dims = self.vectors.shape[:-2] # exclude num_cells and components
        num_edges = self.vectors.shape[-2]
        grd = self.ef.get_grid()
        mai = mint.MultiArrayIter(dims) # assume last dimension is number of edges
        mai.begin()
        for _ in range(mai.getNumIters()):
            inds = tuple(mai.getIndices())
            varname = 'i_' + reduce(__add__, [f'{index:05d}_' for index in inds])
            varname += "vectors"
            all_inds = inds + (slice(0, num_edges), slice(0, 3))
            grd.attach(varname, self.vectors[all_inds])
            mai.next()


    def save_vtk(self, filename):
        grd = self.ef.get_grid()
        grd.dump(filename=filename)


    def get_vectors(self, func_space: FunctionSpace=FunctionSpace.w2h):

        dims = self.ef.get_dims() + (3,)
        res = numpy.empty(dims, numpy.float64)

        getVectors = self.vi.getFaceVectors
        if func_space == FunctionSpace.w1:
            getVectors = self.vi.getEdgeVectors

        mai = mint.MultiArrayIter(self.ef.get_dims()[:-1]) # assume last dimension is number of edges
        mai.begin()
        for _ in range(mai.getNumIters()):

            # get the index set for this iteration
            inds = tuple(mai.getIndices())

            # get the integrated edge values for this time/elev iteration
            slab = inds + (slice(0, self.ef.get_num_faces()*mint.NUM_EDGES_PER_QUAD),)
            data = self.ef.edge_integrated[slab]

            # interpolate the vector field to cell centres
            slab2 = inds + (slice(0, self.ef.get_num_faces()), slice(0, 3))
            res[slab2] = getVectors(data, placement=mint.CELL_BY_CELL_DATA)

            # increment the iterator
            mai.next()

        self.vectors = res
        return res

############################################################################
def main(*, filename: Path='./lfric_diag.nc',
            func_space: FunctionSpace=FunctionSpace.w2h,
            output: str=''):

    ef = ExtensiveField(filename=filename)
    ef.build()
    ef.compute_edge_integrals(func_space)

    cv = CellVectors(ef)
    cv.build()
    vecs = cv.get_vectors(func_space)

    if output:
        cv.attach_vectors_to_grid()
        cv.save_vtk(output)
    else:
        print('vectors:')
        print(vecs)


if __name__ == '__main__':
    defopt.run(main)
