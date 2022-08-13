import mint
from pathlib import Path
from function_space import FunctionSpace
import numpy
import defopt

from extensive_field import ExtensiveField

class CellVectors(object):

    def __init__(self, ef: ExtensiveField):

        self.ef = ef
        self.vi = mint.VectorInterp()
        grid = ef.get_grid()
        self.vi.setGrid(grid)
        points = grid.getPoints()
        # compute the points in the middle of the cell
        self.cell_points = 0.25*points.sum(axis=1)


    def build(self):
        self.vi.buildLocator(numCellsPerBucket=128, periodX=360.0, enableFolding=False)
        self.vi.findPoints(self.cell_points)


    def get_vectors(self, space: FunctionSpace=FunctionSpace.w2h):

        dims = self.ef.get_dims() + (3,)
        res = numpy.empty(dims, numpy.float64)

        getVectors = self.vi.getFaceVectors
        if space == FunctionSpace.w1:
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

        return res

############################################################################
def main(*, filename: Path='./lfric_diag.nc', space: FunctionSpace=FunctionSpace.w2h):

    ef = ExtensiveField(filename=filename)
    ef.build()
    ef.compute_edge_integrals(space)

    cv = CellVectors(ef)
    cv.build()
    vecs = cv.get_vectors(space)
    print(vecs)

if __name__ == '__main__':
    defopt.run(main)
