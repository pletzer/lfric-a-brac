#
# Compute the lateral flux across an irregular boundary
#
import defopt
import mint
from pathlib import Path
import numpy
from . import FunctionSpace
from . import ExtensiveField
from . import Polyline

class LateralFlux(object):

    def __init__(self, extensive_field):
        """
        Constructor
        :param extensive_field: instance of ExtensiveField
        """
        self.ef = extensive_field
        self.pli = mint.PolylineIntegral()
        self.pli.setGrid(self.ef.get_grid())
        self.pli.buildLocator(numCellsPerBucket=128, periodX=360., enableFolding=False)


    def set_target_line(self, xy):
        # need to turn into 3D, still working with lons, lats
        self.xyz = numpy.zeros((xy.shape[0], 3), numpy.float64)
        self.xyz[:, 0] = xy[:, 0]
        self.xyz[:, 1] = xy[:, 1]
        self.pli.computeWeights(self.xyz)


    def compute_fluxes(self):

        # assume last dimension is number of edges
        self.fluxes = numpy.empty(self.ef.get_dims()[:-1], numpy.float64)

        mai = mint.MultiArrayIter(self.ef.get_dims()[:-1]) # assume last dimension is number of edges
        mai.begin()
        for _ in range(mai.getNumIters()):

            # get the index set for this iteration
            inds = tuple(mai.getIndices())

            # get the integrated edge values for this time/elev iteration
            slab = inds + (slice(0, self.ef.get_num_faces()*mint.NUM_EDGES_PER_QUAD),)
            data = self.ef.edge_integrated[slab]

            # retrieve the flux for this time/elev and store its value. Note that the 
            # flux will be in m^2/s since u,v have been multiplied by A and A*cos(y),
            # respectively. 
            self.fluxes[inds] = self.pli.getIntegral(data, placement=mint.CELL_BY_CELL_DATA)

            # increment the iterator
            mai.next()

    def get_fluxes(self):
        return self.fluxes


############################################################################
def main(*, filename: Path='./lfric_diag.nc',
         func_space: FunctionSpace=FunctionSpace.W2H,
         target_line: str='[(-180., -85.), (180., 85.)]'):

    ef = ExtensiveField(filename=filename)
    ef.build()
    ef.compute_edge_integrals(func_space=func_space)

    xy = numpy.array(eval(target_line))
    line = Polyline(xy, planet_radius=1.0)

    lf = LateralFlux(ef)
    lf.set_target_line(line)

    lf.compute_fluxes()
    fluxes = lf.get_fluxes()
    print(f'integrated fluxes: {fluxes}')


if __name__ == '__main__':
    defopt.run(main)


