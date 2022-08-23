#
# Represent a broken line in longitude-latitude space
#
import defopt
import numpy

class Polyline(object):

    def __init__(self, xy, planet_radius: float=1.0):
        """
        Constructor
        :param xy 2d array of size num_points*2 for the longitude, latitude points
        :param planet_radius: radius of the planet
        """
        self.xy = xy
        self.shape = xy.shape
        self.planet_radius = planet_radius


    def __getitem__(self, slc):
        return self.xy[slc]


    def get_lengths(self):
        """
        Get the horizontal lengths of each polyline segment
        """
        # Cartesian points
        num_points = self.xy.shape[0]
        xyz = numpy.zeros((num_points, 3), numpy.float64)
        deg2rad = numpy.pi/180.0
        
        lons, lats = self.xy[:, 0]*deg2rad, self.xy[:, 1]*deg2rad # in radians

        # compute the Cartesian coords
        rhos = numpy.cos(lats)
        # radius of one
        xyz[:, 0] = rhos * numpy.cos(lons) # x
        xyz[:, 1] = rhos * numpy.sin(lons) # y
        xyz[:, 2] = numpy.sin(lats)

        rr0 = xyz[:-1, :]
        rr1 = xyz[1:, :]
        # dot product
        angles = numpy.arccos(
            rr0[:, 0]*rr1[:, 0] + \
            rr0[:, 1]*rr1[:, 1] + \
            rr0[:, 2]*rr1[:, 2] \
                )
        return self.planet_radius * angles


############################################################################
def main(*, target_line: str='[(-180., -85.), (180., 85.)]'):

    xy = numpy.array(eval(target_line))

    pl = Polyline(xy, planet_radius=1.0)
    lengths = pl.get_lengths()
    print(f'total length is: {lengths.sum()}')


if __name__ == '__main__':
    defopt.run(main)
