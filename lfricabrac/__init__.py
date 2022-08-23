from enum import IntEnum
class FunctionSpace(IntEnum):
    W1 = 1
    W2H = 2

__all__ = ('cell_vectors', 'point_vectors', 'extensive_field', 'lateral_flux',
           'line_average_vector', 'polyline', FunctionSpace.W1, FunctionSpace.W2H)



# from .cell_vectors import CellVectors
# from .point_vectors import PointVectors
from .extensive_field import ExtensiveField
from .lateral_flux import LateralFlux
from .polyline import Polyline
