from pathlib import Path
from extensive_field import ExtensiveField
from polyline import Polyline
from lateral_flux import LateralFlux
from function_space import FunctionSpace
import defopt
import numpy



def main(*, filename: Path='./lfric_diag.nc',
            target_line: str='[(-180., -85.), (180., 85.)]',
            u_std_name: str='', v_std_name: str='', 
            func_space: FunctionSpace=FunctionSpace.w2h):

    ef = ExtensiveField(filename=filename)
    ef.build(u_std_name=u_std_name, v_std_name=v_std_name)
    ef.compute_edge_integrals(func_space)

    xy = numpy.array(eval(target_line))
    line = Polyline(xy, planet_radius=1.0)
    lengths = line.get_lengths()

    lf = LateralFlux(ef)
    lf.set_target_line(line)

    lf.compute_fluxes()
    fluxes = lf.get_fluxes()
    print(f'line averages of the vector field: {fluxes/lengths.sum()}')


if __name__ == '__main__':
    defopt.run(main)
