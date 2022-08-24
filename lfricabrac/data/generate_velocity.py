import defopt
from pathlib import Path
import iris
from iris.experimental.ugrid import PARSE_UGRID_ON_LOAD

import xarray # for the time being

import numpy
# from function_space import FunctionSpace
import sympy as sym

import sys
import time


def main(*, filename: Path='./cs2.nc',
            # func_space: FunctionSpace=FunctionSpace.w2h,
            stream_func: str='6371e3 * (sin(y) + cos(y)*cos(x))'):
    
    # vector components
    x = sym.Symbol('x') # lons in radian
    y = sym.Symbol('y') # lats in radian

    a = sym.Symbol('A') # planet radius
    planet_radius = 6371.e3

    u_expr = (sym.diff(stream_func, y) / a).subs(a, planet_radius)
    v_expr = (-sym.diff(stream_func, x) / (a * sym.cos(y))).subs(a, planet_radius)

    # get the edges coordinates


    # we should be using Iris to do this....

    # with PARSE_UGRID_ON_LOAD.context():
        
    #     grid = iris.experimental.ugrid.load.load_mesh(filename, var_name='cs')
    #     or 
    #     grid_cube = iris.load(filename)

    nc = xarray.open_dataset(filename)
    mesh_name = 'cs'

    x_name, y_name = nc[mesh_name].node_coordinates.split()
    xnodes = nc[x_name][:] * numpy.pi/180.
    ynodes = nc[y_name][:] * numpy.pi/180.

    edge_node_connect_name = nc[mesh_name].edge_node_connectivity
    edge_node_connect = nc[edge_node_connect_name][:] - nc[edge_node_connect_name].start_index

    xbeg = xnodes[edge_node_connect[:, 0]]
    xend = xnodes[edge_node_connect[:, 1]]
    ybeg = ynodes[edge_node_connect[:, 0]]
    yend = ynodes[edge_node_connect[:, 1]]
    xedges = 0.5*(xbeg + xend) # edge mid point
    yedges = 0.5*(ybeg + yend) # edge mid point

    # evaluate the stream function at the beg/end points of the edges
    from numpy import cos, sin, pi
    x = xbeg
    y = ybeg
    psibeg = eval(stream_func)
    x = xend
    y = yend
    psiend = eval(stream_func)
    edge_integrals = psiend - psibeg

    num_edges = len(xedges)
    uedges = numpy.zeros((num_edges,), numpy.float64)
    vedges = numpy.zeros((num_edges,), numpy.float64)

    # set the vector components at the mid edge positions
    x = xedges
    y = yedges
    uedges[:] = eval(str(u_expr))
    vedges[:] = eval(str(v_expr))

    # create a new dataset
    ds = xarray.Dataset(
        {'u_in_w2h': (
            ['ncs_egde',],
            uedges,
            {'long_name': 'eastward_wind_at_cell_faces',
                'units': 'm s-1',
                'mesh': 'cs',
                'location': 'edge',
            }
          ),
        'v_in_w2h': (
            ['ncs_egde',],
            vedges,
            {'long_name': 'northward_wind_at_cell_faces',
             'units': 'm s-1',
             'mesh': 'cs',
             'location': 'edge',
            }
          ),
        'edge_integrals': (
            ['ncs_egde',],
            edge_integrals.data,
            {'long_name': 'wind_integrated_at_cell_faces',
             'units': 'm2 s-1',
             'mesh': 'cs',
             'location': 'edge',
            }
          )
        },
        # global attributes
        attrs=dict(command=' '.join(sys.argv), time=time.asctime(),
            stream_function=stream_func, filename=str(filename))
    )
    # add the mesh and the connectivity
    for k in 'cs', 'cs_face_nodes', 'cs_edge_nodes', 'cs_node_x', 'cs_node_y': 
        ds[k] = nc[k]

    filename_out = str(filename).split('.nc')[0] + '_wind.nc'
    print(f'saving the wind components in file {filename_out}')
    ds.to_netcdf(filename_out)


if __name__ == '__main__':
    defopt.run(main)








