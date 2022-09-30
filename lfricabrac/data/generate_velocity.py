import defopt
from pathlib import Path
import iris
from iris.experimental.ugrid import PARSE_UGRID_ON_LOAD

import xarray # for the time being

import numpy
from lfricabrac import FunctionSpace
import sympy as sym

import sys
import time


def main(*, filename: Path='./cs2.nc',
            func_space: FunctionSpace=FunctionSpace.W2H,
            stream_func: str='6371e3 * (sin(y) + cos(y)*cos(x))'):
    
    # vector components
    x = sym.Symbol('x') # lons in radian
    y = sym.Symbol('y') # lats in radian

    a = sym.Symbol('A') # planet radius
    planet_radius = 6371.e3


    if func_space == FunctionSpace.W2H:
      u_expr = (sym.diff(stream_func, y) / a).subs(a, planet_radius)
      v_expr = (-sym.diff(stream_func, x) / (a * sym.cos(y))).subs(a, planet_radius)
    elif func_space == FunctionSpace.W1:
      u_expr = (sym.diff(stream_func, x) / a).subs(a, planet_radius)
      v_expr = (sym.diff(stream_func, y) / (a * sym.cos(y))).subs(a, planet_radius)
    else:
      raise(RuntimeError, "Invalid function space")

    # get the edges coordinates


    # we should be using Iris to do this....

    # with PARSE_UGRID_ON_LOAD.context():
        
    #     grid = iris.experimental.ugrid.load.load_mesh(filename, var_name='cs')
    #     or 
    #     grid_cube = iris.load(filename)

    nc = xarray.open_dataset(filename)
    mesh_name = 'cs'

    x_name, y_name = nc[mesh_name].node_coordinates.split()
    deg2rad =  numpy.pi/180.
    xnodes = nc[x_name][:] * deg2rad
    ynodes = nc[y_name][:] * deg2rad

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

    print(uedges)
    print(vedges)

    # create a new dataset
    ds = xarray.Dataset(
        {'u_in_w2h': (
            ['ncs_edge',],
            uedges,
            {'long_name': 'eastward_wind_at_cell_faces',
                'units': 'm s-1',
                'mesh': 'cs',
                'location': 'edge',
                'coordinates': 'cs_edge_y cs_edge_x',
            }
          ),
        'v_in_w2h': (
            ['ncs_edge',],
            vedges,
            {'long_name': 'northward_wind_at_cell_faces',
             'units': 'm s-1',
             'mesh': 'cs',
             'location': 'edge',
             'coordinates': 'cs_edge_y cs_edge_x',
            }
          ),
        'edge_integrals': (
            ['ncs_edge',],
            edge_integrals.data,
            {'long_name': 'wind_integrated_at_cell_faces',
             'units': 'm2 s-1',
             'mesh': 'cs',
             'location': 'edge',
             'coordinates': 'cs_edge_y cs_edge_x',
            }
          ),
        'cs_edge_x': (
            ['ncs_edge',],
            xedges.data,
            {'standard_name': 'longitude',
             'long_name': 'Characteristic longitude of mesh edges.',
             'units': 'degrees_east',
            }
            ),
        'cs_edge_y': (
            ['ncs_edge',],
            yedges.data,
            {'standard_name': 'latitude',
             'long_name': 'Characteristic latitude of mesh edges.',
             'units': 'degrees_north',
            }
            ),
        },
        # global attributes
        attrs=dict(command=' '.join(sys.argv), time=time.asctime(),
            stream_function=stream_func, filename=str(filename))
    )


    # add the mesh and the connectivity
    for k in nc: 
        ds[k] = nc[k]

    # add the edge_coordinates attribute to the mesh
    ds['cs'].attrs['edge_coordinates'] = 'cs_edge_x cs_edge_y'

    filename_out = str(filename).split('.nc')[0] + '_wind.nc'
    print(f'saving the wind components in file {filename_out}')
    ds.to_netcdf(filename_out)


if __name__ == '__main__':
    defopt.run(main)

