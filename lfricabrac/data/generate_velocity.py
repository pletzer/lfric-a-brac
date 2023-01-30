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
            scalar_func: str='A * cos(y)*cos(x)',
            zaxis: list[float]=[],
            taxis: list[float]=[]):

    """
    Generate a vector field from a scalar function
    :param Path filename: Ugrid file containing the mesh coordinates
    :param FunctionSpace func_space: either FunctionSpace.W2H or FunctionSpace.W1
    :param str scalar_func: either the stream function (if FunctionSpace.W2H) or the potential function (if FunctionSpace.W1). Should be a function of x, y
    :param list[float] zaxis: elevations
    :param list[float] taxis: times
    """
    
    # vector components
    x = sym.Symbol('x') # lons in radian
    y = sym.Symbol('y') # lats in radian

    a = sym.Symbol('A') # planet radius
    planet_radius = 6371.e3


    if func_space == FunctionSpace.W2H:
      u_expr = (sym.diff(scalar_func, y) / a).subs(a, planet_radius)
      v_expr = (-sym.diff(scalar_func, x) / (a * sym.cos(y))).subs(a, planet_radius)
    elif func_space == FunctionSpace.W1:
      u_expr = (sym.diff(scalar_func, x) / a).subs(a, planet_radius)
      v_expr = (sym.diff(scalar_func, y) / (a * sym.cos(y))).subs(a, planet_radius)
    else:
      raise(RuntimeError, "Invalid function space")

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

    npoints = len(xnodes)
    nedges = len(xbeg)

    # Lon is defined up to a periodicity length. 
    # We compute the mid edge poisition first in cartesian coords
    # then back to lon-lat.
    xyz_beg = numpy.zeros((nedges, 3), numpy.float64)
    xyz_beg[:, 0] = numpy.cos(ybeg) * numpy.cos(xbeg)
    xyz_beg[:, 1] = numpy.cos(ybeg) * numpy.sin(xbeg)
    xyz_beg[:, 2] = numpy.sin(ybeg)

    xyz_end = numpy.zeros((nedges, 3), numpy.float64)
    xyz_end[:, 0] = numpy.cos(yend) * numpy.cos(xend)
    xyz_end[:, 1] = numpy.cos(yend) * numpy.sin(xend)
    xyz_end[:, 2] = numpy.sin(yend)

    # edge mid-points
    xyz_edges = 0.5*(xyz_beg + xyz_end)

    # lons
    xedges = numpy.arctan2(xyz_edges[:, 1], xyz_edges[:, 0]) # rads
    # lats
    rhoedges = numpy.sqrt( xyz_edges[:, 0]*xyz_edges[:, 0] + xyz_edges[:, 1]*xyz_edges[:, 1] )
    yedges = numpy.arctan2(xyz_edges[:, 2], rhoedges) # rads


    if not taxis:
        taxis = [0.]
    nt = len(taxis)

    if not zaxis:
        # must have one level
        if func_space == FunctionSpace.W1:
            zaxis = [0.]
        else:
            # need at least two levels for W2
            zaxis = [0., 1.]

    if func_space == FunctionSpace.W1:
        # full levels
        nelev = len(zaxis)
        elevs = zaxis
        wind_integrated_name = 'wind_integrated_at_cell_edges'
    else:
        # half levels
        nelev = len(zaxis) - 1
        elevs = [0.5*(zaxis[i] + zaxis[i+1]) for i in range(nelev)]
        wind_integrated_name = 'wind_integrated_at_cell_faces'

    # evaluate the scalar function at the beg/end points of the edges
    A = planet_radius
    cos = numpy.cos
    sin = numpy.sin
    edge_integrals = numpy.zeros((nt, nelev, nedges), numpy.float64)
    for it in range(nt):
        for iz in range(nelev):
            x, y = xbeg, ybeg
            psibeg = eval(scalar_func)
            x, y = xend, yend
            psiend = eval(scalar_func)
            edge_integrals[it, iz, :] = psiend - psibeg


    # evaluate the vector components at the mid edge positions
    x = xedges
    y = yedges

    uedges = numpy.zeros((nt, nelev, nedges,), numpy.float64)
    vedges = numpy.zeros((nt, nelev, nedges,), numpy.float64)


    print(f'function space: {func_space}')
    print(f'u = {u_expr}')
    print(f'v = {v_expr}')
    for it in range(nt):
        t = taxis[it]
        for iz in range(nelev):
            z = elevs[iz]
            uedges[it, iz, :] = eval(str(u_expr))
            vedges[it, iz, :] = eval(str(v_expr))

    # rad -> deg
    xedges /= deg2rad
    yedges /= deg2rad

    # create a new dataset
    ds = xarray.Dataset({
        'u_in_w2h': (
            ['nt', 'nelev', 'ncs_edge',],
            uedges,
            {'long_name': 'eastward_wind_at_cell_faces',
                'units': 'm s-1',
                'mesh': 'cs',
                'location': 'edge',
                'coordinates': 'cs_edge_y cs_edge_x',
            }
          ),
        'v_in_w2h': (
            ['nt', 'nelev', 'ncs_edge',],
            vedges,
            {'long_name': 'northward_wind_at_cell_faces',
             'units': 'm s-1',
             'mesh': 'cs',
             'location': 'edge',
             'coordinates': 'cs_edge_y cs_edge_x',
            }
          ),
        'edge_integrals': (
            ['nt', 'nelev', 'ncs_edge',],
            edge_integrals,
            {'long_name': wind_integrated_name,
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
        'elevs': (
            ['nelev',],
            elevs,
            ),
        'time': (
            ['nt',],
            taxis,
            {'standard_name': 'time',
             'calendar': 'gregorian',
            'units': 'days since 2000-01-01',
            }
            ),
        },
        # global attributes
        attrs=dict(command=' '.join(sys.argv), time=time.asctime(),
            scalar_function=scalar_func, filename=str(filename))
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

