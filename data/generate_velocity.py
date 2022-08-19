import defopt
from pathlib import Path
import iris
from iris.experimental.ugrid import PARSE_UGRID_ON_LOAD

import xarray # for the time being

import numpy
# from function_space import FunctionSpace
import sympy as sym


def main(*, filename: Path='./cs2.nc',
            # func_space: FunctionSpace=FunctionSpace.w2h,
            stream_func: str='x'):
    
    # vector components
    x = sym.Symbol('x') # lons in radian
    y = sym.Symbol('y') # lats in radian

    a = sym.Symbol('A') # planet radius

    u_expr = (sym.diff(stream_func, y) / a).subs(a, 1.0)
    v_expr = (-sym.diff(stream_func, x) / (a * sym.cos(y))).subs(a, 1.0)

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
    print(f'xnodes = {xnodes}')
    print(f'ynodes = {ynodes}')


    edge_node_connect_name = nc[mesh_name].edge_node_connectivity
    edge_node_connect = nc[edge_node_connect_name][:] - nc[edge_node_connect_name].start_index
    print(f'edge_node_connect = {edge_node_connect}')

    xbeg = xnodes[edge_node_connect[:, 0]]
    xend = xnodes[edge_node_connect[:, 1]]
    ybeg = ynodes[edge_node_connect[:, 0]]
    yend = ynodes[edge_node_connect[:, 1]]
    xedges = 0.5*(xbeg + xend) # edge mid point
    yedges = 0.5*(ybeg + yend) # edge mid point

    # evaluate the stream function at the nodes
    x = xnodes
    y = ynodes
    from numpy import cos, sin, pi
    psis = eval(stream_func)


    # evaluate the vector field at the mid edge points
    x = xedges
    y = yedges

    num_edges = len(xedges)
    uedges = numpy.zeros((num_edges,), numpy.float64)
    vedges = numpy.zeros((num_edges,), numpy.float64)

    uedges[:] = eval(str(u_expr))
    vedges[:] = eval(str(v_expr))

    print(f'x edges: {xedges}')
    print(f'y edges: {yedges}')
    print(f'u edges: {uedges}')
    print(f'v edges: {vedges}')

if __name__ == '__main__':
    defopt.run(main)








