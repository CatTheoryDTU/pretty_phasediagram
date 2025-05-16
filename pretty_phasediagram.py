#!/usr/bin/env python

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from itertools import combinations,product
import argparse
import pprint
import networkx as nx
from shapely.geometry import Polygon
import string

# Phase diagram with example functions to make sure all borders work out

def main():
    ap = argparse.ArgumentParser(description="Create an example phase diagram for e.g. free energy as a function" + \
                                             " of two variables, e.g. temperature and pressure. To use, I suggest" + \
                                             " importing 'plot_boundaries'. To see where boundary nodes are detected" + \
                                             " in the grid, run '-c scattergrid'")
    ap.add_argument("-T","--temperature-range",type=float,nargs='*',default=[300.,600.])
    ap.add_argument("--TC",nargs=2,default=None,type=float,
                    help="Temperature range in Celsius. Default: None, all available data")
    ap.add_argument("-p","--log-pressure-range",type=float,nargs='*',default=[0.,6.])
    ap.add_argument("-g","--graph-step",type=float,default=0.01,
                    help="Step in graph coords when sampling phase boundary. Default: 0.01")
    ap.add_argument("-c","--color", default="bounding_polygons",
                    choices=["white","colorgrid","scattergrid","bounding_polygons"],
                    help="Color type. Default: 'bounding_polygons'. Other options: 'colorgrid' (filled gridpoints, " + \
                         "looks better with high number of gridpoints). Other options: 'white' (no filling)" + \
                         ",'scattergrid' (scatter gridpoints and mark boundary nodes and trinodes, )")
    ap.add_argument("-o","--only-colorgrid",action="store_true",help="Plot only colorgrid. No lines.")
    ap.add_argument("--cmap",default="Set2",help="Colormap. Default: 'Set2'")
    ap.add_argument("-s","--stripe_phase_labels",nargs="*",default=[],
                    help="Phases with stripes. Only works with '-c bounding_polygons'")
    ap.add_argument("--xlabel",default="Temperature (K)")
    ap.add_argument("--ylabel",default="Log pressure")
    ap.add_argument("--title",default="Example phase diagram")
    ap.add_argument("-C","--celsius",action="store_true",help="Display temperature in C. (-T is still Kelvin)")
    ap.add_argument("-f","--fontsize",type=float,default=12.,
                    help="Fontsize for smallest text. Axis labels and titles follow proportionally")
    
    args = ap.parse_args()
    
    # Example functions

    Gfuncs = (
              lambda T,p: 10 + 0.01 * T + 0.5 * p,
              lambda T,p: 12 + 0.005 * T + p**2 - 5 * p,
              lambda T,p: 15 - 0.02 * T + 1.5 * p,
              #lambda T,p: 9.0,
              lambda T,p: 20 - 0.03 * T + 0.5 * p**2,
              lambda T,p: 18 - 0.025 * T + p,
             )
    """
    Gfuncs = (
              lambda T,p: -1 * p * (p - 6), # max 9
              lambda T,p: 8,
              lambda T,p: 20 * (p - 0.5),
              lambda T,p: 20 * (5.5 - p),
              lambda T,p: 10 * (T - 325),
              lambda T,p: 10 * (575 - T),
              lambda T,p: 10 * (T - 325) + 0.2 * (p - 2.5),
              lambda T,p: 10 * (575 - T) + 0.2 * (p - 3.5),
              lambda T,p: 8 + 0.1 * (p - 3)
             )
    """
    Trange = args.temperature_range
    
    celsius = args.celsius
    if args.TC != None:
        Trange = [T + 273.15 for T in args.TC]
        celsius = True
        
    if celsius:
        xlabel = "Temperature (°C)"
    else:
        xlabel = args.xlabel

    prange = args.log_pressure_range
    labels = np.array(list(string.ascii_uppercase[:len(Gfuncs)]))
    
    if args.only_colorgrid:
        plot_colorgrid(Trange,prange,Gfuncs,num_gridpoints=100,colormap='Set2',labels=labels,
                       fontsize=args.fontsize,xlabel=xlabel,ylabel=args.ylabel,title=args.title)
    else:
        plot_diagram(Trange,prange,Gfuncs,
                     graph_step=args.graph_step,
                     plot=True,celsius=celsius,plot_labels=True,
                     color=args.color,colormap=args.cmap,
                     stripe_phase_labels=args.stripe_phase_labels,
                     labels=labels,fontsize=args.fontsize,
                     xlabel=xlabel,ylabel=args.ylabel,title=args.title)
    

def plot_colorgrid(xlims,ylims,funcs,num_gridpoints=25,colormap='Set2',labels=None,
                   xlabel=None,ylabel=None,title=None,fontsize=10.):
    # Make colorplot on grid without boundary lines
    lims = (xlims,ylims)
    if labels is None:
        labels = list(range(len(funcs)))
    assert len(labels) == len(funcs)
    xgrid, ygrid, funcs, phases = get_grid_and_phases(xlims,ylims,funcs,labels,num_gridpoints)
    colorplot(phases,xgrid,ygrid,colormap=colormap,labels=labels,fontsize=fontsize)
    plt.show()
    return


def get_labels(inds,labels=None):
    # Get labels for phase indices. Works recursively for nested tuples/sets/lists (converted to tuples)
    if labels is None:
        return inds
    elif isinstance(inds,np.integer):
        return labels[inds]
    elif isinstance(inds,(tuple,list,set,np.ndarray)):
        return tuple(get_labels(ind,labels) for ind in inds)
    else:
        return inds


def plot_diagram(xlims,ylims,funcs,num_gridpoints=25,graph_step=0.02,init_graph_step=None,
                 plot=True,celsius=False,plot_labels=True,
                 color='white',colormap='Set2',stripe_phase_labels=[],
                 labels=None,
                 xlabel=None,ylabel=None,title=None,fontsize=10.):
    # xlims,ylims:          E.g., temperature in K [300,600], or log pressure [0,6].
    # funcs:                List of functions of x,y, corresponding to phases.
    # num_gridpoints:       Number of grid points in x,y to sample phases. Must be dense enough to have all
    #                       phases optimal on the domain sampled at least once and preferably a little more.
    # graph_step:           Step in graph coords for sampling phase boundary. Decrease for finer sampling
    # init_graph_step:      Initial graph step. Default half of graph_step.
    # plot:                 Plot anything (e.g. boundaries, labels).
    # celsius:              Treats x axis as temperatures in K and display in degrees C.
    # plot_labels:          Plot phase labels inside diagram.
    # color:                "colorgrid" (filled gridpoints, looks better with high number of gridpoints)
    #                       "white" (no filling), "scattergrid"s (scatter gridpoints and mark nodes)
    # colormap:             E.g. "Set2","Dark2","Paired"
    # stripe_phase_labels   Phases whose regions is striped
    # labels:               Phase labels
    # xlabel,ylabel,title   Pyplot parameters
    #
    # Returns
    #
    # graph:                Mathematical graph with nodes e.g. (2,3,4) or (2,3,'xmin') with position 'pos',
    #                       and edges with 'interpolation' (xs,ys) if needed.
    # bounding_polygons     Dict of coords bounding a phase, e.g. {0: (xs,ys), ...} (including figure edge)
    
    lims = (xlims,ylims)
    if labels is None:
        labels = list(range(len(funcs)))
    assert len(labels) == len(funcs)
    xgrid, ygrid, funcs, phases = get_grid_and_phases(xlims,ylims,funcs,labels,num_gridpoints)
    
    # Get mathematical graph with phase and figure nodes
    graph, grid_highlights = get_graph(funcs,phases,xgrid,ygrid,labels=labels,graph_step=graph_step)
    
    bounding_polygons = get_bounding_polygons3(graph)
    
    if plot:
        if color == 'colorgrid':
            figsize=(8, 6)
        else:
            figsize=(6.5, 6)
    
        plt.figure(figsize=figsize)
        
        if plot_labels:
            plot_phaselabels(bounding_polygons,phase_labels=labels,funcs=funcs,fontsize=fontsize)
                
        if color == 'colorgrid':
            colorplot(phases,xgrid,ygrid,colormap=colormap,labels=labels,fontsize=fontsize)
        elif color == 'scattergrid':
            scatterplot(xgrid,ygrid,phases,colormap=colormap)
            for phase_tuple in grid_highlights:
                box, = plt.plot(*grid_highlights[phase_tuple],c='r')
                box.set_clip_on(False)
        elif color == 'bounding_polygons':
            plot_bounding_polygons2(bounding_polygons,phase_labels=labels,colormap=colormap,
                                    xlims=xlims,ylims=ylims,stripe_phase_labels=stripe_phase_labels)

        plot_boundary_lines(graph)
            
        plt.xlim(*xlims)
        plt.ylim(*ylims)

        if celsius:
            ax = plt.gca()
            
            # Extract original tick locations in Kelvin
            original_ticks_K = ax.get_xticks()
            K_spacing = np.round(np.mean(np.diff(original_ticks_K)))  # Estimate spacing

            # Align to the closest rounded Celsius multiple
            K_start = np.ceil((original_ticks_K[0] - 273.15) / K_spacing) * K_spacing + 273.15
            K_end = np.floor((original_ticks_K[-1] - 273.15) / K_spacing) * K_spacing + 273.15
            K_ticks = np.arange(K_start, K_end + K_spacing, K_spacing)
            
            # Set new tick positions and labels
            ax.set_xticks(K_ticks)
            ax.set_xticklabels([f"{T - 273.15:.0f}" for T in K_ticks])
            
            #Reapply domain
            plt.xlim(*xlims)
            
            plt.xlabel("Temperature (°C)")
        
        
        plt.xlabel(xlabel,fontsize=fontsize*12/10)
        plt.ylabel(ylabel,fontsize=fontsize*12/10)
        plt.title(title,fontsize=fontsize*16/10)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.show()
    return graph, bounding_polygons


def get_grid_and_phases(xlims,ylims,funcs,labels,num_gridpoints=25):
    # Get x,y, function grids and make sure it work with vector inputs
    xs = np.linspace(*xlims,num_gridpoints)
    ys = np.linspace(*ylims,num_gridpoints)

    xgrid,ygrid = np.meshgrid(xs,ys)
    
    def wrap_func(f):
        # Wrap to catch constant functions
        def wrapped(x, y):
            result = f(x, y)
            if np.isscalar(result):
                return np.full_like(x, result, dtype=float)
            return result
        return wrapped

    
    #-----------Calculate phases---------------

    fgrids = []
    vector_funcs = []
    for f in funcs:
        # If the function can't take array inputs,
        # vectorize.
        wrapped = wrap_func(f)
        try:
            fgrid = wrapped(xgrid,ygrid)
            vector_func = wrapped
        except (TypeError, ValueError):
            vector_func = np.vectorize(wrapped)
            fgrid = vector_func(xgrid,ygrid)
        fgrids.append(fgrid)
        vector_funcs.append(vector_func)
     
    fgrids = np.array(fgrids)

    possible_phases = set(range(len(fgrids)))
    ind_pairs = combinations(possible_phases,2)
    
    
    for i,j in ind_pairs:
        if np.allclose(fgrids[i],fgrids[j]):
            # If two phases are equal for all x,y
            # Add 1 to the second phase so that it is never optimal
            fgrids[j] += 1.0
            orig_func = vector_funcs[j]
            vector_funcs[j] = lambda *args, **kwargs: orig_func(*args, **kwargs) + 1.0
            print('phase',labels[j],'removed')
    phases = np.argmin(fgrids, axis=0)
    return xgrid, ygrid, vector_funcs, phases


def plot_boundary_lines(phase_graph,linewidth=1.5,plot_figure_edge=False):
    # Plot lines between internal boundary nodes and edge boundary nodes. Not cornerss
    for node0,node1 in phase_graph.edges:
        if 'interpolation' in phase_graph[node0][node1]:
            plt.plot(*phase_graph[node0][node1]['interpolation'],c='k',lw=linewidth)
        elif plot_figure_edge:
            xsys = straight_line(phase_graph,node0,node1)
            plt.plot(*xsys,c='k',lw=linewidth)
    return


def colorplot(phases, xgrid, ygrid, labels=None,fontsize=10.,colormap='Set2'):
    # Plot colorgrid, i.e., the phase gridpoints. Good for testing, but not pretty.
    
    present_phases = np.unique(phases)
    nb_phases = len(present_phases)
    phase_to_color_index = {phase: i for i, phase in enumerate(present_phases)}
    phases_mapped = np.vectorize(phase_to_color_index.get)(phases)
    
    if labels is None:
        labels = present_phases
    
    cmap = plt.get_cmap(colormap)
    
    # Handle both discrete & continuous colormaps
    colors = cmap.colors[:nb_phases] if hasattr(cmap, "colors") else [cmap(i) for i in range(nb_phases)]
    
    listed_cmap = mcolors.ListedColormap(colors)

    # Use BoundaryNorm to map integer values correctly
    norm = mcolors.BoundaryNorm(np.arange(-0.5, nb_phases+0.5, 1), nb_phases)
    
    plt.pcolormesh(xgrid, ygrid, phases_mapped,cmap=listed_cmap,norm=norm)
    cbar = plt.colorbar()
    cbar.set_ticks(np.arange(len(present_phases)))
    cbar.set_ticklabels([labels[phase] for phase in present_phases])
    cbar.ax.tick_params(labelsize=fontsize)

    return


def scatterplot(xgrid,ygrid,phases=None,colormap='Set2'):
    # Intended to be together with grid highlights
    if phases is None:
        scatter_outside = plt.scatter(xgrid,ygrid,c='#737373',marker='x',s=25)
    else:
        colors = plt.get_cmap(colormap).colors
        colors = np.array(colors)
        colorgrid = np.array(colors)[phases].reshape(-1,3)
        
        #Flattening necessary for RGB colors to work
        scatter_outside = plt.scatter(xgrid.flatten(),ygrid.flatten(),c=colorgrid,marker='x',s=25)
    scatter_outside.set_clip_on(False)

    return


def plot_phaselabels(bounding_polygons,phase_labels=None,funcs=None,fontsize=12):
    # Plot phase labels centered in their region of the domain
    to_labels = lambda inds: get_labels(inds,phase_labels)
    for phase in bounding_polygons:
        centroid = bounding_polygons[phase].centroid # Centre of geometry for phase label
        x,y = centroid.coords[0]
        if funcs is not None:
            min_phases = get_min_phases(funcs,x,y)
            if min_phases != {phase}:
                print('Warning! Phase label',to_labels(phase),'written where',to_labels(min_phases),
                      'is optimal')
        plt.text(x,y,to_labels(phase),ha='center',fontsize=fontsize)
    return


def plot_bounding_polygons2(bounding_polygons,phase_labels=None,colormap='Set2',
                            xlims=None,ylims=None,stripe_phase_labels=[],angle=45,stripe_width=0.05):
    # bounding_polygons - dict of Polygons from shapely.geometry to be plotted by plt.fill
    # phase_labels - List of labels to be written centrally in each phase region
    # colormap
    # xlims,ylims - Plot limits. Only necessary for stripes
    # stripe_phase_labels - list of phase labels to stripe
    # angle - angle of stripes in degrees
    # stripe_width - width of stripe as fraction of figue side length 
        
    #to_labels = lambda inds: get_labels(inds,phase_labels)                        
    colors = plt.get_cmap(colormap).colors
    color_counter = 0
    for phase in bounding_polygons:
        xs,ys = bounding_polygons[phase].exterior.xy
        plt.fill(xs,ys,c=colors[color_counter])
        color_counter += 1
            
    if stripe_phase_labels:
        if xlims is None or ylims is None:
            raise Exception('lims must be set for stripes')
        xmin,xmax = xlims
        ymin,ymax = ylims
        len_x,len_y = xmax - xmin, ymax - ymin
        theta = angle * np.pi / 180
        max_iter = 5./stripe_width
        
    for label in stripe_phase_labels:
        if phase_labels is None:
            phase_labels = list(range(max(bounding_polygons.keys())+1))
        if label not in phase_labels:
            raise Exception('Phase label ' + label + ' not in phase_labels')
        phase = list(phase_labels).index(label)
            
        xs,ys = bounding_polygons[phase].exterior.xy
        
        # Convert to normalized figure coordinates
        us = [(x - xmin)/len_x for x in xs]
        vs = [(y - ymin)/len_y for y in ys]
        vertices = np.array((us,vs))
        
        base_polygon = Polygon(vertices.T)
        
        # Rotation matrix for coordinate conversion
        R = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        new_vertices = np.dot(R,vertices)
        
        # Practical coordinate system w,x
        (w_min,w_max), (z_min,z_max) = [(min(coords), max(coords)) for coords in new_vertices]

        new_counter = 0
       
        w_low = w_min
        while w_low < w_max and new_counter < max_iter:
            w_high = w_low + stripe_width
            stripe_vertices_wz = [[w_low,z_min],[w_high,z_min],[w_high,z_max],[w_low,z_max]]
            stripe_vertices = np.dot(R.T,np.array(stripe_vertices_wz).T)
            stripe_polygon = Polygon(stripe_vertices.T)

            # Compute the intersection of the two polygons
            intersection = base_polygon.intersection(stripe_polygon)

            # Plot the intersection if it exists
            if not intersection.is_empty:
                us_inter, vs_inter = intersection.exterior.xy
                xs_inter = [len_x * u + xmin for u in us_inter]
                ys_inter = [len_y * v + ymin for v in vs_inter]
                plt.fill(xs_inter, ys_inter, c=colors[color_counter], alpha=1.0)
            w_low += 2 * stripe_width
            new_counter += 1
        if new_counter == max_iter:
            raise Exception('Infinte loop for stripes ' + label)
        color_counter += 1
    return


def get_enveloping_rectangle(xs,ys,dx,dy):
    # Make rectangle around points with x and y values xs ys
    xmin,xmax = min(xs) - dx, max(xs) + dx
    ymin,ymax = min(ys) - dy, max(ys) + dy

    #return ((xmin,ymin),(xmin,ymax),(xmax,ymax),(xmax,ymin),(xmin,ymin))
    return ((xmin,xmin,xmax,xmax,xmin),(ymin,ymax,ymax,ymin,ymin))


def solve(G1,G2,constrain_coefs,guess,lims=None):
    # Solve G1(x,y) = G2(x,y) with linear constraint
    # guess - guessed point x,y
    
    cx,cy,c0 = constrain_coefs # Constraint: cx * x + cy * y + c0 = 0
    
    if not np.isclose(guess[0] * cx + guess[1] * cy + c0,0):
        print('Warning! Guess is not on line defined by constrain coefficients!')   

    if cx != 0:
        def f(y): return -cy/cx * y - c0/cx

        def equation(y):
            x = f(y)
            return G1(x,y) - G2(x,y)

        y_solution = fsolve(equation,x0=guess[1])

        if len(y_solution) > 1:
            print('Multuple values',y_solution)
        
        xsol,ysol = f(y_solution[0]),y_solution[0]        
    else:
        y = -c0/cy
        
        def equation(x):
            return G1(x,y) - G2(x,y)
            
        x_solution = fsolve(equation,x0=guess[0])
            
        xsol,ysol = x_solution[0], y 
    if lims != None:    
        (xmin,xmax),(ymin,ymax) = lims
        if not (xmin <= xsol <= xmax and ymin <= ysol <= ymax):
            print('WARNING! Solution',xsol,ysol,'outside of plot boundary')
    return xsol,ysol


def solve3(G1,G2,G3,guess,lims=None):
    # Solve trinode G1(x,y) == G2(x,y) == G3(x,y)
    # guess - guessed point x,y
    
    def equations(vars):
        x,y = vars
        eq1 = G1(x,y) - G2(x,y)
        eq2 = G2(x,y) - G3(x,y)
        return [eq1,eq2]
        
    xsol,ysol = fsolve(equations,guess)
    
    if lims != None:
        (xmin,xmax),(ymin,ymax) = lims
        if not (xmin <= xsol <= xmax and ymin <= ysol <= ymax):
            return None,None
            #print('WARNING! Solution',xsol,ysol,'outside of plot boundary')
        
    return xsol,ysol


def straight_line(graph, node0, node1):
    x0, y0 = graph.nodes[node0]['pos']
    x1, y1 = graph.nodes[node1]['pos']
    return [x0, x1], [y0, y1]


def get_bounding_polygons3(phase_graph,phase_labels=None):
    # Get bounding polygons to fill each phase region
    to_labels = lambda inds: get_labels(inds,phase_labels)
    node_phases = {i for node in phase_graph.nodes for i in node
                   if i not in ['xmin','xmax','ymin','ymax']}
    bounding_polygons = {}
    
    for phase in node_phases:
        node_list = [n for n in phase_graph.nodes if phase in n]
        if len(node_list) <= 2:
            # Must be optimal on at least three nodes
            continue
        else:
            subgraph = phase_graph.subgraph(node_list).copy().to_undirected()
            
            # Check that it is a single loop
            if not nx.is_connected(subgraph):
                raise Exception('Subgraph for node ' + str(to_labels(phase)) + ' is not connected. Probably non-contiguous,' + \
                                ' and it most likely is possible to draw by separating the loops in program.')
            if [n for n, d in subgraph.degree() if d != 2]:
                raise Exception('Subgraph for node' + str(to_labels(phase)) + ' is not a loop')
            
            # Go through the single loop    
            cycle = nx.cycle_basis(subgraph)[0]
            xs_all,ys_all = [],[]
            for i in range(len(cycle)):
                node0 = cycle[i]
                node1 = cycle[(i + 1) % len(cycle)]  # wrap around
                if phase_graph.has_edge(node0, node1):
                    if 'interpolation' in phase_graph[node0][node1]:
                        xs, ys = phase_graph[node0][node1]['interpolation']
                    else:
                        xs, ys = straight_line(phase_graph, node0, node1)
                elif phase_graph.has_edge(node1, node0):
                    if 'interpolation' in phase_graph[node1][node0]:
                        xs, ys = phase_graph[node1][node0]['interpolation']
                        xs = xs[::-1]
                        ys = ys[::-1]
                    else:
                        xs, ys = straight_line(phase_graph, node0, node1)
                else:
                    raise Exception('Edge does not exist ' + str(to_labels(node0)) + ', ' + str(to_labels(node1)))
                xs_all.extend(xs)
                ys_all.extend(ys)
            coord_pair_list = list(zip(xs_all, ys_all))    

            bounding_polygons[phase] = Polygon(coord_pair_list)
    return bounding_polygons


def get_min_phases(funcs,x,y,inds=None,rel_tol=1e-6,abs_tol=1e-9):
    # Get optimal phase indices for coords x,y
    if inds is None:
        inds = list(range(len(funcs)))
    elif isinstance(inds,(tuple,set,np.ndarray)):
        inds = list(inds)
    elif not isinstance(inds,list):
        raise Exception('Type ',type(inds),' could not be handled. Index error')

    values = [func(x,y) for i,func in enumerate(funcs) if i in inds]
    min_val = min(values)
    min_inds = [i for i, val in enumerate(values) if np.isclose(val,min_val,rel_tol,abs_tol)]
    min_phases = set(inds[min_ind] for min_ind in min_inds)
    
    return min_phases


def get_graph(funcs,phases,xgrid,ygrid,labels=None,graph_step=0.02,init_graph_step=None):
    # See plot_boundary_lines for inputs
    
    present_phases = np.unique(phases)
    num_phases = len(present_phases)
    
    xmin,xmax,ymin,ymax = xgrid[0,0],xgrid[-1,-1],ygrid[0,0],ygrid[-1,-1]
    lims = ((xmin,xmax),(ymin,ymax))
    len_x,len_y = xmax-xmin, ymax-ymin

    to_labels = lambda inds: get_labels(inds,labels)

    if init_graph_step == None:
        init_graph_step = 0.5 * graph_step

    delta_x = xgrid[0,1] - xgrid[0,0]
    delta_y = ygrid[1,0] - ygrid[0,0]
    
    
    def add_node(unformatted_node,coords):
        #----------Check if on a figure edge----------
        x,y = coords
        edge_labels = []
        for xlim,label in zip((xmin,xmax),('xmin','xmax')):
            if np.isclose(x,xlim):
                edge_labels.append(label)
        for ylim,label in zip((ymin,ymax),('ymin','ymax')):
            if np.isclose(y,ylim):
                edge_labels.append(label)
        if len(edge_labels) > 2:
            raise Exception('Figure edge error. Node ' + str(node) + ' close to ' + str(edge_labels))

        #---------------Format node-------------------
        if isinstance(unformatted_node,set):
            listed_node = sorted(unformatted_node)
        elif isinstance(unformatted_node,np.integer):
            listed_node = [unformatted_node]
        else:
            listed_node = list(unformatted_node)
        node = tuple(listed_node + edge_labels)
        
        #-----Add node if it does not exist with different coords-----
        if node in node_coords:
            if np.allclose(coords,node_coords[node]):
                if len(node) == 3:
                    print('Note! Attempt to write node',to_labels(node),'twice, but same coords')
            else:
                raise Exception('Two nodes (different coords) with label ' + str(to_labels(node)) + \
                                '. Possible reason: Non-contiguous phase regions.')
        else:
            node_coords[node] = coords
    
    def resolve_fig_edge(funcs,coords1,coords2,constrain_coefs,present_phases,nb_points=10):
        # Add finer grid between the points with coords1 and coords2
        # constrain_coefs should define a line between them
        x1,y1 = coords1
        x2,y2 = coords2
        xs = np.linspace(x1,x2,num=nb_points)
        ys = np.linspace(y1,y2,num=nb_points)
        line_phases = []
        for x,y in zip(xs,ys):
            values = [func(x,y) for i,func in enumerate(funcs)]
            phase = np.argmin(values)
            if phase not in present_phases:
                present_phases.append(phase)
            line_phases.append(phase)
        present_phases.sort()
        curr_phase = line_phases[0]
        for x,y,phase in zip(xs,ys,line_phases):
            if phase != curr_phase:
                xsol,ysol = solve(funcs[phase],funcs[curr_phase],constrain_coefs,guess=(x,y),lims=lims)
                phase_inds = get_min_phases(funcs,xsol,ysol,inds=present_phases)
                if not set((curr_phase,phase)) <= phase_inds:
                    # Not resolved enough
                    return False
                else:
                    add_node(phase_inds,(xsol,ysol))
                    detected_gridpoints[tuple(sorted(phase_inds))] = ((x1,x2),(y1,y2))
            curr_phase = phase
        print('Interpolation successful')
        return True
    
    def test_gridpoint_pair(vu1,vu2):
        # Find phase change from between two gridpoints vu1,vu2,
        # where phase 1 is optimal on vu1 and phase 2 on vu2
        # If same phase, do nothing
        
        # Constrain to line between the gridpoints
        x1,y1 = xgrid[vu1],ygrid[vu1]
        x2,y2 = xgrid[vu2],ygrid[vu2]
        
        A, B = y1-y2,x2-x1
        C = -(A*x1 + B*y1)
        constrain_coefs = (A,B,C)
        
        phase_ind1,phase_ind2 = phases[vu1],phases[vu2]
        if phase_ind1 != phase_ind2:
            midpoint = (0.5*(x1+x2), 0.5*(y1+y2))
            x,y = solve(funcs[phase_ind1],funcs[phase_ind2],constrain_coefs,guess=midpoint,lims=lims)
            phase_inds = get_min_phases(funcs,x,y,inds=present_phases) # Check for more optimal phases
            if not set((phase_ind1,phase_ind2)) <= phase_inds:
                print('Figure edge not sampled enough. Expected phase border:',
                      to_labels((phase_ind1,phase_ind2)))
                print('Optimal phases at that point:', to_labels(phase_inds),'Trying interpolation')
                if not resolve_fig_edge(funcs,(x1,y1),(x2,y2),constrain_coefs,present_phases):
                    raise Exception('Figure boundary error. Increase num_gridpoints.')
            else:
                add_node(phase_inds,(x,y))
                detected_gridpoints[tuple(sorted(phase_inds))] = ((x1,x2),(y1,y2))
            
        return

    #------------------Find nodes along figure boundary-------------------
    
    node_coords = {} # (phase1, phase2): (x,y), e.g. (2,1): (300,2.5)
    vmax,umax = phases.shape[0] - 1, phases.shape[1] - 1
    
    # Only needed for tutorial
    detected_gridpoints = {} # (phase1, phase2): ((x1,x2,...),(y1,y2,...))
    
    # Corners
    for x,y in [(x,y) for x in (xmin,xmax) for y in (ymin,ymax)]:
        add_node(get_min_phases(funcs,x,y,inds=present_phases),(x,y))
    
    # ymin
    for u in range(umax):
        test_gridpoint_pair((0,u),(0,u+1))
    
    # xmax
    for v in range(vmax):
        test_gridpoint_pair((v,-1),(v+1,-1))
    
    # ymax
    for u in range(umax-1,-1,-1):
        test_gridpoint_pair((-1,u+1),(-1,u))
    
    # xmin
    for v in range(vmax-1,-1,-1):
        test_gridpoint_pair((v+1,0),(v,0))
    
    edge_node_coords = node_coords.copy()

    #------------------------Find boundary nodes in the interior------------------------
    
    for v in range(vmax):
        for u in range(umax):
            gridpoints = [(v,u),(v+1,u),(v,u+1),(v+1,u+1)]
            quadset = set([phases[gp] for gp in gridpoints])

            # Examine sets of sample gridpoints in squares.
            # If 3 or more are different -> trinode or higher
            
            if len(quadset) == 3:
                # Trinode
                x,y = solve3(*[funcs[phase_ind] for phase_ind in quadset],(xgrid[v,u],ygrid[v,u]),lims=lims)
                if x is None or y is None:
                    # Outside fig boundary
                    continue
                phase_inds = get_min_phases(funcs,x,y,present_phases)
                if not quadset <= phase_inds:
                    # Not valid trinode for studied phases
                    continue
                add_node(phase_inds,(x,y))
                detected_gridpoints[tuple(phase_inds)] = (tuple(xgrid[gp] for gp in gridpoints),
                                                          tuple(ygrid[gp] for gp in gridpoints))
            elif len(quadset) == 4:
                # Several trinodes or higher points
                subsets = list(combinations(quadset, 3))
                for subset in subsets:
                    x,y = solve3(*[funcs[phase_ind] for phase_ind in subset],(xgrid[v,u],ygrid[v,u]),lims=lims)
                    if x is None or y is None:
                        # Outside fig boundary
                        continue
                    phase_inds = get_min_phases(funcs,x,y,inds=present_phases) #Check for optimal phases
                    if not set(subsets) <= phase_inds:
                        # Not valid trinode for studied phases
                        continue
                    add_node(phase_inds,(x,y))
                    detected_gridpoints[tuple(phase_inds)] = (tuple(xgrid[gp] for gp in gridpoints),
                                                              tuple(ygrid[gp] for gp in gridpoints))

    #----------------------------------Find all edges-------------------------------------

    plotlines = {}
    phase_node_coords = {key: val for key,val in node_coords.items()
                         if sum(isinstance(i, (int,np.int64)) for i in key) > 1}

    unfinished_nodes = list(phase_node_coords.keys())
    all_nodes = list(node_coords.keys())
    boundary_count = {key: 0 for key in phase_node_coords}

    graph = nx.DiGraph()
    graph.add_nodes_from(list(node_coords.keys()))

    limlabels = ['xmin','xmax','ymin','ymax']

    for pair in combinations(list(present_phases) + limlabels,2):
        # Loop through unique pairs of phases
        node_group = tuple(node for node in all_nodes if set(pair) <= set(node)) # pair in node
        if len(node_group) == 0:
            continue
        elif len(node_group) == 1:
            if not set(pair) < set(limlabels) and len(node_group) == 3:
                print('Node pair warning, pair',to_labels(pair),'only in node',to_labels(node_group[0]))
            continue
        elif len(node_group) != 2:
            #print(labels,type(labels))
            raise Exception('Cannot make edges with more than two nodes. Phases: ' + str(to_labels(pair)) + \
                            '\n' + ' ' * 11 + 'Nodes: ' + str(to_labels(node_group)) + '\n' + \
                            ' ' * 11 + 'Possible reason: Non-contiguous phases. Check with plot_colorgrid')
        node0,node1 = node_group
        
        if any(lim in pair for lim in limlabels):
            graph.add_edge(node0,node1)
        else:
            x0,y0 = node_coords[node0]
            xf,yf = node_coords[node1]
            
            dx, dy = xf - x0, yf - y0
            
            f0,f1 = [funcs[phase] for phase in pair]
            
            # Note that u,v will be graph coordinates such that
            # u: 0 -> 1, v: 0 -> 1 as x: xmin -> xmax, y: ymin -> ymax 
            
            def get_uv_norm(coords1,coords2,lims=lims):
                # Get distance in u,v graph coordinates
                (xmin,xmax),(ymin,ymax) = lims
                x1,y1 = coords1
                x2,y2 = coords2
                delta_x,delta_y = x2-x1, y2-y1
                delta_u = delta_x / (xmax - xmin)
                delta_v = delta_y / (ymax - ymin)
                uv_norm = np.linalg.norm((delta_u,delta_v))

                return uv_norm
            
            def get_coords_given_graph_dist(coords1,coords2,graph_dist,lims=lims):
                # Given two points in x,y coords, get new coords graph_dist (u,v) away
                # from point1 in the direction of point2
                uv_norm = get_uv_norm(coords1,coords2,lims=lims)
                
                x1,y1 = coords1
                x2,y2 = coords2
                delta_x,delta_y = x2-x1, y2-y1
                
                ratio = graph_dist / uv_norm 
                coords3 = (x1 + ratio * delta_x, y1 + ratio * delta_y)
                return coords3
            
            if get_uv_norm((x0,y0),(xf,yf)) <= init_graph_step:
                xs,ys = [x0,xf],[y0,yf]
                print('Warning! No interpolation on boundary',to_labels(pair),'Too short boundary')
                print('Consider decreasing graph_step')
            else:
                guessx,guessy = get_coords_given_graph_dist((x0,y0),(xf,yf),init_graph_step)
                # Get line perpendicular (in graph) to x0,y0;xf,yf
                # The perpendicular line to du=u1-u0, dv=v1-v0 has A=du, B=dv
                c0 = -(guessx * dx / len_x ** 2 + guessy * dy / len_y ** 2)
                constr = (dx / len_x ** 2,dy / len_y ** 2,c0)
                
                x,y = solve(f0,f1,constr,(guessx,guessy),lims=lims)
                
                # Check for optimal phases. If the phase pair are not optimal, the solution is false
                phase_inds = get_min_phases(funcs,x,y,inds=present_phases)

                if not set(pair) <= phase_inds:         
                    dx_wrong, dy_wrong = x - x0, y - y0
                    
                    # If the phase boundary is smooth, the wrong vector mirrored in x0,y0
                    # should be a good guess
                    mirrored = (x0 - dx_wrong,y0 - dy_wrong)
                    guessx,guessy = get_coords_given_graph_dist((x0,y0),mirrored,init_graph_step)
                    c0 = guessx * dx_wrong + guessy * dy_wrong
                    constr = (-dx_wrong,-dy_wrong,c0)
                    
                    x,y = solve(f0,f1,constr,(guessx,guessy),lims=lims)
                    phase_inds = get_min_phases(funcs,x,y,inds=present_phases)
                    if not set(pair) <= phase_inds:
                        print("Warning! Error in phase boundary",to_labels(pair),"close to",x0,y0)

                # Get the rest of the boundary by taking steps constant in u,v graph coordinates
                xs, ys = [x0,x],[y0,y]
                counter = 0
                max_steps = int(10 / graph_step)
                finished = False
                while counter <= max_steps and not finished:
                    x_m1,y_m1 = xs[-1],ys[-1]
                    
                    if get_uv_norm((x_m1,y_m1),(xf,yf)) <= graph_step:
                        # Close to end point (node)
                        xs.append(xf)
                        ys.append(yf)
                        finished = True
                    else:
                        x_m2,y_m2 = xs[-2],ys[-2]
                        dx_prev = x_m1 - x_m2
                        dy_prev = y_m1 - y_m2
                        prev_graph_step = get_uv_norm((x_m2,y_m2),(x_m1,y_m1))

                        # Get guess by extrapolating line from the two previous points
                        guessx,guessy = get_coords_given_graph_dist((x_m2,y_m2),(x_m1,y_m1),prev_graph_step + graph_step)

                        # Perpendicular line i graph coordinates
                        c0 = -(guessx * dx_prev / len_x ** 2 + guessy * dy_prev / len_y ** 2)
                        constr = (dx_prev / len_x ** 2,dy_prev / len_y ** 2,c0)

                        x,y = solve(f0,f1,constr,(guessx,guessy),lims=lims)

                        phase_inds = get_min_phases(funcs,x,y,inds=present_phases)
                        
                        # Not quite sure why I wrote the below
                        # Maybe error search?
                        if not set(pair) <= phase_inds:
                            print('Error')
                            xs.append(xf)
                            ys.append(yf)
                            finished = True
                        else:
                            xs.append(x)
                            ys.append(y)
                        
                    counter += 1
                    if counter == max_steps:
                        raise Exception('Did not find endpoint for phases ' + to_labels(pair))
            graph.add_edge(node0,node1,interpolation=(xs,ys))
            plotlines[node_group] = (xs,ys)
            
    # Count that nodes multiplicity match number of edges
    for node in graph.nodes:
        exp_edges = len(node)
        if sum(lim in node for lim in limlabels) == 2:
            exp_edges -= 1
        nb_edges = graph.out_degree(node) + graph.in_degree(node)
        if nb_edges != exp_edges:
            print('Warning! Expected',exp_edges,'for node',to_labels(node),', got',nb_edges)
    
    for node in graph.nodes:
        graph.nodes[node]["pos"] = node_coords[node]
    
    # Get highlights of gridpoints where a node is detected. To show how it works
    grid_highlights = {}
    for phase_tuple in detected_gridpoints:
        grid_highlights[phase_tuple] = get_enveloping_rectangle(*detected_gridpoints[phase_tuple],
                                                                0.5*delta_x,0.5*delta_y)
    return graph, grid_highlights


def plot_graph(graph):
    pos = nx.get_node_attributes(graph, 'pos')
    nx.draw(graph, pos, with_labels=True)
    plt.show()

if __name__ == "__main__":
    main()

