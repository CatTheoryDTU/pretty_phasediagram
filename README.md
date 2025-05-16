PRETTY PHASE DIAGRAM
Written by Mikael Valter-Lithander

Draws a phase diagram with smooth phase borders

The program draws a phase diagram, defined by the function of minimum value of a list of functions of two variables x and y. The variables may be (temperature, pressure), or two scaling descriptors, or include a potential. An example function is Gibbs free energy of a number of surface terminations.

Drawing a crude phase diagram is not hard; ```plot_colorgrid``` accomplishes this by evaluating all functions on a grid and plot it point by point.

The pretty part is that the program finds the phase borders inside the diagram and plots them smoothly. This is done with ```plot_diagram```. This is done by finding nodes between at least three phases or at least two phases and the figure border.

USAGE

Running ```pretty_phasediagram.py``` will generate an example diagram. ```pretty_phasediagram.py -h``` will show more options if you want to play around with the example.

To use the program for custom data, import ```plot_colorgrid``` (and optionally ```plot_colorgrid``` as a backup) to your script and run it. Note that ```xlims = (xmin, xmax)```, ```ylims = (ymin, ymax)```. The functions in the list ```funcs``` have to be proper functions, either defined by ```def``` or ```lambda``` and only be dependent on x and y. If there is a third variable, say, a label that is used to look up a potential energy, it can be wrapped as ```func = lambda x, y: original_func(x, y, label=label)

Also note that the diagram is defined by the minimum of the functions. If you are interested in the maximum, use ```[-func for func in funcs]```.

