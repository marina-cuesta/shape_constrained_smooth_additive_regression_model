import numpy as np

def generate_equidistant_knots(x,n_intervals):
    """
    Generate a sequence of equally spaced knots within the range of values of x.

    :param x: (1D array) input x values
    :param n_intervals: (int) number of internal intervals

    :return 1D array: generated knot sequence
    """
    ## left and right bounds for the intervals
    xl = np.min(x)
    xr = np.max(x)
    # compute internal knots
    knots = np.linspace(xl, xr, n_intervals+1)
    return knots

