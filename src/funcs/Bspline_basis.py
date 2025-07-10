import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import block_diag

def extend_knots_sequence(knots,bdeg):
    """
    Extend a knot sequence by adding bdeg knots at each end using average spacing.

    :param knots: (1D array)input knots sequence
    :param bdeg: (int) degree of the B-splines basis

    :return (1D array): extended knots sequence
    """
    ## compute mean increment in knot sequence
    dx = (knots[-1] - knots[0])/(len(knots)-1)
    ## compute external knots on the left side and sort
    external_left_knots = np.sort(knots[0] - dx * np.arange(1, bdeg + 1))
    ## compute external knots on the right side and sort
    external_right_knots = np.sort(knots[-1] + dx * np.arange(1, bdeg + 1))
    ## concatenate all knots
    extended_knots = np.concatenate((external_left_knots,knots,external_right_knots))
    return extended_knots


def univariate_Bspline_basis(x, extended_knots, bdeg):
    """
    Given an extended knots sequence and a degree, compute the B-spline basis functions and
    return a matrix with their evaluation at input values x.

    :param x: (1D array) input x variable values
    :param extended_knots: (1D array) extended knots sequence
    :param bdeg: (int) degree of the B-spline basis

    :return (2D array): matrix with B-spline basis evaluations at each x value
    """
    ## number of B-splines in the basis
    n_Bsplines = len(extended_knots) - bdeg - 1

    ## prepare matrix B to store Bspline basis
    B = np.zeros(shape=(len(x), n_Bsplines))

    ## fill the matrix B by evaluating each basis function
    for l in range(n_Bsplines):
        # create coeffs  for the l-th Bspline
        coeffs = np.zeros(n_Bsplines, dtype=int)
        coeffs[l] = 1
        # Create a BSpline object and evaluate with coeffs to get the l-th basis function in the basis
        Bspline_l = BSpline(extended_knots, coeffs, bdeg)
        # Evaluate the l-th basis function at x values and store1
        B[:, l] = Bspline_l(x)
    return B


def Bspline_basis_and_penalty(X,extended_knots_list,bdeg_list):
    """
    compute the full B-spline basis matrix (B) and the identifiability quadratic penalty matrix (PI)
    including all variables in a smooth additive regression model.

    :param X: (DataFrame) dataframe with the x variable values as columns
    :param extended_knots_list: (list of 1D arrays) extended knots sequences for each x variable. One list per variable.
    :param bdeg_list: (list of ints) degrees of the B-spline basis for each x variable

    :return (tuple): 
        - B (2D array): full B-spline basis matrix for all variables in X
        - PI (2D array): block-diagonal identifiability penalty matrix for all variables in X
    """
    ## data characteristics
    n_data =X.shape[0]
    d_var = X.shape[1]

    ## initialize necessary variables
    # vector of ones for the identifiability
    ones = np.ones(shape=(n_data, 1))
    # matrix B to store all the basis, starting with the vector of ones
    B = ones
    # matrix PI to store the identifiability quadratic penalty matrices
    PI = 0

    ## loop over each variable and compute its B-spline basis matrix and its identifiability penalty matrix
    for var in range(d_var):
        ## get x values, knots and bdeg of this var
        x_var = X.iloc[:,var].to_numpy()
        extended_knots_var = extended_knots_list[var]
        bdeg_var = bdeg_list[var]

        ## compute the B-splines basis of this var
        B_var = univariate_Bspline_basis(x_var, extended_knots_var, bdeg_var)
        ## concatenate B_var basis to B matrix by columns
        B = np.concatenate((B, B_var), axis=1)

        ## compute the identifiability quadratic penalty matrix of this var
        PI_var = B_var.T @ ones @ ones.T @ B_var
        ## add PI_var to PI
        PI = block_diag(PI, PI_var)

    return  B, PI