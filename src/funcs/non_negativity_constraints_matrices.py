import numpy as np
from math import comb
from scipy.interpolate import BSpline, PPoly

def compute_matrices_H(bdeg):
    """
    Compute the set of H matrices to represent the right-hand side of the equations for
    the univariate polynomial non-negativity conditions in Proposition 1(d) of Bertsimas and Popescu (2002).

    :param bdeg: (int) degree of the B-spline basis

    :return: (list of 2D arrays) list of 2 * bdeg + 1 H matrices. the first bdeg matrices correspond
    to the homogeneous equations (= 0), while the remaining ones are used in the non-homogeneous
    equations (= value ≠ 0)
    """
    # The ones in matrices H corresponding to the homogeneous equations are located on its
    # even antidiagonals, while the ones in matrices H corresponding to the non-homogeneous equations
    # are situated on the odd antidiagonals
    diag_homogeneous = np.linspace( 1 - bdeg, bdeg - 1, bdeg, dtype=np.int8)
    diag_non_homogeneous = np.linspace(-bdeg, bdeg, bdeg + 1, dtype=np.int8)
    H_list = []
    for k in np.concatenate([diag_homogeneous,diag_non_homogeneous]):
        # Create a matrix with 1s along the corresponding diagonal (and 0 in the rest) and rotate it 90 degrees along axis 0 to get the antidiagonal.
        H = np.eye(bdeg + 1, k=k, dtype=np.int32)
        # H = np.rot90(H,axes=(1,0))
        H = np.rot90(H,axes=(0,1))
        H_list.append(H)
    return H_list


def compute_matrices_W(extended_knots,bdeg):
    """
    Compute the set of W matrices to represent the right-hand side of the non-homogeneous equations
    for the univariate polynomial non-negativity conditions in Proposition 1(d) of Bertsimas and Popescu (2002).
    For each internal interval of the B-spline basis of a variable, one matrix W is defined. Each W matrix
    contains the monomial weights evaluated at the corresponding interval knots, as stated by
    the non-negativity conditions.

    :param extended_knots: (1D array) extended knots sequence
    :param bdeg: (int) degree of the B-spline basis

    :return: (list of 2D arrays) list of W matrices, one for each internal interval of the B-spline basis
    """
    ## compute original knots before extending
    knots = extended_knots[bdeg:-bdeg]
    ## list to store all W matrixes
    W_list = []
    ## one W matrix per internal interval: knot 0 to knot k-1
    for knot in range(len(knots)-1):
        ## initialize and fill the W in the interval
        W_knot = np.zeros(shape=(bdeg + 1, bdeg + 1))
        for q in range(bdeg + 1):
            for m in range(q+1):
                for r in range(m, bdeg + m - q+1):
                    W_knot[q, r] += (
                        comb(r, m)
                        * comb(bdeg - r, q - m)
                        * (knots[knot]) ** (r - m)
                        * (knots[knot + 1]) ** m
                    )
        ## add W in the interval to the list of matrices
        W_list.append(W_knot)
    return W_list


def polynomial_shift_to_x(coeffs, shift):
    """
    Convert the coefficients in a polynomial from the (x - shift) basis to the standard x basis,
    using the binomial theorem.

    :param coeffs: (1D array) coefficients of the polynomial in the (x - shift) basis
    :param shift: (float) shift value used in the original basis

    :return: (1D array) coefficients of the polynomial in the standard x basis
    """
    ## polynomial degree
    d = len(coeffs) - 1
    ## initialize the vector to store the shifted coefficients
    shifted_coeffs = np.zeros(len(coeffs))

    ## apply the binomial theorem expansion to each original coefficient
    for i in range(d + 1):
        for j in range(i + 1):
            shifted_coeffs[j] += coeffs[i] * (-shift) ** (i - j) * comb(i, j)
    return shifted_coeffs


def compute_matrices_G(extended_knots, bdeg):
    """
    Compute the set of G matrices to represent the right-hand side of the non-homogeneous equations
    for the univariate polynomial non-negativity conditions in Proposition 1(d) of Bertsimas and Popescu (2002).
    For each internal interval of the B-spline basis of a variable, one matrix G is defined. Each G matrix
    contains the coefficients of the active B-spline functions in the interval expressed in the standard x basis.

    :param extended_knots: (1D array) extended knots sequence
    :param bdeg: (int) degree of the B-spline basis

    :return: (list of 2D arrays) list of G matrices, one for each internal interval of the B-spline basis
    """

    ## number of intervals
    k = len(extended_knots) - 2*bdeg  -1
    ## number of Bspline functions in the basis
    n_Bsplines = len(extended_knots) - bdeg - 1

    ## initialize list to store matrices G
    G_list = []

    ## compute each G_q matrix
    for q in range(bdeg +1  , k + bdeg + 1 ):
        ## first block of zeros
        G_q_left_block = np.zeros((bdeg +1, q-(bdeg +1)))
        ## second block of zeros
        G_q_right_block = np.zeros((bdeg +1, k + bdeg - q))

        ## middle block: extracting the coefficients of the non-zero B-spline basis functions in the interval
        G_q_middle_block=np.empty((bdeg +1, 0))
        for d in range(bdeg + 1):
            ## create Bspline object
            coeffs = np.zeros(n_Bsplines, dtype=int)
            coeffs[q - bdeg + d -1] =1
            Bspline_object = BSpline(extended_knots, coeffs, bdeg)
            Bspline_object_ppoly = PPoly.from_spline(Bspline_object,extrapolate=False)

            ## identify the segment corresponding to the interval [t_q, t_(q+1)]
            index = np.searchsorted(Bspline_object_ppoly.x, extended_knots[q-1], side='right') - 1
            ## extract spline coefficients reversing the order
            Bspline_function_coeffs = Bspline_object_ppoly.c[:, index][::-1].T

            ## shift polynomial coefficients from the (x - knot) basis to the standard x basis
            knot_q = extended_knots[q - 1]
            monomial_coeffs = polynomial_shift_to_x(Bspline_function_coeffs, shift = knot_q)

            ## Stack transformed coefficients into G_q_middle_block matrix
            G_q_middle_block = np.column_stack((G_q_middle_block, monomial_coeffs))

        ## matrix G is formed by the left zeros block, the middle block and the right zeros block
        G_q = np.hstack((G_q_left_block,G_q_middle_block, G_q_right_block))
        ## add G in interval q to the list of matrices
        G_list.append(G_q)
    return G_list