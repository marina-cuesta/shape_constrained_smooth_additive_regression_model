from scipy.interpolate import BSpline, PPoly
from src.funcs.Bspline_basis import *
import numpy as np

def polynomial_coefficients_additive_regression_model(hat_theta, knots_list, bdeg_list):
    """
    Given the estimated theta coefficients in a B-spline smooth additive regression model,
    compute the piecewise polynomial coefficients (in power basis form) of all univariate curves
    in the model over each internal interval.

    :param hat_theta: (1D array) estimated theta coefficient vector of the model
    :param knots_list: (list of lists) knots sequences for each X variable. One list per variable
    :param bdeg_list: (list of ints) degrees of the B-spline basis for each x variable

    :return: (list of 2D arrays) Polynomial coefficient matrices for each univariate curve.
             Each matrix has shape (number of internal intervals of the var , bdeg of the var + 1), with rows
             for intervals and columns for polynomial coefficients ordered from highest to lowest degree.
    """
    ## extending knots in each variable according to its bdeg
    extended_knots_list = []
    for var in range(len(knots_list)):
        knots_var = knots_list[var]
        bdeg_var = bdeg_list[var]
        extended_knots_list_var = extend_knots_sequence(knots_var, bdeg_var)
        extended_knots_list.append(extended_knots_list_var)

    ## delete first position of theta corresponding to alpha
    hat_theta_without_alpha = hat_theta[1:]

    ## initialize list to store the coefficients
    spline_coeffs_list = []

    ## extract the coefficients for all variables
    for var in range(len(hat_theta_without_alpha)):
        ## hat_theta, extended knots and bdeg of this variable
        hat_theta_var = hat_theta_without_alpha[var]
        extended_knots_var = extended_knots_list[var]
        bdeg_bar = bdeg_list[var]

        ## B-spline object corresponding to variable var
        B_spline_var = BSpline(extended_knots_var, hat_theta_var, bdeg_bar)

        ## polynomial coefficients of B_spline_var
        poly_var = PPoly.from_spline(B_spline_var, extrapolate=False)
        spline_coeffs_var = poly_var.c.T

        ## internal knots of extended_knots except for the last one
        internal_knots_ix = np.arange(bdeg_bar, len(extended_knots_var) - bdeg_bar - 1)

        ## selecting rows corresponding to the internal knots but the last one in spline_coeffs_var
        spline_coeffs_var = spline_coeffs_var[internal_knots_ix, :]

        ## add the coefficients of this var to the list
        spline_coeffs_list.append(spline_coeffs_var)
    return spline_coeffs_list


def save_fitted_models_polycoeffs_txt(filename, X_vars_dict, bdegs_dict, knots_dict, alphas_dict, poly_coeffs_dict):
    """
    Save the results of the fitted B-spline (shape-constrained) additive regression model in tge polynomial
    coefficient form in a structured text file. This includes response variables, their associated covariates
    and degrees, knots, alpha intercepts, and the polynomial coefficients of the fitted curves of each response variable.

    :param filename: (str) path and name of the text file to save the results
    :param X_vars_dict: (dict of {str: list of str}) dictionary mapping each response variable to its corresponding list of covariates
    :param bdegs_dict: (dict of {str: list of int}) dictionary mapping each response variable to a list of B-spline degrees per covariate
    :param knots_dict: (dict of {str: list of 1D arrays}) dictionary mapping each response variable to a list of knot sequences,
                        where each knot sequence is a 1D numpy array
    :param alphas_dict: (dict of {str: float}) dictionary mapping each response variable to its estimated intercept value
    :param poly_coeffs_dict: (dict of {str: list of 2D arrays}) dictionary mapping each response variable to a list of polynomial coefficient matrices, one 2D array per covariate

    :return: None
    """
    ## splines
    splines = list(X_vars_dict.keys())

    ## computing number of splines
    n_splines = len(X_vars_dict)

    ## setting some formats of the txt file
    decimals = 10

    ## txt file to store results and completing it
    with open(filename, "w") as file:

        ## number of splines
        file.write(f"param\t splines_n := {n_splines};\n\n")

        ## number of variables involved in each spline
        file.write(f"param\t splines_nVariables := \n")
        for spline in range(n_splines):
            y_var_spline = splines[spline]
            n_variables = len(X_vars_dict[y_var_spline])
            file.write(f"{spline+1}\t{n_variables}\n")
        file.write(";\n\n")

        ## y var in the inequality of the opt problem where the spline is involved
        file.write(f"param\t splines_variablesY := \n")
        for spline in range(n_splines):
            file.write(f"{spline+1}\t{splines[spline]}\n")
        file.write(";\n\n")

        ## sigma functions involved in the approximation of each spline
        file.write(f"param\t splines_variablesSigma := \n")
        for spline in range(n_splines):
            y_var_spline = splines[spline]
            n_variables = len(X_vars_dict[y_var_spline])
            for var in range(n_variables):
                var_vame = X_vars_dict[y_var_spline][var]
                file.write(f"{spline+1}\t{var+1}\t'sigma_{spline+1}_{var_vame}'\n")
        file.write(";\n\n")

        ## variables X involved in the approximation of each spline
        file.write(f"param\t splines_variablesX := \n")
        for spline in range(n_splines):
            y_var_spline = splines[spline]
            n_variables = len(X_vars_dict[y_var_spline])
            for var in range(n_variables):
                var_vame = X_vars_dict[y_var_spline][var]
                file.write(f"{spline+1}\t{var+1}\t'{var_vame}'\n")
        file.write(";\n\n")

        ## number of intervals in each X var involved in the approximation of each spline
        file.write(f"param\t splines_nIntervals := \n")
        for spline in range(n_splines):
            y_var_spline = splines[spline]
            ## getting the knots_list of this spline
            knots_list_spline = knots_dict[y_var_spline]
            ## computing nIntervals in each variable from knots_list of this spline
            nIntervals = [len(inner_list) - 1 for inner_list in knots_list_spline]
            for var in range(len(nIntervals)):
                file.write(f"{spline+1}\t{var+1}\t{nIntervals[var]}\n")
        file.write(";\n\n")

        ## degree of each X var involved in the approximation of each spline
        file.write(f"param\t splines_nDegree := \n")
        for spline in range(n_splines):
            y_var_spline = splines[spline]
            ## getting the bdeg_list of this spline
            bdeg_list_spline = bdegs_dict[y_var_spline]
            for var in range(len(bdeg_list_spline)):
                file.write(f"{spline+1}\t{var+1}\t{bdeg_list_spline[var]}\n")
        file.write(";\n\n")

        ## alpha parameter of each spline
        file.write(f"param\t splines_alpha := \n")
        for spline in range(n_splines):
            y_var_spline = splines[spline]
            ## getting the alpha parameter of this spline
            alpha_spline = alphas_dict[y_var_spline]
            file.write(f"{spline+1}\t{alpha_spline:.{decimals}f}\t\n")
        file.write(";\n\n")

        ## knots of each variable involved in each spline
        file.write(f"param\t splines_l\n")
        for spline in range(n_splines):
            y_var_spline = splines[spline]
            ## getting the knots_list of this spline
            knots_list_spline = knots_dict[y_var_spline]
            for var in range(len(knots_list_spline)):
                # getting the knots of this var
                knots_var = knots_list_spline[var]
                ## write the index of this spline + var in spline
                 # column index
                file.write(f"[{spline+1},{var+1},*]\t:=\n")
                for row in range(len(knots_var)):
                    knots_var_row = knots_var[row]
                    file.write(f"{row + 1}\t{knots_var_row:.{decimals}f}\n")
                if (spline < n_splines-1) or (var < (len(knots_list_spline) - 1)):
                    file.write("\n")
        file.write(";\n\n")

        ## spline  of each variable involved in each spline
        file.write(f"param\t splines\n")
        for spline in range(n_splines):
            y_var_spline = splines[spline]
            ## getting the spline_coeffs_list of this spline
            spline_coeffs_list_spline = poly_coeffs_dict[y_var_spline]
            for var in range(len(spline_coeffs_list_spline)):
                # sep between cols
                sep = 4
                # getting the splines coeffs of this var
                spline_coeffs_var = spline_coeffs_list_spline[var]
                # max number of digits in each column
                width_cols_coeffs = [max(len(f'{coeff:.{decimals}f}') for coeff in col) for col in spline_coeffs_var.T]
                ## write the index of this spline + var in spline
                 # column index
                column_ix = list(range(1, spline_coeffs_var.shape[1] + 1))
                file.write(f"[{spline + 1},{var + 1},*,*]\t:\t{'\t'.join(map(str, column_ix))}\t:=\n")
                for row in range(spline_coeffs_var.shape[0]):
                    file.write(f"{row + 1}\t")
                    for col in range(spline_coeffs_var.shape[1]):
                        space = ' ' * sep * ((col + 1) != spline_coeffs_var.shape[1])
                        file.write(f"{spline_coeffs_var[row, col]:>{width_cols_coeffs[col]}.{decimals}f}{space}")
                    file.write("\n")
                if (spline < n_splines-1) or (var < (len(knots_list_spline) - 1)):
                    file.write("\n")

        file.write(";")
        return