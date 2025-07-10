from src.funcs.Bspline_basis import *
from src.funcs.theta_estimation_conic_optimization import *
from scipy.optimize import differential_evolution
from collections import namedtuple

def hat_f(x, hat_theta, extended_knots_list, bdeg_list):
    """
    Evaluate the fitted B-spline smooth additive regression model at a given point x.

    :param x: (list or 1D array) values of the covariates at which to evaluate the fitted model
    :param hat_theta: (1D array) estimated theta coefficient vector of the model
    :param extended_knots_list: (list of lists) extended knots sequences for each X variable. One list per variable
    :param bdeg_list: (list of ints) degrees of the B-spline basis for each x variable

    :return: (float) fitted value of the B-spline smooth additive regression model at the input x
    """
    # input x vector as a data frame
    x_df = pd.DataFrame([x])
    # compute the B-spline basis matrix for x
    B, _ = Bspline_basis_and_penalty(x_df, extended_knots_list, bdeg_list)
    # compute the fitted value
    hat_f_x = float((B @ hat_theta)[0])
    return hat_f_x


def hat_f_global_min_max(hat_theta, X, extended_knots_list, bdeg_list):
    """
    Compute the global minimum and maximum of the fitted B-spline smooth additive regression model
    within the domain defined by the ranges of the observed X values.

    :param hat_theta: (1D array) estimated theta coefficient vector of the model
    :param X: (DataFrame) training observed data of the covariates X
    :param extended_knots_list: (list of lists) extended knots sequences for each X variable. One list per variable
    :param bdeg_list: (list of ints) degrees of the B-spline basis for each x variable

    :return: (tuple of floats) (min_val, max_val) global minimum and maximum of the fitted model on the domain defined by X.
    """
    ## find X domain bounds from the observed X values
    bounds = [(X[col].min(), X[col].max()) for col in X.columns]

    # minimize hat_f
    min_val = differential_evolution(
        lambda x: hat_f(x, hat_theta, extended_knots_list, bdeg_list),
        bounds=bounds
    ).fun
    # maximize hat_f (minimize negative)
    max_val = -differential_evolution(
        lambda x: -hat_f(x, hat_theta, extended_knots_list, bdeg_list),
        bounds=bounds
    ).fun
    return min_val, max_val

def compute_univariate_fitted_curves(hat_theta, B, extended_knots_list,bdeg_list):

    """
    Compute the fitted values of each univariate curve corresponding to one of the covariatesin a B-spline smooth
    additive regression model.

    :param hat_theta: (1D array) estimated theta coefficient vector of the model
    :param B: (2D array) full design matrix containing the evaluation of the B-spline basis functions at the observed training values in X and a first column of ones
    :param extended_knots_list: (list of lists) extended knots sequences for each X variable. One list per variable
    :param bdeg_list: (list of ints) degrees of the B-spline basis for each x variable

    :return (tuple):
        - hat_theta_split (list of 1D arrays): [intercept, theta_var1, theta_var2, ...] split version of hat_theta
        - fitted_curves_list (list of 1D arrays): fitted values of each of the univariate curves in the model
    """
    #####################################################
    ## splitting B and hat_theta into a list of arrays ##
    #####################################################

    ## computing cutting points to split: the index range for each variable
    # number of intervals in each variable (taking into account that I'm working with extended_knots_list
    n_intervals = np.array([len(inner_list) for inner_list in extended_knots_list]) - 1 - [2 * bdeg_var for bdeg_var in bdeg_list]
    # cutting points based on cumulative sum of number of intervals and degree
    cutting_points = np.concatenate(([1], 1 + np.cumsum(n_intervals) + np.cumsum(bdeg_list)))[:-1]

    ## splitting B discarding first column corresponding to the vector of 1s
    B = np.split(B, cutting_points, axis=1)[1:]
    ## splitting hat_theta into alpha (pos 0) and theta parameters for each variable
    hat_theta_split = np.split(hat_theta, cutting_points)

    ################################################
    ## computing fitted curves as theta_var*B_var ##
    ################################################
    fitted_curves_list = [B[var] @ hat_theta_split[var + 1] for var in range(len(B))]

    return hat_theta_split, fitted_curves_list


def shape_constrained_additive_regression_model_estimation(X, y, knots_list, bdeg_list,
                                                           df_interpolation=None,
                                                           df_underestimation=None,
                                                           df_overestimation=None,
                                                           lower_bound=None, upper_bound=None,
                                                           bounds_constraint_modelling=None):
    """
    Fitting a shape-constrained additive regression model using a B-spline approach from X and y as data inputs,
    the knots for each variable and the degree for each variable. The fitting allows incorporating the following
    shape constraints:
        - Exact interpolation at specified data points.
        - Underestimation or overestimation at specified data points (pointwise).
        - Global or local (pointwise) lower and upper bounds on the estimated function.

    :param X: (DataFrame) training observed data of the covariates X
    :param y: (1D array) training response y observed values corresponding to X
    :param knots_list: (list of lists) knots sequences for each X variable. One list per variable
    :param bdeg_list: (list of ints) degrees of the B-spline basis for each x variable

    :param df_interpolation: (DataFrame, optional) data points where the estimated function must interpolate exactly
                             Rows in df_interpolation must be disjoint from those in df_underestimation and df_overestimation
    :param df_underestimation: (DataFrame, optional) data points where the estimated function must underestimate y
                               Rows in df_underestimation must be disjoint from those in df_interpolation and df_overestimation
    :param df_overestimation: (DataFrame, optional) data points where the estimated function must overestimate y
                              Rows in df_overestimation must be disjoint from those in df_interpolation and df_underestimation

    :param lower_bound: (float, optional) imposed lower bound to the estimated function
    :param upper_bound: (float, optional) imposed upper bound to the estimated function
    :param bounds_constraint_modelling: (str, optional) method to enforce the lower and/or upper bounds. Must be either "pointwise" (local) or "bertsimas" (global)


    :return: namedtuple with:
        - hat_theta: (list of 1D arrays) estimated theta coefficient vectors split by the intercept and each of the covariates, i.e., [theta_0, theta_1,...,theta_p]
        - hat_y: (1D array) fitted response values predicted by the shape-constrained smooth additive regression model.
    """
    #########################
    #### checking inputs ####
    #########################
    ##  X and y must have the same length
    if len(X) != len(y):
        raise ValueError("X and y must have same length")
    ## n_intervals_list and bdeg_list must have same length as the columns of X
    if X.shape[1] != len(knots_list) or X.shape[1] != len(bdeg_list):
        raise ValueError("n_intervals_list and bdeg_list must have same length as the columns of X")
    ## The knots sequence of eacg variable must be increasingly ordered
    for index, knots in enumerate(knots_list):
        if any(np.diff(knots) <= 0):
            raise ValueError(f"Each knot sequence in 'knots_list' must be strictly increasing. Problem at index {index}.")
    ## All the degrees must be greater than 0
    if any(bdeg < 1 for bdeg in bdeg_list):
        raise ValueError("The B-spline degrees of each variable (bdeg_list) must an integer greater than 0.")


    ###########################################
    #### B-spline basis: B and PI matrices ####
    ###########################################

    ## extending knots in each variable according to its bdeg
    extended_knots_list = []
    for var in range(len(knots_list)):
        knots_var = knots_list[var]
        bdeg_var = bdeg_list[var]
        extended_knots_list_var = extend_knots_sequence(knots_var, bdeg_var)
        extended_knots_list.append(extended_knots_list_var)

    ## computing B and PI
    B, PI = Bspline_basis_and_penalty(X, extended_knots_list, bdeg_list)


    #########################################################
    #### estimating theta coefficient vector: hat_theta  ####
    #########################################################

    ## if lower and/or upper bounds are required with the conditions of Bertsimas and Popescu (2002) modelling approach:
    if lower_bound is not None or upper_bound is not None and bounds_constraint_modelling == "bertsimas":

        ## estimating theta with no lower bound and upper bound enforced on the estmated function
        hat_theta = theta_estimation_conic_optimization(B=B, PI=PI,
                                                        X=X, y=y, extended_knots_list=extended_knots_list,
                                                        bdeg_list=bdeg_list,
                                                        df_interpolation=df_interpolation,
                                                        df_underestimation=df_underestimation,
                                                        df_overestimation=df_overestimation,
                                                        lower_bound=None, upper_bound=None,
                                                        bounds_constraint_modelling=bounds_constraint_modelling)
        ## computing hat_y
        hat_y = B @ hat_theta

        ## computing global minimum and maximum of the estimated function across the whole domain
        y_min_global, y_max_global = hat_f_global_min_max(hat_theta, X, extended_knots_list, bdeg_list)

        ## if the estimated function (hat_f) overpasses the required bounds, we estimate theta again imposing those
        # bounds,with adequate omega_lower_bound and omega_upper_bound vectors
        if (lower_bound is not None and (lower_bound-y_min_global>0.001)) or (upper_bound is not None and (y_max_global-upper_bound>0.001)):
            ## computing the fitted values of the univariate curves in the smooth additive model
            hat_theta_split, fitted_curves_list = compute_univariate_fitted_curves(hat_theta, B, extended_knots_list, bdeg_list)

            ## computing omega_lower_bound and omega_upper_bound
            omega_lower_bound = [np.min(fitted_curve) for fitted_curve in fitted_curves_list]
            omega_lower_bound = omega_lower_bound / sum(omega_lower_bound)
            omega_upper_bound = [np.max(fitted_curve) for fitted_curve in fitted_curves_list]
            omega_upper_bound = omega_upper_bound / sum(omega_upper_bound)

            ## re-estimating theta imposing the lower and/or upper bound on the estimated function using omega_lower_bound and omega_upper_bound
            hat_theta = theta_estimation_conic_optimization(B=B, PI=PI,
                                                            X=X, y=y,
                                                            extended_knots_list=extended_knots_list,
                                                            bdeg_list=bdeg_list,
                                                            df_interpolation=df_interpolation,
                                                            df_underestimation=df_underestimation,
                                                            df_overestimation=df_overestimation,
                                                            lower_bound=lower_bound, upper_bound=upper_bound,
                                                            bounds_constraint_modelling=bounds_constraint_modelling,
                                                            omega_lower_bound=omega_lower_bound, omega_upper_bound=omega_upper_bound)
            ## computing hat_y
            hat_y = B @ hat_theta

    ## if lower and/or upper bounds are required with the pointwise approach or if there are not bounds constraint at all
    else:
        hat_theta = theta_estimation_conic_optimization(B=B, PI=PI,
                                                        X=X, y=y,
                                                        extended_knots_list=extended_knots_list, bdeg_list=bdeg_list,
                                                        df_interpolation=df_interpolation,
                                                        df_underestimation=df_underestimation,
                                                        df_overestimation=df_overestimation,
                                                        lower_bound=lower_bound, upper_bound=upper_bound,
                                                        bounds_constraint_modelling=bounds_constraint_modelling)

    ## computing hat_y
    hat_y = B @ hat_theta

    ## splitting theta by variable and computing fitted univariate curves
    hat_theta_split, fitted_curves_list = compute_univariate_fitted_curves(hat_theta, B, extended_knots_list, bdeg_list)

    ## raising a warning if the mean of the fitted curve of any var is different to 0
    mean_fitted_curves = [np.mean(inner_list) for inner_list in fitted_curves_list]
    mean_curves_exceeding_0 = np.where(np.abs(np.array(mean_fitted_curves)) > 0.05)[0]
    if len(mean_curves_exceeding_0) > 0:
        vars_error = X.columns[mean_curves_exceeding_0]
        vars_error = ", ".join(vars_error)
        error_message = (f"The mean of the fitted curve of the following variable(s) is not 0: {vars_error}")
        raise ValueError(error_message)

    #################################
    ## preparing results to return ##
    #################################
    BsplineAdditiveModel = namedtuple('Bspline_additive_model', ['hat_theta', 'hat_y'])
    return BsplineAdditiveModel(hat_theta_split, hat_y)


def model_prediction(hat_theta, extended_knots_list, bdeg_list,X_new):
    """
    Compute predicted values of the additive B-spline regression model at new X input data.

    :param hat_theta: (1D array) estimated theta coefficient vector of the model
    :param extended_knots_list: (list of lists) extended knots sequences for each X variable. One list per variable
    :param bdeg_list: (list of ints) degrees of the B-spline basis for each x variable
    :param X_new: (DataFrame) New covariate data for prediction, with the same number of columns and the same variable order as the original training data `X`.

    :return (1D array) Predicted response values evaluated at X_new
    """
    ## concatenate hat_theta to have all coefficients in a single list
    hat_theta_concatenate = np.concatenate(hat_theta)
    ## Compute B matrix but evaluated in X_new
    B_new,  = Bspline_basis_and_penalty(X_new, extended_knots_list, bdeg_list)
    ## prediction
    hat_y_new = B_new @ hat_theta_concatenate
    return hat_y_new