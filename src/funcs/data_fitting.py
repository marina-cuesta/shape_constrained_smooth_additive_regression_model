import os
from src.funcs.shape_constrained_additive_regression_model import *
from src.funcs.utils.knots import *
from src.funcs.output_file import *


def models_fitting_and_txt_result(folder,data_name, data,
                                   y_vars, X_vars_dict,
                                   n_intervals_dict, bdegs_dict,
                                   dfs_interpolation_dict=None,
                                   dfs_underestimation_dict=None,
                                   dfs_overestimation_dict=None,
                                   lower_bounds_dict=None,
                                   upper_bounds_dict=None,
                                   bounds_constraint_modelling_dict=None):

    """
    Fit a smooth additive regression models (with shape constraints if specified) to a list of response variables,
    indicated by y_vars. Each response variable is modelled with a set of X covariates specified in X_vars_dict.
    The resulting polynomial coefficients of all fitted models are saved to a .txt file.

    :param folder: (str) Subfolder name under 'results/txt files/' where the .txt output is saved
    :param data_name: (str) name of the dataset
    :param data: (DataFrame) input dataset containing both response and predictor variables
    :param y_vars: (list of str) names of the response variables to be modeled
    :param X_vars_dict: (dict of {str: list of str}) dictionary mapping each response variable to its corresponding list of covariates
    :param n_intervals_dict: (dict of {str: list of int}) dictionary mapping each response variable to a list of number of intervals per covariate
    :param bdegs_dict: (dict of {str: list of int}) dictionary mapping each response variable to a list of B-spline degrees per covariate
    :param dfs_interpolation_dict: (dict of {str: DataFrame}, optional) dictionary mapping each response variable to a DataFrame of interpolation constraints
    :param dfs_underestimation_dict: (dict of {str: DataFrame}, optional) dictionary mapping each response variable to a DataFrame of pointwise underestimation constraints
    :param dfs_overestimation_dict: (dict of {str: DataFrame}, optional) dictionary mapping each response variable to a DataFrame of pointwise overestimation constraints
    :param lower_bounds_dict: (dict of {str: float}, optional)  dictionary mapping each response variable to a lower bound
    :param upper_bounds_dict: (dict of {str: float}, optional)  dictionary mapping each response variable to an upper bound
    :param bounds_constraint_modelling_dict: (dict of {str: str}, optional) dictionary mapping each response variable to a string specifying
                                              how bounds constraints are modeled: "pointwise" (local) or "bertsimas" (global)

    :return: (dict of {str: 1D array}) dictionary mapping each response variable to its vector of fitted values
    """

    ####################################################
    #### adapting  parameters for shape constraints ####
    ####################################################

    if dfs_interpolation_dict is None:
        dfs_interpolation_dict={key: None for key in y_vars}
    if dfs_underestimation_dict is None:
        dfs_underestimation_dict = {key: None for key in y_vars}
    if dfs_overestimation_dict is None:
        dfs_overestimation_dict = {key: None for key in y_vars}
    if lower_bounds_dict is None:
        lower_bounds_dict={key: None for key in y_vars}
    if upper_bounds_dict is None:
        upper_bounds_dict={key: None for key in y_vars}
    if bounds_constraint_modelling_dict is None:
        bounds_constraint_modelling_dict={key: None for key in y_vars}

    ###########################
    #### preparing results ####
    ###########################
    # path to save txt files
    path_txt_save = os.path.join(os.getcwd(), 'results', 'txt files', folder, data_name)
    os.makedirs(path_txt_save, exist_ok=True)

    ## initialize dicts to store polynomial coefficients
    alphas_dict = {key: None for key in y_vars}
    poly_coeffs_dict = {key: None for key in y_vars}
    ## initialize dict to store knots
    knots_dict = {key: None for key in y_vars}

    ## initialize dict to store hat_y
    hat_y_dict = {key: None for key in y_vars}

    ########################################################
    #### fit an additive regression model to each y_var ####
    ########################################################
    for y_var in y_vars:

        ##################################
        #### preparing necessary data ####
        ##################################
        ## extract y variable
        y = data[y_var].to_numpy()
        ## extract X variables
        X_vars = X_vars_dict[y_var]
        X = data[X_vars]

        ## extract bdeg of X_vars
        bdeg_list = bdegs_dict[y_var]

        ## setting knots sequence for every variable in X_vars
        n_intervals_list = n_intervals_dict[y_var]
        knots_list = [generate_equidistant_knots(X.iloc[:, col], n_intervals=n_intervals_list[col]) for col
                      in range(X.shape[1])]
        ## adding knots of y_varto dict
        knots_dict[y_var] = knots_list

        ## extracting variables related to shape constraints corresponding to y_var
        df_interpolation = dfs_interpolation_dict.get(y_var, None)
        df_underestimation = dfs_underestimation_dict.get(y_var, None)
        df_overestimation = dfs_overestimation_dict.get(y_var, None)
        lower_bound = lower_bounds_dict.get(y_var, None)
        upper_bound = upper_bounds_dict.get(y_var, None)
        bounds_constraint_modelling = bounds_constraint_modelling_dict.get(y_var, None)


        ######################
        ### model fitting ####
        ######################

        fitted_model = shape_constrained_additive_regression_model_estimation(X=X, y=y,
                                                                              knots_list=knots_list,
                                                                              bdeg_list=bdeg_list,
                                                                              df_interpolation=df_interpolation,
                                                                              df_underestimation=df_underestimation,
                                                                              df_overestimation=df_overestimation,
                                                                              lower_bound=lower_bound,
                                                                              upper_bound=upper_bound,
                                                                              bounds_constraint_modelling=bounds_constraint_modelling)
        ## estimated coefficients
        hat_theta = fitted_model.hat_theta
        ## alpha is hat_theta[0]
        alpha = hat_theta[0][0]
        ## add alpha to dicts
        alphas_dict[y_var] = alpha

        ## predicted values
        hat_y = fitted_model.hat_y
        hat_y_dict[y_var] = hat_y

        ##############################################################################
        ##### polynomial coefficients of the fitted B-spline curve in the model  #####
        ##############################################################################

        ## computing the polynomial coefficients of the fitted additive model
        poly_coeffs = polynomial_coefficients_additive_regression_model(hat_theta, knots_list, bdeg_list)
        ## add spline_coeffs_list to dicts
        poly_coeffs_dict[y_var] = poly_coeffs

    ############################
    #### preparing txt file ####
    ############################
    ## setting the file name
    filename = os.path.join(path_txt_save, data_name + "-poly_coeffs_additive_regression_model.txt")
    ## creating and saving the txt file
    save_fitted_models_polycoeffs_txt(filename, X_vars_dict,bdegs_dict, knots_dict,alphas_dict, poly_coeffs_dict)

    return hat_y_dict








