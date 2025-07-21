import os
import time
from src.funcs.data_fitting import *
from src.funcs.utils.error_metrics import *

def run_dataset(folder, data_name, n_data, y_vars, X_vars_dict, n_intervals_dict, bdegs_dict,
                dfs_interpolation_dict=None,
                dfs_underestimation_dict=None,
                dfs_overestimation_dict=None,
                lower_bounds_dict=None,
                upper_bounds_dict=None,
                bounds_constraint_modelling_dict=None):
    """
    Run the smooth additive regression modeling pipeline for a given dataset.

    Given a dataset name and number of data points (which together form the CSV filename), this function first reads
    the corresponding dataset. Then, it fits a smooth additive regression model (with optional shape constraintsif
    specified) for each response variable in `y_vars`, using the corresponding covariates specified in `X_vars_dict`.
    Supported optional shape constraints include:
         - Exact interpolation at selected data points
         - Pointwise underestimation and overestimation
         - Pointwise or global lower and upper bounds on the estimated function

    The function:
        1) Exports a `.txt` file containing the B-spline polynomial coefficients of the fitted model.
        2) Prints the execution time for fitting the model to each response variable in `y_vars`.
        3) Reports and prints in-sample prediction error metrics for each response variable using the observed data.


    :param folder: (str) subfolder name under 'results/txt files/' where the .txt output is saved
    :param data_name: (str) name of the dataset
    :param n_data: (int) number of data points indicated in the  csv filename to read
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

    :return: None
    """

    ## reading data
    path_read_data = os.path.join(os.getcwd(), 'data', folder)
    data_name_read = '_'.join((data_name, 'ndata', str(n_data)))
    data = pd.read_csv(os.path.join(path_read_data, data_name_read + '.csv'), sep=',')

    ## fitting a smooth additive regression model to each response variable in y_vars
    start_time = time.perf_counter()
    hat_y_dict, hat_theta = models_fitting_and_txt_result(
        folder=folder,
        data_name=data_name_read,
        data=data,
        y_vars=y_vars,
        X_vars_dict=X_vars_dict,
        n_intervals_dict=n_intervals_dict,
        bdegs_dict=bdegs_dict,
        dfs_interpolation_dict=dfs_interpolation_dict,
        dfs_underestimation_dict=dfs_underestimation_dict,
        dfs_overestimation_dict=dfs_overestimation_dict,
        lower_bounds_dict=lower_bounds_dict,
        upper_bounds_dict=upper_bounds_dict,
        bounds_constraint_modelling_dict=bounds_constraint_modelling_dict
    )
    end_time = time.perf_counter()
    print(f'Execution time for {data_name}: {end_time - start_time:.6f} seconds')

    ## in sample prediction errors
    for y_var in y_vars:
        y = data[y_var].to_numpy()
        hat_y = hat_y_dict[y_var]
        print(f'Prediction errors for {data_name}, response {y_var}:')
        print(prediction_errors(y, hat_y))
    print('-' * 50)


#############################
#### experiments results ####
#############################

## benchmark MINLPlib ex6_2_13 instance
run_dataset(
    folder = 'MINLPlib',
    data_name='ex6_2_13',
    n_data=1185,
    y_vars=['y'],
    X_vars_dict={'y': ['x2', 'x3','x4','x5', 'x6', 'x7']},
    n_intervals_dict={'y': [10, 10, 10,10, 10, 10]},
    bdegs_dict={'y': [3, 3, 3,3, 3, 3]}
)

## benchmark MINLPlib ex6_2_5 instance
run_dataset(
    folder='MINLPlib',
    data_name='ex6_2_5',
    n_data=1770,
    y_vars=['y'],
    X_vars_dict={'y': ['x2','x3','x4','x5','x6', 'x7','x8', 'x9','x10']},
    n_intervals_dict={'y': [10, 10, 10,10, 10, 10,10, 10, 10]},
    bdegs_dict={'y': [3, 3, 3,3, 3, 3, 3, 3, 3]},
)

## benchmark MINLPlib ex6_2_7 instance
run_dataset(
    folder='MINLPlib',
    data_name='ex6_2_7',
    n_data=1770,
    y_vars=['y'],
    X_vars_dict={'y': ['x2', 'x3','x4','x5','x6', 'x7','x8','x9', 'x10']},
    n_intervals_dict={'y': [10, 10, 10,10, 10, 10,10, 10, 10]},
    bdegs_dict={'y': [3, 3, 3,3, 3, 3, 3, 3, 3]},
)

## Real case study: Hydro Unit Commitment Problem
run_dataset(
    folder='HUC',
    data_name='HUC',
    n_data=435,
    y_vars=['p'],
    X_vars_dict={'p': ['v', 'q']},
    n_intervals_dict={'p': [10, 10]},
    bdegs_dict={'p': [4, 4]}, # degrees:[1,1], [2, 2], [3, 3], [4, 4]
    dfs_interpolation_dict={
        'p': pd.DataFrame({
            'v': [15000.0, 33000.0],
            'q': [8.5, 42.0],
            'y': [2.595542077205786, 24.08936535826008]
        })
    },
    lower_bounds_dict={'p': 2.595542077205786},
    upper_bounds_dict={'p': 24.08936535826008},
    bounds_constraint_modelling_dict={'p': 'bertsimas'}
)
