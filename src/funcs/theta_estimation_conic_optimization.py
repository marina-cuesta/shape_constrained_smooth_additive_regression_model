from mosek.fusion import *
from scipy.linalg import qr
import pandas as pd
from src.funcs.non_negativity_constraints_matrices import *
from src.funcs.Bspline_basis import Bspline_basis_and_penalty

def theta_estimation_conic_optimization(B, PI,
                                        X,y,extended_knots_list,bdeg_list,
                                        df_interpolation=None,
                                        df_underestimation=None,
                                        df_overestimation=None,
                                        lower_bound=None,upper_bound=None,
                                        bounds_constraint_modelling="bertsimas",
                                        omega_lower_bound = None, omega_upper_bound = None ):
    """
    Estimate the coefficient theta vector in a shape-constrained smooth additive regression model using B-splines.
    The estimation allows incorporating the following shape constraints:
        - Exact interpolation at specified data points.
        - Underestimation or overestimation at specified data points (pointwise).
        - Global or local (pointwise) lower and upper bounds on the estimated function.
    Global lower and upper bound constraints are enforced applying the polynomial non-negativity conditions
    described in Bertsimas and Popescu (2002).

    The objective is to minimize a penalized least-squares loss function, where the penalty term ensures
    coefficient identifiability. The optimization problem is reformulated as a conic program and solved
    using the MOSEK solver.

    :param B: (2D array) full design matrix containing the evaluation of the B-spline basis functions at the observed training values in X and a first column of ones
    :param PI: (2D array of) penalty matrix used to enforce identifiability
    :param X: (DataFrame) training observed data of the covariates X
    :param y: (1D array) training response y observed values corresponding to X
    :param extended_knots_list: (list of lists) extended knots sequences for each X variable. One list per variable
    :param bdeg_list: (list of ints) degrees of the B-spline basis for each x variable

    :param df_interpolation: (DataFrame, optional) data points where the estimated function must interpolate exactly
                             Rows in df_interpolation must be disjoint from those in df_underestimation and df_overestimation
    :param df_underestimation: (DataFrame, optional) data points where the estimated function must underestimate y
                               Rows in df_underestimation must be disjoint from those in df_interpolation and df_overestimation
    :param df_overestimation: (DataFrame, optional) data points where the estimated function must overestimate y
                              Rows in df_overestimation must be disjoint from those in df_interpolation and df_underestimation

    :param lower_bound: (float, optional) imposed lower bound to the estimated function
    :param upper_bound: (float, optional) imposed upper bound to the estimated function
    :param bounds_constraint_modelling: (str, optional) method to enforce the lower and/or upper bounds. If not None, must be either "pointwise" (local) or "bertsimas" (global)
    :param omega_lower_bound: (1D array, optional) weights for the univariate lower bound constraint per variable. This parameter must be not None if bounds_constraint_modelling=="bertsimas"
    :param omega_upper_bound: (1D array, optional) weights for the univariate upper bound constraint per variable. This parameter must be not None if bounds_constraint_modelling=="bertsimas"

    :return: (1D array) estimated theta coefficient vector
    """

    ##############################################################
    #### base conic optimization model (no shape constraints) ####
    ##############################################################

    ## initialize mosek model
    model = Model("theta estimation with conic optimization")

    ## compute penalized gram matrix of the model
    model_matrix = B.T @ B + PI

    ## checking if there is any dependent column in model_matrix and deleting them in model_matrix and B
    Q, R, P = qr(model_matrix, pivoting=True)
    rank = np.sum(np.abs(np.diag(R)) > 1e-9)
    dependent_cols = P[rank:]
    if len(dependent_cols) > 0:
        model_matrix = np.delete(model_matrix, dependent_cols, axis=1)
        model_matrix = np.delete(model_matrix, dependent_cols, axis=0)
        B = np.delete(B, dependent_cols, axis=1)

    ## initialize theta variable
    len_theta = B.shape[1]
    theta = model.variable("theta", len_theta, Domain.unbounded())

    ## initialize u variable
    u = model.variable("u", 1, Domain.greaterThan(0.0))

    ## define objective function
    objective_function_expr = Expr.sub(u, Expr.mul(2.0, Expr.dot(y, Expr.mul(B, theta))))
    model.objective("Objective", ObjectiveSense.Minimize, objective_function_expr)

    ## compute the cholesky decomposition of model_matrix such that model_matrix = F^t * F
    model_matrix_cone = np.linalg.cholesky(model_matrix, upper=True)

    ## set up the rotated cone constraint
    second_order_rotated_cone_constrain_expr = Expr.vstack(u, 0.5, Expr.mul(model_matrix_cone, theta))
    model.constraint("SecondOrderRotatedConeConstraint", second_order_rotated_cone_constrain_expr,
                     Domain.inRotatedQCone())


    ############################################################################
    #### interpolation/underestimation/overestimation pointwise constraints ####
    ############################################################################

    ## check that df_interpolation, df_underestimation and df_overestimation are different
    # df_interpolation vs df_underestimation
    if (df_interpolation is not None) and (df_underestimation is not None):
        common_12 = pd.merge(df_interpolation, df_underestimation, how='inner')
        if len(common_12)!=0:
            error_message = (f"df_interpolation and df_underestimation have common rows")
            raise ValueError(error_message)
    # df_interpolation vs df_overestimation
    if (df_interpolation is not None) and (df_overestimation is not None):
        common_13 = pd.merge(df_interpolation, df_overestimation, how='inner')
        if len(common_13)!=0:
            error_message = (f"df_interpolation and df_overestimation have common rows")
            raise ValueError(error_message)
    # df_underestimation vs df_overestimation
    if (df_underestimation is not None) and (df_overestimation is not None):
        common_23 = pd.merge(df_underestimation, df_overestimation, how='inner')
        if len(common_23) != 0:
            error_message = (f"df_underestimation and df_overestimation have common rows")
            raise ValueError(error_message)

    ## apply interpolation constraints
    if df_interpolation is not None:
        ## getting X and y values to interpolate
        X_interpolation = df_interpolation[X.columns]
        y_interpolation = df_interpolation['y'].to_numpy()

        ## ERROR CHECKING: all rows of df_interpolation must be in (X,y)
        # find matching indices of X_interpolation values in X
        matching_positions_X = [np.where((X.values == row).all(axis=1))[0][0] for row in X_interpolation.values]
        # check that all X_interpolation are in X
        cond1 = len(matching_positions_X) == len(X_interpolation)
        # check that y values also match
        cond2 = np.all(y[matching_positions_X] == y_interpolation)
        if not cond1 or not cond2:
            raise TypeError("X and y values in df_interpolation must be included in X and y")

        ## adding constraint B(X_interpolation)*theta = y_interpolation
        interpolation_expression =  Expr.mul(B[matching_positions_X,:],theta)
        model.constraint("interpolation constraint",interpolation_expression, Domain.equalsTo(y_interpolation))

    ## apply pointwise underestimation constraints
    if df_underestimation is not None:
        ## getting X and y values to interpolate
        X_underestimation = df_underestimation[X.columns]
        y_underestimation = df_underestimation['y'].to_numpy()

        ## NO ERROR CHECKING: the rows in df_overestimation might not be in observed (X,y) data

        ## computing B in X_underestimation
        B_underestimation, PI= Bspline_basis_and_penalty(X_underestimation, extended_knots_list, bdeg_list)
        # deleting columns if needed
        if len(dependent_cols) > 0:
            B_underestimation = np.delete(B_underestimation, dependent_cols, axis=1)

        ## adding constraint B_underestimation*theta = y_underestimation
        underestimation_expression =  Expr.mul(B_underestimation,theta)
        model.constraint("underestimation constraint",underestimation_expression, Domain.lessThan(y_underestimation))

    ## apply pointwise overestimation constraints
    if df_overestimation is not None:

        ## getting X and y values to interpolate
        X_overestimation = df_overestimation[X.columns]
        y_overestimation = df_overestimation['y'].to_numpy()

        ## NO ERROR CHECKING: the rows in df_overestimation might not be in observed (X,y) data

        ## computing B in X_overestimation
        B_overestimation,PI= Bspline_basis_and_penalty(X=X_overestimation,
                                                      extended_knots_list=extended_knots_list,
                                                      bdeg_list=bdeg_list)
        # deleting columns if needed
        if len(dependent_cols) > 0:
            B_overestimation = np.delete(B_overestimation, dependent_cols, axis=1)

        ## adding constraint B_overestimation*theta = y_underestimation
        overestimation_expression = Expr.mul(B_overestimation, theta)
        model.constraint("overestimation constraint", overestimation_expression, Domain.lessThan(y_overestimation))

    ############################################
    #### upper and lower bound constraints  ####
    ############################################

    if ((lower_bound is not None) or (upper_bound is not None)):

        ## checking inputs
        if (lower_bound is not None or upper_bound is not None) and  bounds_constraint_modelling not in ["bertsimas", "pointwise"]:
            raise TypeError("'bounds_constraint_modelling' must be one of 'bertsimas' or 'pointwise' when lower_bound or upper_bound is specified")
        if ((lower_bound is not None) and (upper_bound is not None)) and (lower_bound >= upper_bound):
            raise TypeError("'lower_bound' must be lower than 'upper_bound'")
        if ((lower_bound is not None) and (omega_lower_bound is None)):
            raise TypeError("If 'lower_bound' is not None, 'omega_lower_bound' must be a vector of weights for the bounds in each variable.")
        if ((upper_bound is not None) and (omega_upper_bound is None)):
            raise TypeError("If 'upper_bound' is not None, 'omega_upper_bound' must be a vector of weights for the bounds in each variable.")

        ## forcing alpha (theta[0]) to be np.mean(y)
        alpha = np.mean(y)
        model.constraint("Theta_0_fixed", theta.index(0), Domain.equalsTo(alpha))

        #### BOUNDS CONSTRAINT MODELING WITH POINTWISE APPROACH ####
        if bounds_constraint_modelling == "pointwise":

            # Constraint domain depending on lower or upper bound
            if lower_bound is None:
                constraint_domain = Domain.lessThan(upper_bound)
            elif upper_bound is None:
                constraint_domain = Domain.greaterThan(lower_bound)
            else:
                constraint_domain = Domain.inRange(lower_bound, upper_bound)

            ## adding bounds constraint with pointwise approach
            model.constraint("Bounds constraint pointwise", Expr.mul(B, theta), constraint_domain)


        #### BOUNDS CONSTRAINT MODELING WITH BERTSIMAS AND POPESCU APPROACH ####
        elif bounds_constraint_modelling == "bertsimas":

            ## information about variables
            var_names = X.columns
            d_var = len(var_names)

            ## slicing theta variable to each var
            # number of intervals in each variable (computed from extended_knots_list)
            n_intervals = np.array([len(inner_list) for inner_list in extended_knots_list]) - 1 - [2 * bdeg_var for bdeg_var in bdeg_list]
            # cutting points based on cumulative sum of number of intervals and degree
            cutting_points = np.concatenate(([0], np.cumsum(n_intervals) + np.cumsum(bdeg_list))) + 1
            # slicing thetas variables
            theta_dict = {
                i: theta.slice(cutting_points[i], cutting_points[i + 1])
                for i in range(len(cutting_points) - 1)
            }

            ## dictionaries to store Z variables for lower and upper constraints
            Z_lower_var_dict = {}
            Z_upper_var_dict = {}

            ## adding the constraints for each variable arising from the Bertsimas and Popescu non negativity conditions
            for var in range(d_var):

                ## name of the var
                var_name = var_names[var]
                ## degree and knots of the var
                bdeg_var = bdeg_list[var]
                extended_knots_var = extended_knots_list[var]
                ## number of intervals
                n_intervals_var = n_intervals[var]

                ## compute matrices H, W and G of this var
                H_matrices = compute_matrices_H(bdeg_var)
                W_matrices = compute_matrices_W(extended_knots_var, bdeg_var)
                G_matrices = compute_matrices_G(extended_knots_var, bdeg_var)

                #### constraints for lower bound
                if lower_bound is not None:

                    ## compute phi_lower value
                    phi_lower = (lower_bound - alpha) * omega_lower_bound[var]

                    ## create dictionary to store the Z_lower variables in this var
                    Z_lower_var_dict[var_name] = {}

                    ## add the corresponding constraints in each interval
                    for q in range(n_intervals_var):

                        ## getting matrices of interval q
                        Z_lower_var_dict[var_name][q] = model.variable(
                            f"Z_q_lower_bound_var_{var_name}_interval_{q}",
                            Domain.inPSDCone(bdeg_var + 1))
                        W_q = W_matrices[q]
                        G_q = G_matrices[q]

                        ## constraints for the first set of equalities
                        for l in range(bdeg_var):
                            model.constraint(
                                f"Lower bound: Trace_Constraint_first_equality_var_{var_name}_interval_{q}_l_{l}",
                                Expr.dot(H_matrices[l], Z_lower_var_dict[var_name][q]),
                                Domain.equalsTo(0))

                        ## constraints for the second set of equalities
                        # Compute Frobenius inner product ⟨H_l, Z_q⟩_F
                        frob_terms = []
                        for l in range(bdeg_var, len(H_matrices)):
                            frob_expr = Expr.dot(H_matrices[l], Z_lower_var_dict[var_name][q])
                            frob_terms.append(frob_expr)
                        frob_vector = Expr.vstack(frob_terms)
                        # expresion for W_q @ G_q @ theta
                        WG_theta_expr = Expr.mul(W_q @ G_q, theta_dict[var])
                        # compute W_q * phi_vector
                        phi_lower_vector = np.array([phi_lower] + [0] * bdeg_var)
                        W_phi_lower_result = W_q @ phi_lower_vector
                        # adding constraint
                        model.constraint(
                            f"Lower bound: Trace_Constraint_second_equality_var_{var_name}_interval_{q}",
                            Expr.sub(WG_theta_expr, frob_vector), Domain.equalsTo(W_phi_lower_result))


                #### constraints for upper bound
                if upper_bound is not None:

                    ## compute phi_upper value
                    phi_upper = (upper_bound - alpha) * omega_upper_bound[var]

                    ## create dictionary to store the Z_upper variables in this var
                    Z_upper_var_dict[var_name] = {}

                    ## add the corresponding constraints in each interval
                    for q in range(n_intervals_var):

                        ## getting matrices of interval q
                        Z_upper_var_dict[var_name][q] = model.variable(
                            f"Z_q_upper_bound_var_{var_name}_interval_{q}",
                            Domain.inPSDCone(bdeg_var + 1))
                        W_q = W_matrices[q]
                        G_q = G_matrices[q]

                        ## constraints for the first set of equalities
                        for l in range(bdeg_var):
                            model.constraint(
                                f"Upper bound: Trace_Constraint_first_equality_var_{var_name}_interval_{q}_l_{l}",
                                Expr.dot(H_matrices[l], Z_upper_var_dict[var_name][q]),
                                Domain.equalsTo(0))


                        ## constraints for the second set of equalities
                        # Compute Frobenius inner product ⟨H_l, Z_q⟩_F
                        frob_terms = []
                        for l in range(bdeg_var, len(H_matrices)):
                            frob_expr = Expr.dot(H_matrices[l], Z_upper_var_dict[var_name][q])
                            frob_terms.append(frob_expr)
                        frob_vector = Expr.vstack(frob_terms)
                        # expresion for W_q @ G_q @ theta
                        WG_theta_expr = Expr.mul(W_q @ G_q, theta_dict[var])
                        # compute W_q * phi_vector
                        phi_upper_vector = np.array([phi_upper] + [0] * bdeg_var)
                        W_phi_upper_result = W_q @ phi_upper_vector
                        # adding constraint
                        model.constraint(
                            f"Upper bound: Trace_Constraint_second_equality_var_{var_name}_interval_{q}",  Expr.add(frob_vector, WG_theta_expr),
                            Domain.equalsTo(W_phi_upper_result))


    ###########################################
    #### solving model to obtain hat_theta ####
    ###########################################
    model.solve()
    hat_theta = theta.level()

    ## if dependent columns in B were identified, retrieve full theta filled with zeros in the corresponding positions
    if len(dependent_cols)>0:
        n_total_basis_functions = len(hat_theta) + len(dependent_cols)
        # initialize full theta with zeros
        hat_theta_full = np.zeros(n_total_basis_functions)
        # create a mask to mark the positions that have not been deleted
        mask = np.ones(n_total_basis_functions, dtype=bool)
        mask[dependent_cols] = False
        # fill the positions that have not been deleted with hat_theta
        hat_theta_full[mask] = hat_theta
        # update hat_theta
        hat_theta = hat_theta_full

    ## checking if alpha == mean(y)
    if (abs(hat_theta[0] - np.mean(y)) > abs(0.01 * np.mean(y))):
        raise Exception("theta estimation failed: alpha != mean(y)")

    return hat_theta