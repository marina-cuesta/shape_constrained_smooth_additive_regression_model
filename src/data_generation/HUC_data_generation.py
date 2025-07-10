import os
import numpy as np
import pandas as pd

## path to save data
path_save_data = os.path.join(os.getcwd(), 'data', 'HUC')
os.makedirs(path_save_data, exist_ok=True)

## data name for filename
data_name = 'HUC'

###############################################################
#### deciding number of data based on number of parameters ####
###############################################################

## numer of parameters
n_X_vars = 2
n_intervals = 10
max_degree = 4

n_parameters = 1+ n_X_vars * (n_intervals+max_degree)

## number of data
n_data_per_n_parameters = 15
n_data = n_parameters * n_data_per_n_parameters


###################################
#### preparing X random sample ####
###################################

np.random.seed(1234)
epsilon = 0.01

#### variable v
Vlb = 15000000/1000
Vub = 33000000/1000
v = np.random.uniform(Vlb + epsilon, Vub - epsilon, n_data - 2)
v = np.concatenate(([Vlb], v, [Vub]))

#### variable q
Qlb = 8.5
Qub = 42
q = np.random.uniform(Qlb + epsilon, Qub - epsilon, n_data - 2)
q = np.concatenate(([Qlb], q, [Qub]))

# Create DataFrame
df = pd.DataFrame(
    np.column_stack((v, q)),
    columns=['v', 'q']
)


###########################################################
### computing p variable within the objective function ####
###########################################################

def HUC_data_p_variable(X):

    ## v and q variables
    v = X['v']
    q = X['q']

    ## params for p computation
    param_L = [4.09863600116008, -1.2553594229534, 0.160530264942775, -9.76201903589132E-03, 0.000309429429972963, -4.92928898248035E-06, 3.11519548768E-08]
    param_K = [307.395, 3.88E-05, -4.37E-12, 2.65E-19, -8.87E-27, 1.55E-34, -1.11E-42]
    param_Llb = 385
    param_R0 = 0.01

    ## p computation
    p = (9.81 / 1000) * q * sum((param_L[h] * q ** (h) * (
                sum((param_K[k] * (1000 * v) ** (k)) for k in range(7)) - param_Llb - param_R0 * q ** 2)) for h in
                            range(7))

    return np.array(p)

p = HUC_data_p_variable(df)
df['p']=p

## save df
filename = os.path.join(path_save_data,f"{data_name}_ndata_{str(n_data)}.csv ")
df.to_csv(filename, index=False)



