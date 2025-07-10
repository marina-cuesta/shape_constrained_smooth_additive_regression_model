#######################################################
#######################################################
#### Data simulation of ex6_2_13 MINLPlib instance ####
#######################################################
#######################################################

import os
import numpy as np
import pandas as pd

## path to save data
path_save_data = os.path.join(os.getcwd(), 'data', 'MINLPlib_instances')
os.makedirs(path_save_data, exist_ok=True)

## data name for filename
data_name = 'ex6_2_13'

###############################################################
#### deciding number of data based on number of parameters ####
###############################################################

## number of parameters
n_X_vars = 6
n_intervals = [10]*n_X_vars
degrees = [3]*n_X_vars
n_parameters = 1 + sum(n_intervals) + sum(degrees)

## number of data
n_data_per_n_parameters = 15
n_data = n_parameters * n_data_per_n_parameters

###################################
#### preparing X random sample ####
###################################

np.random.seed(1234)
epsilon = 0.01

# Generate random values within the given ranges with correct min and max values
x2 = np.random.uniform(1e-7 + epsilon, 0.08 - epsilon, n_data - 2)
x2 = np.concatenate(([1e-7], x2, [0.08]))

x3 = np.random.uniform(1e-7 + epsilon, 0.08 - epsilon, n_data - 2)
x3 = np.concatenate(([1e-7], x3, [0.08]))

x4 = np.random.uniform(1e-7 + epsilon, 0.3 - epsilon, n_data - 2)
x4 = np.concatenate(([1e-7], x4, [0.3]))

x5 = np.random.uniform(1e-7 + epsilon, 0.3 - epsilon, n_data - 2)
x5 = np.concatenate(([1e-7], x5, [0.3]))

x6 = np.random.uniform(1e-7 + epsilon, 0.62 - epsilon, n_data - 2)
x6 = np.concatenate(([1e-7], x6, [0.62]))

x7 = np.random.uniform(1e-7 + epsilon, 0.62 - epsilon, n_data - 2)
x7 = np.concatenate(([1e-7], x7, [0.62]))

# Create DataFrame
df = pd.DataFrame(
    np.column_stack((x2, x3, x4, x5, x6, x7)),
    columns=['x2', 'x3', 'x4', 'x5', 'x6', 'x7']
)


#################################################
### computing y variable: objective function ####
#################################################
## for readability, we separate the generation of y values by blocks containing the same X variables

## compute ly
y1 = (
        np.log(x2/(3*x2 + 6*x4 + x6))*x2 + np.log(x4/(3*x2 + 6*x4 + x6))*x4 + np.log(x6/(3*x2 + 6*x4 + x6))*x6
        + np.log(3*x2 + 6*x4 + 1.6*x6)*(3*x2 + 6*x4 + 1.6*x6)
        + 2*np.log(x2/(2.00000019368913*x2 + 4.64593*x4 + 0.480353*x6))*x2
        + np.log(x2/(1.00772874182154*x2 + 0.724703350369523*x4 + 0.947722362492017*x6))*x2
        + 6*np.log(x4/(3.36359157977228*x2 + 6*x4 + 1.13841069150863*x6))*x4
        + 1.6*np.log(x6/(1.6359356134845*x2 + 3.39220996773471*x4 + 1.6*x6))*x6
        - 3*np.log(x2)*x2 - 6*np.log(x4)*x4 - 1.6*np.log(x6)*x6
      )

## compute y2
y2 = (
        np.log(x3/(3*x3 + 6*x5 + x7))*x3 + np.log(x5/(3*x3 + 6*x5 + x7))*x5 + np.log(x7/(3*x3 + 6*x5 + x7))*x7
        + np.log(3*x3 + 6*x5 + 1.6*x7)*(3*x3 + 6*x5 + 1.6*x7)
        + 2*np.log(x3/(2.00000019368913*x3 +4.64593*x5 + 0.480353*x7))*x3
        + np.log(x3/(1.00772874182154*x3 +0.724703350369523*x5 + 0.947722362492017*x7))*x3
        + 6*np.log(x5/(3.36359157977228*x3 + 6*x5 + 1.13841069150863*x7))*x5
        + 1.6*np.log(x7/(1.6359356134845*x3 + 3.39220996773471*x5 + 1.6*x7))*x7
        - 3*np.log(x3)*x3 - 6*np.log(x5)*x5 - 1.6*np.log(x7)*x7
    )

## compute y= y1+y2
df['y'] = y1+y2

## save df
filename = os.path.join(path_save_data,f"{data_name}_ndata_{str(n_data)}.csv ")
df.to_csv(filename, index=False)



