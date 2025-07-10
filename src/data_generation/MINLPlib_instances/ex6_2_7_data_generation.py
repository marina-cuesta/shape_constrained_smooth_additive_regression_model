######################################################
######################################################
#### Data simulation of ex6_2_5 MINLPlib instance ####
######################################################
######################################################

import os
import numpy as np
import pandas as pd

## path to save data
path_save_data = os.path.join(os.getcwd(), 'data', 'MINLPlib_instances')
os.makedirs(path_save_data, exist_ok=True)

## data name for filename
data_name = 'ex6_2_7'

###############################################################
#### deciding number of data based on number of parameters ####
###############################################################

## number of parameters
n_X_vars = 9
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
x2 = np.random.uniform(1e-7 + epsilon, 0.4 - epsilon, n_data - 2)
x2 = np.concatenate(([1e-7], x2, [0.4]))

x3 = np.random.uniform(1e-7 + epsilon, 0.4 - epsilon, n_data - 2)
x3 = np.concatenate(([1e-7], x3, [0.4]))

x4 = np.random.uniform(1e-7 + epsilon, 0.4 - epsilon, n_data - 2)
x4 = np.concatenate(([1e-7], x4, [0.4]))

x5 = np.random.uniform(1e-7 + epsilon, 0.1 - epsilon, n_data - 2)
x5 = np.concatenate(([1e-7], x5, [0.1]))

x6 = np.random.uniform(1e-7 + epsilon, 0.1 - epsilon, n_data - 2)
x6 = np.concatenate(([1e-7], x6, [0.1]))

x7 = np.random.uniform(1e-7 + epsilon, 0.1 - epsilon, n_data - 2)
x7 = np.concatenate(([1e-7], x7, [0.1]))

x8 = np.random.uniform(1e-7 + epsilon, 0.5 - epsilon, n_data - 2)
x8 = np.concatenate(([1e-7], x8, [0.5]))

x9 = np.random.uniform(1e-7 + epsilon, 0.5 - epsilon, n_data - 2)
x9 = np.concatenate(([1e-7], x9, [0.5]))

x10 = np.random.uniform(1e-7 + epsilon, 0.5 - epsilon, n_data - 2)
x10 = np.concatenate(([1e-7], x10, [0.5]))

# Create DataFrame
df = pd.DataFrame(
    np.column_stack((x2, x3, x4, x5, x6, x7,x8,x9,x10)),
    columns=['x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']
)


#################################################
### computing y variable: objective function ####
#################################################
## for readability, we separate the generation of y values by blocks containing the same X variables

## compute y1
term1 = 2.4088 * x2 + 8.8495 * x5 + 2.0086 * x8
term2 = 2.248 * x2 + 7.372 * x5 + 1.868 * x8
term3 = 2.248 * x2 + 5.82088173817021 * x5 + 0.382446861901943 * x8
term4 = 0.972461133672523 * x2 + 7.372 * x5 + 1.1893141713454 * x8
term5 = 1.86752460515164 * x2 + 2.61699842799583 * x5 + 1.868 * x8
y1 = (
    np.log(term1) * (10.4807341082197 * x2 + 38.5043409542885 * x5 + 8.73945638067505 * x8)
    + 0.240734108219679 * np.log(x2) * x2
    + 2.64434095428848 * np.log(x5) * x5
    + 0.399456380675047 * np.log(x8) * x8
    - 0.240734108219679 * np.log(term1) * x2
    - 2.64434095428848 * np.log(term1) * x5
    - 0.399456380675047 * np.log(term1) * x8
    + 11.24 * np.log(x2) * x2
    + 36.86 * np.log(x5) * x5
    + 9.34 * np.log(x8) * x8
    - 11.24 * np.log(term2) * x2
    - 36.86 * np.log(term2) * x5
    - 9.34 * np.log(term2) * x8
    + np.log(term2) * term2
    + 2.248 * np.log(x2) * x2
    + 7.372 * np.log(x5) * x5
    + 1.868 * np.log(x8) * x8
    - 2.248 * np.log(term3) * x2
    - 7.372 * np.log(term4) * x5
    - 1.868 * np.log(term5) * x8
    - 12.7287341082197 * np.log(x2) * x2
    - 45.8763409542885 * np.log(x5) * x5
    - 10.607456380675 * np.log(x8) * x8
)

## compute y2
term1 = 2.4088 * x3 + 8.8495 * x6 + 2.0086 * x9
term2 = 2.248 * x3 + 7.372 * x6 + 1.868 * x9
term3 = 2.248 * x3 + 5.82088173817021 * x6 + 0.382446861901943 * x9
term4 = 0.972461133672523 * x3 + 7.372 * x6 + 1.1893141713454 * x9
term5 = 1.86752460515164 * x3 + 2.61699842799583 * x6 + 1.868 * x9
y2 = (
    np.log(term1) * (10.4807341082197 * x3 + 38.5043409542885 * x6 + 8.73945638067505 * x9)
    + 0.240734108219679 * np.log(x3) * x3
    + 2.64434095428848 * np.log(x6) * x6
    + 0.399456380675047 * np.log(x9) * x9
    - 0.240734108219679 * np.log(term1) * x3
    - 2.64434095428848 * np.log(term1) * x6
    - 0.399456380675047 * np.log(term1) * x9
    + 11.24 * np.log(x3) * x3
    + 36.86 * np.log(x6) * x6
    + 9.34 * np.log(x9) * x9
    - 11.24 * np.log(term2) * x3
    - 36.86 * np.log(term2) * x6
    - 9.34 * np.log(term2) * x9
    + np.log(term2) * term2
    + 2.248 * np.log(x3) * x3
    + 7.372 * np.log(x6) * x6
    + 1.868 * np.log(x9) * x9
    - 2.248 * np.log(term3) * x3
    - 7.372 * np.log(term4) * x6
    - 1.868 * np.log(term5) * x9
    - 12.7287341082197 * np.log(x3) * x3
    - 45.8763409542885 * np.log(x6) * x6
    - 10.607456380675 * np.log(x9) * x9
)

## compute y3
term1 = 2.4088 * x4 + 8.8495 * x7 + 2.0086 * x10
term2 = 2.248 * x4 + 7.372 * x7 + 1.868 * x10
term3 = 2.248 * x4 + 5.82088173817021 * x7 + 0.382446861901943 * x10
term4 = 0.972461133672523 * x4 + 7.372 * x7 + 1.1893141713454 * x10
term5 = 1.86752460515164 * x4 + 2.61699842799583 * x7 + 1.868 * x10
y3 = (
    np.log(term1) * (10.4807341082197 * x4 + 38.5043409542885 * x7 + 8.73945638067505 * x10)
    + 0.240734108219679 * np.log(x4) * x4
    + 2.64434095428848 * np.log(x7) * x7
    + 0.399456380675047 * np.log(x10) * x10
    - 0.240734108219679 * np.log(term1) * x4
    - 2.64434095428848 * np.log(term1) * x7
    - 0.399456380675047 * np.log(term1) * x10
    + 11.24 * np.log(x4) * x4
    + 36.86 * np.log(x7) * x7
    + 9.34 * np.log(x10) * x10
    - 11.24 * np.log(term2) * x4
    - 36.86 * np.log(term2) * x7
    - 9.34 * np.log(term2) * x10
    + np.log(term2) * term2
    + 2.248 * np.log(x4) * x4
    + 7.372 * np.log(x7) * x7
    + 1.868 * np.log(x10) * x10
    - 2.248 * np.log(term3) * x4
    - 7.372 * np.log(term4) * x7
    - 1.868 * np.log(term5) * x10
    - 12.7287341082197 * np.log(x4) * x4
    - 45.8763409542885 * np.log(x7) * x7
    - 10.607456380675 * np.log(x10) * x10
)

## compute y= y1+y2+y3
df['y']=y1+y2+y3

## save df
filename = os.path.join(path_save_data,f"{data_name}_ndata_{str(n_data)}.csv ")
df.to_csv(filename, index=False)



