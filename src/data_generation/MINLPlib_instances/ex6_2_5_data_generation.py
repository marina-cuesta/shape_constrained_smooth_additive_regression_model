import os
import numpy as np
import pandas as pd

## path to save data
path_save_data = os.path.join(os.getcwd(), 'data', 'MINLPlib_instances')
os.makedirs(path_save_data, exist_ok=True)

## data name for filename
data_name = 'ex6_2_5'

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
x2 = np.random.uniform(1e-7 + epsilon, 40.30707 - epsilon, n_data - 2)
x2 = np.concatenate(([1e-7], x2, [40.30707]))

x3 = np.random.uniform(1e-7 + epsilon, 40.30707 - epsilon, n_data - 2)
x3 = np.concatenate(([1e-7], x3, [40.30707]))

x4 = np.random.uniform(1e-7 + epsilon, 40.30707 - epsilon, n_data - 2)
x4 = np.concatenate(([1e-7], x4, [40.30707]))

x5 = np.random.uniform(1e-7 + epsilon, 5.14979 - epsilon, n_data - 2)
x5 = np.concatenate(([1e-7], x5, [5.14979]))

x6 = np.random.uniform(1e-7 + epsilon, 5.14979 - epsilon, n_data - 2)
x6 = np.concatenate(([1e-7], x6, [5.14979]))

x7 = np.random.uniform(1e-7 + epsilon, 5.14979 - epsilon, n_data - 2)
x7 = np.concatenate(([1e-7], x7, [5.14979]))

x8 = np.random.uniform(1e-7 + epsilon, 54.54314 - epsilon, n_data - 2)
x8 = np.concatenate(([1e-7], x8, [54.54314]))

x9 = np.random.uniform(1e-7 + epsilon, 54.54314 - epsilon, n_data - 2)
x9 = np.concatenate(([1e-7], x9, [54.54314]))

x10 = np.random.uniform(1e-7 + epsilon, 54.54314 - epsilon, n_data - 2)
x10 = np.concatenate(([1e-7], x10, [54.54314]))

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
# common denominator
common_denominator = x4 + x7 + x10
y1 = np.log(x4 / common_denominator) * x4 +  np.log(x7 / common_denominator) * x7 + np.log(x10 / common_denominator) * x10

## compute y2
# Compute intermediate terms
term1 = 3.9235 * x2 + 6.0909 * x5 + 0.92 * x8
term2 = 3.664 * x2 + 5.168 * x5 + 1.4 * x8
term3 = 4.0643 * x2 + 5.7409 * x5 + 1.6741 * x8
y2 = (
        np.log(term1) * (26.9071667605344 * x2 + 41.7710875549227 * x5 + 6.30931398488382 * x8)
        - 9.58716676053442 * np.log(term1) * x2
        - 16.9310875549227 * np.log(term1) * x5
        - 0.309313984883821 * np.log(term1) * x8
        - 18.32 * np.log(term2) * x2
        - 25.84 * np.log(term2) * x5
        - 7 * np.log(term2) * x8
        + np.log(term3) * term3
        - 4.0643 * np.log(4.0643 * x2 + 3.22644664511275 * x5 + 1.44980651607875 * x8) * x2
        - 5.7409 * np.log(5.31147575751424 * x2 + 5.7409 * x5 + 0.00729924451284409 * x8) * x5
        - 1.6741 * np.log(2.25846661774355 * x2 + 3.70876916588753 * x5 + 1.6741 * x8) * x8
        + 1.0000000000000178 * np.log(x2) * x2
        + np.log(x5) * x5
        + 1.0000000000000009 * np.log(x8) * x8
)


## compute y3
# Compute intermediate terms
term1 = 3.9235 * x3 + 6.0909 * x6 + 0.92 * x9
term2 = 3.664 * x3 + 5.168 * x6 + 1.4 * x9
term3 = 4.0643 * x3 + 5.7409 * x6 + 1.6741 * x9
y3 = (
    np.log(term1) * (26.9071667605344 * x3 + 41.7710875549227 * x6 + 6.30931398488382 * x9)
    - 9.58716676053442 * np.log(term1) * x3
    - 16.9310875549227 * np.log(term1) * x6
    - 0.309313984883821 * np.log(term1) * x9
    - 18.32 * np.log(term2) * x3
    - 25.84 * np.log(term2) * x6
    - 7 * np.log(term2) * x9
    + np.log(term3) * term3
    - 4.0643 * np.log(4.0643 * x3 + 3.22644664511275 * x6 + 1.44980651607875 * x9) * x3
    - 5.7409 * np.log(5.31147575751424 * x3 + 5.7409 * x6 + 0.00729924451284409 * x9) * x6
    - 1.6741 * np.log(2.25846661774355 * x3 + 3.70876916588753 * x6 + 1.6741 * x9) * x9
    + 1.0000000000000178 * np.log(x3) * x3
    + np.log(x6) * x6
    + 1.0000000000000009 * np.log(x9) * x9
)


## compute y= y1+y2+y3
df['y']=y1+y2+y3


## save df
filename = os.path.join(path_save_data,f"{data_name}_ndata_{str(n_data)}.csv ")
df.to_csv(filename, index=False)



