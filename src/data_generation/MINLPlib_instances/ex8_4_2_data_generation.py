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
data_name = 'ex8_4_2'

###############################################################
#### deciding number of data based on number of parameters ####
###############################################################

## numer of parameters
n_X_vars = 7
max_n_intervals = 10

## parameters regarding variables xj+x22, xj^2+x23 and xj^3+x24, j = 1,3,5,7,9,11,13,15,17
n_parameters_1 = 3 * (max_n_intervals+2)

## parameters regarding x_j
n_parameters_2 = max_n_intervals + 3

## parameters regarding x_22
n_parameters_3 =  max_n_intervals + 2

## parameters regarding x_23
n_parameters_4 =  max_n_intervals + 2

## parameters regarding x_24
n_parameters_5 =  max_n_intervals + 2



## total number of parameters
n_parameters = 1+ n_parameters_1 + n_parameters_2 +n_parameters_3+n_parameters_4+n_parameters_5

## number of data
n_data_per_n_parameters = 15
n_data = n_parameters * n_data_per_n_parameters

###################################
#### preparing X random sample ####
###################################

np.random.seed(1234)
epsilon = 0.01


# Generate random values within the given ranges with correct min and max values
x1 = np.random.uniform(-0.5 + epsilon, 0.5 - epsilon, n_data - 2)
x1 = np.concatenate(([-0.5], x1, [0.5]))

x2 = np.random.uniform(5.4 + epsilon, 6.4 - epsilon, n_data - 2)
x2 = np.concatenate(([5.4], x2, [6.4]))

x3 = np.random.uniform(0.4 + epsilon, 1.4 - epsilon, n_data - 2)
x3 = np.concatenate(([0.4], x3, [1.4]))

x4 = np.random.uniform(4.9 + epsilon, 5.9 - epsilon, n_data - 2)
x4 = np.concatenate(([4.9], x4, [5.9]))

x5 = np.random.uniform(1.3 + epsilon, 2.3 - epsilon, n_data - 2)
x5 = np.concatenate(([1.3], x5, [2.3]))

x6 = np.random.uniform(3.9 + epsilon, 4.9 - epsilon, n_data - 2)
x6 = np.concatenate(([3.9], x6, [4.9]))

x7 = np.random.uniform(2.1 + epsilon, 3.1 - epsilon, n_data - 2)
x7 = np.concatenate(([2.1], x7, [3.1]))

x8 = np.random.uniform(4.1 + epsilon, 5.1 - epsilon, n_data - 2)
x8 = np.concatenate(([4.1], x8, [5.1]))

x9 = np.random.uniform(2.8 + epsilon, 3.8 - epsilon, n_data - 2)
x9 = np.concatenate(([2.8], x9, [3.8]))

x10 = np.random.uniform(3 + epsilon, 4 - epsilon, n_data - 2)
x10 = np.concatenate(([3], x10, [4]))

x11 = np.random.uniform(3.9 + epsilon, 4.9 - epsilon, n_data - 2)
x11 = np.concatenate(([3.9], x11, [4.9]))

x12 = np.random.uniform(3.2 + epsilon, 4.2 - epsilon, n_data - 2)
x12 = np.concatenate(([3.2], x12, [4.2]))

x13 = np.random.uniform(4.7 + epsilon, 5.7 - epsilon, n_data - 2)
x13 = np.concatenate(([4.7], x13, [5.7]))

x14 = np.random.uniform(2.3 + epsilon, 3.3 - epsilon, n_data - 2)
x14 = np.concatenate(([2.3], x14, [3.3]))

x15 = np.random.uniform(5.6 + epsilon, 6.6 - epsilon, n_data - 2)
x15 = np.concatenate(([5.6], x15, [6.6]))

x16 = np.random.uniform(2.3 + epsilon, 3.3 - epsilon, n_data - 2)
x16 = np.concatenate(([2.3], x16, [3.3]))

x17 = np.random.uniform(6 + epsilon, 7 - epsilon, n_data - 2)
x17 = np.concatenate(([6], x17, [7]))

x18 = np.random.uniform(1.9 + epsilon, 2.9 - epsilon, n_data - 2)
x18 = np.concatenate(([1.9], x18, [2.9]))

x19 = np.random.uniform(6.9 + epsilon, 7.9 - epsilon, n_data - 2)
x19 = np.concatenate(([6.9], x19, [7.9]))

x20 = np.random.uniform(1 + epsilon, 2 - epsilon, n_data - 2)
x20 = np.concatenate(([1], x20, [2]))

x21 = np.random.uniform(0 + epsilon, 10 - epsilon, n_data - 2)
x21 = np.concatenate(([0], x21, [10]))

x22 = np.random.uniform(-2 + epsilon, 2 - epsilon, n_data - 2)
x22 = np.concatenate(([-2], x22, [2]))

x23 = np.random.uniform(-2 + epsilon, 2 - epsilon, n_data - 2)
x23 = np.concatenate(([-2], x23, [2]))

x24 = np.random.uniform(-2 + epsilon, 2 - epsilon, n_data - 2)
x24 = np.concatenate(([-2], x24, [2]))


# Create DataFrame
df = pd.DataFrame(
    np.column_stack((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12,
                     x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24)),
    columns=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10',
             'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19',
             'x20', 'x21', 'x22', 'x23', 'x24']
)


###########################################
### defining variable l to approximate ####
###########################################

l1 = x22*x1 + x1**2*x23 + x1**3*x24

l2 = x22*x3 + x3**2*x23 + x3**3*x24

l3 = x22*x5 + x5**2*x23 + x5**3*x24

l4 = x22*x7 + x7**2*x23 + x7**3*x24

l5 = x22*x9 + x9**2*x23 + x9**3*x24

l6 = x22*x11 + x11**2*x23 + x11**3*x24

l7 = x22*x13 + x13**2*x23 + x13**3*x24

l8 = x22*x15 + x15**2*x23 + x15**3*x24

l9 = x22*x17 + x17**2*x23 + x17**3*x24

l10 = x22*x19 + x19**2*x23 + x19**3*x24

## adding the variables to df
df['l1'] = l1
df['l2'] = l2
df['l3'] = l3
df['l4'] = l4
df['l5'] = l5
df['l6'] = l6
df['l7'] = l7
df['l8'] = l8
df['l9'] = l9
df['l10'] = l10


#########################
### saving dataframe ####
#########################

filename = os.path.join(path_save_data, "_".join((data_name,"data",str(n_data))) +  '.csv' )
df.to_csv(filename, index=False)


