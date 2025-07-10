######################################################
######################################################
#### Data simulation of ex7_2_3 MINLPlib instance ####
######################################################
######################################################

import os
import numpy as np
import pandas as pd

## path to save data
path_save_data = os.path.join(os.getcwd(), 'data', 'MINLPlib_instances')
os.makedirs(path_save_data, exist_ok=True)

## data name for filename
data_name = 'ex7_2_3'

## number of data
n_data = 1500

###################################
#### preparing X random sample ####
###################################

np.random.seed(1234)
epsilon = 0.01

x1 = np.random.uniform( 100 + epsilon, 10000 - epsilon, n_data-2)
x1 = np.concatenate(([100], x1, [10000]))

x2 = np.random.uniform( 1000 + epsilon, 10000 - epsilon, n_data-2)
x2 = np.concatenate(([1000], x2, [10000]))

x3 = np.random.uniform( 1000 + epsilon, 10000 - epsilon, n_data-2)
x3 = np.concatenate(([1000], x3, [10000]))

x4 = np.random.uniform( 10 + epsilon, 1000 - epsilon, n_data-2)
x4 = np.concatenate(([10], x4, [1000]))

x5 = np.random.uniform( 10 + epsilon, 1000 - epsilon, n_data-2)
x5 = np.concatenate(([10], x5, [1000]))

x6 = np.random.uniform( 10 + epsilon, 1000 - epsilon, n_data-2)
x6 = np.concatenate(([10], x6, [1000]))

x7 = np.random.uniform( 10 + epsilon, 1000 - epsilon, n_data-2)
x7 = np.concatenate(([10], x7, [1000]))

x8 = np.random.uniform( 10 + epsilon, 1000 - epsilon, n_data-2)
x8 = np.concatenate(([10], x8, [1000]))

## generate the X vars df
df = pd.DataFrame(np.column_stack((x1, x2, x3, x4, x5, x6, x7, x8)), columns=['x1', 'x2','x3', 'x4', 'x5', 'x6', 'x7', 'x8'])


#################################################
### computing l variables in the constraints ####
#################################################

## compute l1 in constraint e2
l1 =  x1 * x6
df['l1'] = l1

## compute l2 in constraint e3
l2 =  x2 * x7 - x2 * x4
df['l2'] = l2

## compute l3 in constraint e4
l3 = x3*x8 - x3*x5
df['l3'] = l3


# ## some chexks!!!
# df["e2"] = 833.33252*df.x4/(df.x1/df.x6) + 100/df.x6 - 83333.333/(df.x1*df.x6)
# df["left_term"] = 833.33252*df.x4 + 100 * df.x1 - 83333.333
# df["constraint"] = df["l1"] >= df["left_term"]

# ## some checks!!!
# df["e3"] =  1250*x5/df.x2/df.x7 + df.x4/df.x7 - 1250*df.x4/df.x2/df.x7
# df["left_term"] = 1250*(x5 - df.x4)
# df["constraint"] = df["l1"] >= df["left_term"]


#########################
### saving dataframe ####
#########################

filename = os.path.join(path_save_data, "_".join((data_name,"data",str(n_data))) +  '.csv' )
df.to_csv(filename, index=False)



