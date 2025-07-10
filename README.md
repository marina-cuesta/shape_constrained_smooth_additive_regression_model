# Shape-Constrained Smooth Additive Regression Models

This repository provides a Python implementation to fit shape-constrained smooth additive regression models to a response variable `y` based on a set of covariates `X` with a  B-spline approach. The model supports the following shape constraints:

- Exact interpolation at specified data points,
- Pointwise underestimation or overestimation,
- Pointwise lower and upper bounds on the estimated function,
- Global lower and upper bounds on the estimated function across the entire observed `X` domain, enforced using polynomial non-negativity conditions as described in Bertsimas and Popescu (2002).

This repository is associated with the following paper:

> Cuesta, M., D’Ambrosio, C., Durban, M., Guerrero, V., & Trindade, R. S. (2025).  *On leveraging constrained smooth additive regression models for global optimization*. Manuscript submitted for publication.

## Repository structure

```  
shape_constrained_smooth_additive_regression_model/
│
├── data/                         # Input datasets
├── results/
│   └── txt files/                # Model output text files
├── src/
│   ├── funcs/                    # Core functions and utilities
│   ├── main.py                   # Main script to run models
├── environment.yml               # Conda environment
├── setup.py                      # (Optional) Setup file
└── README.md
```



## Installation

First, clone the repository and navigate to its root directory:

```bash
git clone https://github.com/marina-cuesta/shape_constrained_smooth_additive_regression_model.git
cd shape_constrained_smooth_additive_regression_model
```

Then, install the required dependencies using the provided environment.yml file. Create and activate the environment with:
```bash
conda env create -f environment.yml
conda activate additive_model_estimation_env
```
**Important**: This project uses on the MOSEK solver (version 11.0.4). A valid MOSEK license is required. Academic users can obtain a free license in https://www.mosek.com/products/academic-licenses/.

Finally, to enable Python imports like from src.funcs..., install the project locally using the provided setup.py:
```bash
pip install .
```

## Usage

The file `main.py` centralizes the code execution through the function `run_dataset`.  

To reproduce the approximation results used in the experiments section of the paper, the script includes one call to `run_dataset` per dataset instance.

Each call defines:
- `folder`: name of the subfolder inside the `data/` folder where the `.csv` file is located. This matchs the subfolder name in `results/` where the `.txt` results will be saved.
- `data_name` and `n_data`: the dataset name and number of data points, which together form the CSV filename to load.
- `y_vars`: list of response variables to approximate.
- `X_vars_dict`: dictionary specifying the covariates used to model each `y_var`.
- `n_intervals_dict`: the number of intervals to use in each covariate (list of number of intervals) of each `y_var` to model (dictionary),
- `bdegs_dict`:the degrees to use in each covariate (list of number of intervals) of each `y_var` to model (dictionary),
- Optional shape constraints:
  - `dfs_interpolation_dict`: dataframe for interpolation constraints on each `y_var`.
  - `dfs_underestimation_dict`: dataframe for pointwise underestimation constraints  on each `y_var`.
  - `dfs_overestimation_dict`: dataframe for pointwise overestimation constraints  on each `y_var`.
  - `lower_bounds_dict`: Lower bound on the estimated function for each `y_var`.
  - `upper_bounds_dict`: Upper bound on the estimated function for each `y_var`.
  - `bounds_constraint_modelling_dict`:  approach to model the lower and/or upper bounds in each `y_var`. Either "pointwise" for local constraints applied to the observed data or "bertsimas" for global constraints enforced via the Bertsimas and Popescu (2002) approach.



The following is an example call from `main.py`:

```python
run_dataset(
    folder='HUC', # the Hydro unit commitment problem folder
    data_name='HUC', # the HUC instance is selected  
    n_data=435,  # with 435 data points
    y_vars=['p'], # modelling the 'p' variable within the dataset
    X_vars_dict={'p': ['v', 'q']}, # 'p' is modelled with these covariates
    n_intervals_dict={'p': [10, 10]}, ## all covariates with 10 intervals
    bdegs_dict={'p': [4, 4]}, ## all covariates with degree 4
    dfs_interpolation_dict={
        'p': pd.DataFrame({
            'v': [15000.0, 33000.0],                          # the fitted model must interpolate at these specified data points
            'q': [8.5, 42.0],
            'y': [2.595542077205786, 24.08936535826008]
        })
    },   
    lower_bounds_dict={'p': 2.595542077205786}, # a lower bound for the estimated model is provided
    upper_bounds_dict={'p': 24.08936535826008}, # an upper bound for the estimated model is provided
    bounds_constraint_modelling_dict={'p': 'bertsimas'} # the bounds are required to be global across the whole X domain, and then, modelled with Bertsimas and Popescu (2002) approach.
)
```

You can modify the parameters. For example, if you wish to model covariate `'v'` with 5 intervals instead of 10:

```python
n_intervals_dict = {'p': [5, 10]}
```

## Contact

For questions, feedback, or collaboration inquiries, please contact:

Marina Cuesta
[marina.cuesta@uc3m.es](mailto:marina.cuesta@uc3m.es)  
Department of Statistics  
Universidad Carlos III de Madrid
