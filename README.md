# oceanmixedlayers

This package includes varoius methods for computing the ocean surface mixed layer depth from temperature, salinity, and density fields.  

## Installation (existing environment)  

To install from source:  
```
pip install -e .
```

### Creating a clean conda install with minimal external packages  

This package utilizes numpy and gsw for functionality of the mld software.  Notebooks and other tests may require additional packages, such as xarray and matplotlib.  To create a clean conda environment with the minimum required packages follow:  
```
conda create -n oceanmixedlayers numpy  
conda activate oceanmixedlayers  
conda install -c conda-forge gsw  
```  

You can then activate the oml environment and follow the installation instructions into an existing environment as above.  


### Optional additional packages to use all scripts and notebooks:  

*Warning* for some reason conda does not appear to successfully build these packages after gsw is installed.  Building these packages before installing gsw does seem to work.  This error should be better understood.  

```
conda install matplotlib jupyter netcdf4 xarray ipykernel  
```

A reminder, Jupyter environments can be added with:  
```
python -m ipykernel install --user --name myenv --display-name "oceanmixedlayers"  
```

## Instructions for using the installed package  

Examples and short tests are given in notebook form in the tests folder.  Some examples require downloading the Argo profile database (see ftp://usgodae.org/pub/outgoing/argo/). Idealized profiles can be constructed for testing the interfaces as well, without need for obtaining external data.  

The most useful example for usual implementation is probably in tests/Argo_Examples, which takes the Argo profiles and computes the gridded data for each of the algorithms included here.  

## Heat Content Example

Compute vertically integrated heat content (per unit area) down to a specified depth using Conservative Temperature. The calculation integrates `(CT - T_ref)` times a reference density and heat capacity with proper partial-cell handling.

```
import numpy as np
from oceanmixedlayers import oceanmixedlayers

# Define a simple column: 1 m layers to 300 m
N = 300
thck = np.ones(N)                # layer thickness [m]
z_c = -(np.arange(N) + 0.5)      # layer centers [m], negative downward
CT = np.ones(N) * 10.0           # Conservative Temperature [deg C]

oml = oceanmixedlayers()

# Heat content down to 200 m (negative downward)
hc = oml.heat_content(z_c, thck, CT, depth=-200.0)

# With a nonzero reference temperature (e.g., 2 deg C)
hc_ref = oml.heat_content(z_c, thck, CT, depth=-200.0, T_ref=2.0)

print("HC [J/m^2] =", hc)
print("HC with T_ref [J/m^2] =", hc_ref)
```

Notes:
- `depth` is negative downward (e.g., `-200.0` for 200 m).
- Defaults: `rho_ref=1025.0 kg/m^3`, `cp_ref=3991.86795711963 J/(kg K)`, `T_ref=0.0 C`.
- `z_c`, `thck`, and `CT` can be 1D or broadcastable ND arrays; NaNs are handled like the PE anomaly routines.
