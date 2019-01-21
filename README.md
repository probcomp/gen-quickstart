# gen-examples

## Installation

### Jupyter and IJulia

These tutorials use [Jupyter notebooks](https://jupyter.org/).
If you already have a `jupyter` installation, you can install the Julia notebook kernel using:
```
JUPYTER=$(which jupyter) JULIA_PROJECT=. julia -e 'using Pkg; Pkg.build("IJulia")'
```
If you do not already have a `jupyter` installation, the IJulia package will install one by itself:
```
JULIA_PROJECT=. julia -e 'using Pkg; Pkg.build("IJulia")'
```
If you have trouble installing Jupyter or the IJulia kernel, see the [IJulia package documentation](https://github.com/JuliaLang/IJulia.jl).

## Python environment and Python packages

Some tutorials also rely on Python modules installed in a Python environment.
We recommend creating a [Python virtual environment](https://virtualenv.pypa.io/en/latest/) for use with the examples, and installing the following packages into this environment with `pip`:

- [matplotlib](https://matplotlib.org/users/installing.html#installing): used in most of the tutorials

- [tensorflow](https://www.tensorflow.org/install/pip): used in one tutorial (optional)

The notebooks have been tested using a Python 3 environment.

After setting up the Python environment with the Python packages listed above, instruct the PyCall Julia package to use that environment by running:
```
julia -e 'using Pkg; ENV["PYTHON"] = "<python>"; Pkg.build("PyCall")'
```
where `<python`> is the absolute path to the python executable within the virtual environment.
