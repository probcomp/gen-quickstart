# gen-examples

This repository contains tutorial notebooks introducing different features and/or applications of Gen.
The notebooks walk you through programs that use Gen.
Some notebooks also include exercises that expect you to write code or write written responses.

## Installation

### Jupyter and IJulia

These tutorials use [Jupyter notebooks](https://jupyter.org/).
If you already have a `jupyter` installation, you can install the Julia notebook kernel by running the following from the `gen-examples/` directory:
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

After setting up the Python environment with the Python packages listed above, instruct the PyCall Julia package to use this Python environment by running the following from the `gen-examples/` directory:
```
JULIA_PROJECT=. julia -e 'using Pkg; ENV["PYTHON"] = "<python>"; Pkg.build("PyCall")'
```
where `<python>` is the absolute path to the python executable within the virtual environment you created.
If you have trouble building PyCall, see the [PyCall package documentation](https://github.com/JuliaPy/PyCall.jl#specifying-the-python-version).

## Running the notebooks

After running the installation steps above, start a Jupyter server by running the following from the `gen-examples/` directory:
```
JULIA_PROJECT=. jupyter notebook
```
This should open a browser window that shows the current directory listing.
Open the appropriate notebook and walk through the cells.

By setting the environment variable `JULIA_PROJECT` to the `gen-examples/` directory, we are instructing Julia to use the Julia environment defined by `gen-examples/Manifest.toml`.
This environment has the necessary Julia dependencies required to run all the notebooks.
