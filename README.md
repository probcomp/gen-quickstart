# Gen Example Jupyter Notebooks

This repository contains Jupyter notebooks that contain tutorials on specific features and/or applications of Gen.
The notebooks walk you through programs that use Gen.
Some notebooks also include exercises that expect you to write code or fill in written responses.

These notebooks assume some familiarity with the [Julia programming language](https://julialang.org/).

For reference on Gen see:

- The [Gen documentation](https://probcomp.github.io/Gen/dev/)

- The [GenTF documentation](https://probcomp.github.io/GenTF/dev/)

- The [GenViz documentation](https://probcomp.github.io/GenTF/dev/)


## Installation

These notebooks have been tested on Ubuntu Linux and Mac OS X.

### Julia

The notebooks require [Julia 1.0](https://julialang.org/downloads/).
Before moving on to the next step, verify that you have a working Julia installation.

### Jupyter and IJulia

This repository uses [Jupyter notebooks](https://jupyter.org/).
If you already have a `jupyter` installation, you can install the Julia notebook kernel by running the following from the `gen-examples/` directory:
```
JUPYTER=$(which jupyter) JULIA_PROJECT=. julia -e 'using Pkg; Pkg.build("IJulia")'
```
If you do not already have a `jupyter` installation, the IJulia package will install one by itself:
```
JULIA_PROJECT=. julia -e 'using Pkg; Pkg.build("IJulia")'
```
If you have trouble installing Jupyter or the IJulia kernel, see the [IJulia package documentation](https://github.com/JuliaLang/IJulia.jl).
Before moving onto the next step, verify that you have a working Jupyter / IJulia installation, by launching a Jupyter server:
```
JULIA_PROJECT=. jupyter notebook
```
and creating a new IJulia notebook by navigating to New -> Julia 1.0.x in the Jupyter browser interface, and running the following:
```julia
using Gen
```

### Python environment and Python packages

Some notebooks also rely on Python modules installed in a Python environment.
We recommend creating a [Python virtual environment](https://virtualenv.pypa.io/en/latest/) for use with the examples, and installing the following packages into this environment with `pip`:

- [matplotlib](https://matplotlib.org/users/installing.html#installing): used in many of the notebooks for basic plotting

- [tensorflow](https://www.tensorflow.org/install/pip): used in one tutorial (optional)

The notebooks have been tested using a Python 3 environment.

After setting up the Python environment with the Python packages listed above, instruct the PyCall Julia package to use this Python environment by running the following from the `gen-examples/` directory:
```
JULIA_PROJECT=. julia -e 'using Pkg; ENV["PYTHON"] = "<python>"; Pkg.build("PyCall")'
```
where `<python>` is the absolute path to the python executable within the virtual environment you created.
If you have trouble building PyCall, see the [PyCall package documentation](https://github.com/JuliaPy/PyCall.jl#specifying-the-python-version).

### Browser

Certain notebooks also make use Javascript-based visualizations that are generated in the output of certain notebook cells and displayed inline in the notebook.
These have been tested succesfully with recent versions of Firefox and Google Chrome on Ubuntu and Mac OS.

## Running the notebooks

After running the installation steps above, start a Jupyter server by running the following from the `gen-examples/` directory:
```
JULIA_PROJECT=. jupyter notebook
```
This should open a browser window that shows the content of the `gen-examples/` directory.
By setting the environment variable `JULIA_PROJECT` to the `gen-examples/` directory, we are instructing Julia to use the Julia environment defined by `gen-examples/Manifest.toml`.
This environment has the necessary Julia dependencies required to run all the notebooks.

Each directory within `gen-examples/` contains a tutorial.
To run a tutorial in a given directory, use the Jupyter browser interface to open the `.ipynb` file within the directory, and run the cells one by one.
The recommended order of the notebooks is:

- tutorial-modeling-intro

- tutorial-iterative-inference

- tutorial-importance-sampling
