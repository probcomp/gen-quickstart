<!-- #region -->
# Gen Quick Start

This repository contains Jupyter notebooks that contain tutorials on specific features and/or applications of Gen.
The notebooks walk you through programs that use Gen.
Some notebooks also include exercises that expect you to write code or fill in written responses.

These notebooks assume some familiarity with the [Julia programming language](https://julialang.org/).

For reference on Gen see:

- The [Gen documentation](https://www.gen.dev/docs/dev/)

- Documentation for [GenPyTorch.jl](https://probcomp.github.io/GenPyTorch.jl/dev/) and [GenTF.jl](https://probcomp.github.io/GenTF/dev/)

- Other packages in the [Gen ecosystem](https://www.gen.dev/ecosystem)

You can also find HTML rendered versions of the tutorial notebooks in this repository [on the Gen website](https://www.gen.dev/tutorials).

## Getting started

To use the Jupyter notebooks in this repository, you will need to [install Jupyter](https://jupyter.org/install). 

Gen is a package for the Julia language, so you will also need to [download and install Julia](https://julialang.org/downloads/).
(Click the "help" link under your OS for specific installation instructions; we recommend following the instructions to "add Julia to PATH.")

Clone this repository, and inside the `gen-quickstart` directory, run

```bash
$ JULIA_PROJECT=. julia -e 'import Pkg; Pkg.instantiate()'
$ JULIA_PROJECT=. julia -e 'import Pkg; Pkg.build()'
```

to install Gen and its dependencies. You can then run

```bash
$ JULIA_PROJECT=. julia -e 'using IJulia; notebook(; dir=".")'
```

to start Jupyter. (It should also work to run `jupyter notebook` from your shell.) 
Navigate to the [Tutorials](Tutorials.ipynb) notebook to get started!

## TensorFlow or PyTorch integration

Some of our tutorials use [PyCall](https://github.com/JuliaPy/PyCall.jl) to interact with
Python installations of PyTorch and TensorFlow. These tutorials are in their own directories
with their own Julia `Project.toml` files, and require special setup.

To use them, make sure that `python` (version 3 only) is in your PATH and pointing to a Python environment that has PyTorch and TensorFlow installed. (This may in turn require installing other dependencies; please see PyTorch and TensorFlow websites for details. For more information on how PyCall and IJulia find the right versions of `python` and `jupyter`, please see the respective documentation of those packages.) Change into the tutorial directory of interest (e.g., `tutorials/pytorch`), and run:

```bash
$ export PYTHON=`which python`
$ JULIA_PROJECT=. julia -e 'import Pkg; Pkg.instantiate(); Pkg.build()'
```

When Julia builds `PyCall`, it should use the version of Python pointed to by your PYTHON environment variable.
<!-- #endregion -->
