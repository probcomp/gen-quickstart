# Gen Example Jupyter Notebooks

This repository contains Jupyter notebooks that contain tutorials on specific features and/or applications of Gen.
The notebooks walk you through programs that use Gen.
Some notebooks also include exercises that expect you to write code or fill in written responses.

These notebooks assume some familiarity with the [Julia programming language](https://julialang.org/).

For reference on Gen see:

- The [Gen documentation](https://probcomp.github.io/Gen/dev/)

- The [GenTF documentation](https://probcomp.github.io/GenTF/dev/)

- The [GenViz documentation](https://probcomp.github.io/GenTF/dev/)

## Quick Installation

First obtain [docker](https://www.docker.com/).

Next, build the image using the following command:

    $ docker build -t gen:v0 .

Then run the image using:

    $ docker run -it --name gen -p 8080:8080 -p 8090:8090 gen:v0

Open `localhost:8080` in your browser and begin with `tutorial-modeling-intro`.

All the changes made to the notebooks will be saved in your docker container.

To stop the image, run `ctrl+c`.

To restart the image and resume your work, run:

    $ docker start -ia gen

## Advanced Installation

These notebooks have been tested on Ubuntu Linux 16.04 and Mac OS X.
Below we provide a rough documentation of the steps taken in the [Dockerfile](./Dockerfile), which you can reproduce manually to install Gen natively on your machine.

### Julia

The notebooks require [Julia 1.0](https://julialang.org/downloads/).
Before moving on to the next step, verify that you have a working Julia installation.

### Python environment and Python packages

The notebooks rely on Python modules installed in a Python 3 environment.
Create a [Python virtual environment](https://virtualenv.pypa.io/en/latest/) for use with the examples, and installing the following packages into this environment with `pip`:

- [jupyter](https://jupyter.org/install#installing-jupyter-with-pip): used to run the notebook sever (required)
- [matplotlib](https://matplotlib.org/users/installing.html#installing): used in many of the notebooks for basic plotting (required)
- [tensorflow](https://www.tensorflow.org/install/pip): used in one tutorial (optional)

After setting up the Python environment with the Python packages listed above, instruct the PyCall Julia package to use this Python environment by running the following from the `gen-examples/` directory:
```
JULIA_PROJECT=$(pwd) julia -e 'using Pkg; ENV["PYTHON"] = "<python>"; Pkg.build("PyCall")'
JULIA_PROJECT=$(pwd) julia -e 'using Pkg; ENV["PYTHON"] = "<python>"; Pkg.build("PyPlot")'
```
where `<python>` is the absolute path to the python3 executable within the virtual environment you created.
If you have trouble building PyCall, see the [PyCall package documentation](https://github.com/JuliaPy/PyCall.jl#specifying-the-python-version).

### Jupyter and IJulia

This repository uses [Jupyter notebooks](https://jupyter.org/).
First activate the virtual environment created in the previous section.

Install the Julia notebook kernel by running the following from the `gen-examples/` directory:
```
JUPYTER=$(which jupyter) JULIA_PROJECT=$(pwd) julia -e 'using Pkg; Pkg.build("IJulia")'
```
Next start the notebook server using
```
JULIA_PROJECT=$(pwd) jupyter notebook
```
and create a new IJulia notebook by navigating to New -> Julia 1.0.x in the Jupyter browser interface.
Now run the following command in a new cell to verify the installation is working (it may take a minute for the library to pre-compile).
```julia
using Gen
```

### Browser

Certain notebooks also make use Javascript-based visualizations that are generated in the output of certain notebook cells and displayed inline in the notebook.
These have been tested successfully with recent versions of Firefox and Google Chrome on Ubuntu and Mac OS.

## Running the notebooks

After running the installation steps above, activate your python virtual environment and start a Jupyter server by running the following from the `gen-examples/` directory:
```
JULIA_PROJECT=$(pwd) jupyter notebook
```
This command should open a browser window that shows the content of the `gen-examples/` directory.
By setting the environment variable `JULIA_PROJECT` to the `gen-examples/` directory, we are instructing Julia to use the Julia environment defined by `gen-examples/Manifest.toml`.
This environment has the necessary Julia dependencies required to run all the notebooks.

Each directory within `gen-examples/` contains a tutorial.
To run a tutorial in a given directory, use the Jupyter browser interface to open the `.ipynb` file within the directory, and run the cells one by one.
The recommended order of the notebooks is:

- tutorial-modeling-intro

- tutorial-iterative-inference

- .. More coming soon!
