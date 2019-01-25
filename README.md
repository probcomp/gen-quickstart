# Gen Jupyter Notebooks

This repository contains Jupyter notebooks that contain tutorials on specific features and/or applications of Gen.
The notebooks walk you through programs that use Gen.
Some notebooks also include exercises that expect you to write code or fill in written responses.
The recommended order of the notebooks is:

- tutorial-modeling-intro

- tutorial-iterative-inference

- .. More coming soon!

These notebooks assume some familiarity with the [Julia programming language](https://julialang.org/).

For reference on Gen see:

- The [Gen documentation](https://probcomp.github.io/Gen/dev/)

- The [GenTF documentation](https://probcomp.github.io/GenTF/dev/)

- The [GenViz documentation](https://probcomp.github.io/GenTF/dev/)

## Browser Requirements

Some of the notebooks make use of JavaScript-based visualizations that are displayed inline in the notebook.
These have been tested successfully with recent versions of Firefox and Google Chrome on Ubuntu and Mac OS.

## Docker Installation

The easiest way to run the notebooks is using Docker.

First obtain [docker](https://www.docker.com/).
Make sure to run the [post-installation steps](https://docs.docker.com/install/linux/linux-postinstall/) so that you can run the docker commands smoothly without needing sudo access.

Next, build the image using the following command:

    $ docker build -t gen:v0 .

Then run the image using:

    $ docker run -it --name gen -p 8080:8080 -p 8090:8090 gen:v0

Open `localhost:8080` in your browser and begin with `tutorial-modeling-intro`.

All the changes made to the notebooks will be saved in your docker container.

To stop the image, run `ctrl+c`.

To restart the image and resume your work, run:

    $ docker start -ia gen

## Manual Installation

These notebooks have been tested on Ubuntu Linux 16.04 and Mac OS X.
To install Gen natively on your machine, please view the commands taken in the [Dockerfile](./Dockerfile), which is based on Ubuntu Linux 16.04.
The steps in the Dockerfile can be reproduced on your machine but will require slight variations depending on your local development setup.

We recommend installing the Python dependencies, jupyter matplotlib tensorflow, into a dedicated Python virtual environment and working in that environment while running the installation commands.

### Running the notebooks

After running the installation steps above, activate your python virtual environment and start a Jupyter server by running the following from the `gen-examples/` directory:
```
JULIA_PROJECT=$(pwd) jupyter notebook
```
This command should open a browser window that shows the content of the `gen-examples/` directory.
By setting the environment variable `JULIA_PROJECT` to the `gen-examples/` directory, we are instructing Julia to use the Julia environment defined by `gen-examples/Manifest.toml`.
This environment has the necessary Julia dependencies required to run all the notebooks.

Each directory within `gen-examples/` contains a tutorial.
To run a tutorial in a given directory, use the Jupyter browser interface to open the `.ipynb` file within the directory, and run the cells one by one.
