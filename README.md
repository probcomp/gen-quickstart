# Gen Jupyter Notebooks

This repository contains Jupyter notebooks that contain tutorials on specific features and/or applications of Gen.
The notebooks walk you through programs that use Gen.
Some notebooks also include exercises that expect you to write code or fill in written responses.

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

Next, clone this Github repository using https:

    $ git clone https://github.com/probcomp/gen-examples.git

Next, build the image using the following command:

    $ docker build -t gen:v0 .

Then run the image using:

    $ docker run -it --name gen -p 8080:8080 -p 8090:8090 -p 8091:8091 -p 8092:8092 gen:v0

Open `localhost:8080` in your browser and open the 'Start Here.ipynb' notebook.

All the changes made to the notebooks will be saved in your docker container.

To stop the image, run `ctrl+c`.

To restart the image and resume your work, run:

    $ docker start -ia gen

## Manual Installation

These notebooks have been tested on Ubuntu Linux 16.04 and Mac OS X.
To install Gen natively on your machine, please view the commands taken in the [Dockerfile](./Dockerfile), which is based on Ubuntu Linux 16.04.
The steps in the Dockerfile can be reproduced on your machine but will require slight variations depending on your local development setup.

Below is a brief documentation of the steps taken in the Dockerfile.

1. Install global dependencies from Ubuntu APT.

    ```bash
    $ apt-get update -qq \
    && apt-get install -qq -y \
        hdf5-tools \
        python3-dev \
        python3-tk \
        wget \
        virtualenv \
        zlib1g-dev
    ```

2. Create a [Python virtual environment](https://virtualenv.pypa.io/en/latest/) for use with the examples (make sure to create a virtual environment in a writable directory), and insatll the following packages into this environment with `pip`:

    - [jupyter](https://jupyter.org/install#installing-jupyter-with-pip): used to run the notebook sever (required)
    - [matplotlib](https://matplotlib.org/users/installing.html#installing): used in many of the notebooks for basic plotting (required)
    - [tensorflow](https://www.tensorflow.org/install/pip): (recommended)

    ```bash
    $ virtualenv -p /usr/bin/python3 /venv
    $ . /venv/bin/activate && pip install jupyter matplotlib tensorflow
    ```

3. Download and install [Julia](https://julialang.org). Note that we create a soft-link of the `julia` executable in `/usr/bin/`. You should create a soft link to the executable to any writable directory that is on your PATH.

    ```bash
    $ wget https://julialang-s3.julialang.org/bin/linux/x64/1.0/julia-1.0.3-linux-x86_64.tar.gz
    $ tar -xzv < julia-1.0.3-linux-x86_64.tar.gz
    $ ln -s /julia-1.0.3/bin/julia /usr/bin/julia
    ```

4. Set the `JULIA_PROJECT` environment variable to `/path/to/gen-examples/` (i.e. path where this repository was cloned). Make that `JULIA_PROJECT` is set correctly on your machine whenever working with Gen or with these examples.

    ```bash
    $ export JULIA_PROJECT=/path/to/gen-examples
    ```

5. Build and precompile the Julia packages. The main libraries that are being built are [PyCall](https://github.com/JuliaPy/PyCall.jl) and [IJulia](https://github.com/JuliaLang/IJulia.jl). We make sure that `python` (version 3 only) and `jupyter` are in our PATH and pointing to the right Python environment and Jupyter installation, respectively, by activating the Python virtual environment we created. Since we have activated the virtual environment in the commands below, the `build()` command uses the version of Python and Jupyter in the virtual environment. (For more information on how PyCall and IJulia find the right versions of `python` and `jupyter`, please see the respective documentation of those packages.)

    ```bash
    $ . /venv/bin/activate && julia -e 'using Pkg; Pkg.build()'
    $ . /venv/bin/activate && julia -e 'using Pkg; Pkg.API.precompile()'
    ```

6. Run the Jupyter server! The notebooks should be available in your browser at `localhost:8080`. Remember to make sure that your `JULIA_PROJECT` is correctly set (step 4) before running this command again.

    ```bash
    $ . /venv/bin/activate && jupyter notebook \
                        --ip='0.0.0.0' \
                        --port=8080 \
                        --no-browser \
                        --NotebookApp.token= \
                        --allow-root \
                        --NotebookApp.iopub_data_rate_limit=-1
    ```

## Running the Notebooks

Start by opening the 'Start Here.ipynb' notebook, which contains links to the other notebooks in the intended order.
