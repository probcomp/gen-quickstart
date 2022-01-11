# Gen Scripts

This folder contains scripts for setting up Gen and its dependencies. They are tested on Ubuntu 18.04.4 LTS.

We first need to install Python and Julia as direct dependencies. Please run `install_python.sh` to install Python 3.7.6 if you do not already have it available and reload your shell. Please also note that we stay on 3.7, since Tensorflow GPU does not yet support Python 3.8 (at the time of writing) as well as that this will switch your system Python to 3.7.6. If you do not wish to do so, please modify the script and remove the line `pyenv global 3.7.6`. If you do this, you need to make sure the right Python version is available when you run Tensorflow whether through pyenv, virtual environments or other means.

Once Python is installed, please execute `install_julia.sh` to install Julia.

Afterwards, you are ready to install Gen via `install_gen_option_002_native.sh`. This script will natively install Gen and open a Jupyter notebook so you can start working with it.