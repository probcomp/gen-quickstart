# Gen Scripts

This folder contains scripts for setting up Gen and its dependencies. They are tested on Ubuntu 18.04.4 LTS.

Gen can be installed in two ways: Natively on Ubuntu or within Docker. We will describe both ways in dedicated sections.

## Docker

In order to install Docker, please use the script `install_docker.sh` and reboot to add your user to the docker group so you can run Docker without root privileges.

If you have an Nvidia graphics card, you might also want to execute `install_nvidia_docker.sh` to install Nvidia Docker. Please run `docker run --gpus all nvidia/cuda:9.0-base nvidia-smi` afterwards to verify you can see your graphics card inside Docker.

Once Docker is set up, you can run `install_gen_option_001_docker.sh` to build and run Gen within Docker. Once the container runs, it will ask you whether you would like to open Gen's Jupyter notebook in a browser.

## Native Installation of Gen
However, in order to more fluently develop and debug with Gen it might be desirable to install it into your local system. For this, we first need to install Python and Julia as direct dependencies.

Please run `install_python.sh` to install Python 3.7.6 if you do not already have it available and reload your shell. Please also note that we stay on 3.7, since Tensorflow GPU does not yet support Python 3.8 (at the time of writing) as well as that this will switch your system Python to 3.7.6. If you do not wish to do so, please modify the script and remove the line `pyenv global 3.7.6`. If you do this, you need to make sure the right Python version is available when you run Tensorflow whether through pyenv, virtual environments or other means.

Once Python is installed, please execute `install_julia.sh` to install Julia.

Afterwards, you are ready to install Gen via `install_gen_option_002_native.sh`. This script will natively install Gen and open a Jupyter notebook like with Docker so you can start working with it.

# Extras

This folder might occasionally contain extras. For instance, we currently provide an alternative Dockerfile that uses Ubuntu 18.04 instead of 16.04. In order to use it, please run
```bash
docker build -t gen:ubuntu1804 -f Dockerfile.ubuntu1804 .
docker run -d --name gen -p 8080:8080 -p 8090:8090 -p 8091:8091 -p 8092:8092 gen:ubuntu1804
```
