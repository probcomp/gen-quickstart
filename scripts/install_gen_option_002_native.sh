#!/usr/bin/env bash

# Precondition: Need Julia to be installed
if ! [[ -x "$(command -v julia)" ]]; then
    echo "Gen requires Julia. Please install it first. Thank you."
    exit 1
fi

# Install Dependencies
sudo apt update && apt install -y hdf5-tools python3-dev python3-tk virtualenv wget zlib1g-dev

# Clone Project
mkdir -p ${HOME}/GenProjects
ORIG_DIR=${PWD}
cd ${HOME}/GenProjects
git clone https://github.com/probcomp/gen-quickstart.git
cd gen-quickstart
export JULIA_PROJECT=${HOME}/GenProjects/gen-quickstart

# Create virtual environment
virtualenv -p /usr/bin/python3 env
# Blacklist it for git
echo "env/" > .gitignore
# Activate it
. ./env/bin/activate
# Install Python dependencies into virtual environment
pip install jupyter matplotlib tensorflow

# Compile Gen
julia -e 'using Pkg; Pkg.build()'
julia -e 'using Pkg; Pkg.API.precompile()'

# Run Gen
echo "Official Gen Getting Started Guide: https://probcomp.github.io/Gen/dev/getting_started/"
echo "You can return to your original directory with: cd ${ORIG_DIR}"
echo "You can exit the virtual environment with: source deactivate"
echo "In REPL you can press ] and type 'add https://github.com/probcomp/Gen' followed by Ctrl+C and 'using Pkg; Pkg.test(\"Gen\")'."
jupyter notebook --NotebookApp.iopub_data_rate_limit=-1
cd ${ORIG_DIR}
