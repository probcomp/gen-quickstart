FROM            ubuntu:16.04
MAINTAINER      MIT Probabilistic Computing Project

RUN             apt-get update -qq \
                && apt-get install -qq -y \
                    hdf5-tools \
                    python3-dev \
                    python3-tk \
                    wget \
                    virtualenv \
                    zlib1g-dev

RUN             apt-get install -qq -y git
RUN             git config --global user.name "Gen User"
RUN             git config --global user.email "email@example.com"

RUN             virtualenv -p /usr/bin/python3 /venv
RUN             . /venv/bin/activate && pip install jupyter jupytext matplotlib tensorflow torch

# Specify Julia version. Find current version on https://julialang.org/downloads/
ARG             JULIA_VERSION_SHORT="1.5"
ARG             JULIA_VERSION_FULL="${JULIA_VERSION_SHORT}.3"
ENV             JULIA_INSTALLATION_PATH=/opt/julia

RUN             wget https://julialang-s3.julialang.org/bin/linux/x64/${JULIA_VERSION_SHORT}/julia-${JULIA_VERSION_FULL}-linux-x86_64.tar.gz && \
                tar zxf julia-${JULIA_VERSION_FULL}-linux-x86_64.tar.gz && \
                mkdir -p "${JULIA_INSTALLATION_PATH}" && \
                mv julia-${JULIA_VERSION_FULL} "${JULIA_INSTALLATION_PATH}/" && \
                ln -fs "${JULIA_INSTALLATION_PATH}/julia-${JULIA_VERSION_FULL}/bin/julia" /usr/local/bin/ && \
                rm julia-${JULIA_VERSION_FULL}-linux-x86_64.tar.gz && \
                julia -e 'import Pkg; Pkg.add("IJulia")'

ADD             . /gen-quickstart
ENV             JULIA_PROJECT=/gen-quickstart

RUN             . /venv/bin/activate && julia -e 'using Pkg; Pkg.build()'
RUN             . /venv/bin/activate && julia -e 'using Pkg; Pkg.API.precompile()'

WORKDIR         /gen-quickstart

ENTRYPOINT      . /venv/bin/activate && jupyter notebook \
                    --ip='0.0.0.0' \
                    --port=8080 \
                    --no-browser \
                    --NotebookApp.token= \
                    --allow-root \
                    --NotebookApp.iopub_data_rate_limit=-1
