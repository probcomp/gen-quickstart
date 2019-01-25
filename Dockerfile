FROM            ubuntu:16.04
MAINTAINER      MIT Probabilistic Computing Project

RUN             apt-get update -qq \
                && apt-get install -qq -y \
                    hdf5-tools \
                    python3-pip \
                    python3-tk \
                    wget \
                    zlib1g-dev

RUN             pip3 install jupyter matplotlib tensorflow

RUN             wget https://julialang-s3.julialang.org/bin/linux/x64/1.0/julia-1.0.3-linux-x86_64.tar.gz
RUN             tar -xzv < julia-1.0.3-linux-x86_64.tar.gz
RUN             ln -s /julia-1.0.3/bin/julia /usr/bin/julia

ADD             . /gen-examples
ENV             JULIA_PROJECT=/gen-examples

RUN             julia -e 'using Pkg; Pkg.build()'
RUN             julia -e 'using Pkg; Pkg.API.precompile()'

WORKDIR         /gen-examples

ENTRYPOINT      jupyter notebook \
                    --ip='0.0.0.0' \
                    --port=8080 \
                    --no-browser \
                    --NotebookApp.token= \
                    --allow-root \
                    --NotebookApp.iopub_data_rate_limit=-1
