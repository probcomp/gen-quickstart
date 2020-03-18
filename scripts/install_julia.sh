#!/usr/bin/env bash
# Find download link on https://julialang.org/downloads/
VERSION_LONG=1.3.1
VERSION_SHORT=1.3
wget https://julialang-s3.julialang.org/bin/linux/x64/${VERSION_SHORT}/julia-${VERSION_LONG}-linux-x86_64.tar.gz
tar xvfa julia-${VERSION_LONG}-linux-x86_64.tar.gz
rm julia-${VERSION_LONG}-linux-x86_64.tar.gz
sudo mv julia-${VERSION_LONG}/ /opt
sudo ln -s /opt/julia-${VERSION_LONG}/bin/julia /usr/local/bin/julia

# echo PATH=\$PATH:/opt/julia-${VERSION_LONG}/bin/ >> ~/.profile
# source ~/.profile

julia --version
