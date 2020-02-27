#!/usr/bin/env bash
sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev

curl https://pyenv.run | bash

echo 'export PATH="${HOME}/.pyenv/bin:${PATH}"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
export PATH="${HOME}/.pyenv/bin:${PATH}"
source ~/.bashrc

echo "You can use pyenv install --list to list available Python versions."
pyenv install 3.7.6
pyenv global 3.7.6
