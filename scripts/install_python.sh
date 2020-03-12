#!/usr/bin/env bash
sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev

curl https://pyenv.run | bash

if whiptail --yesno "Is it OK to install PyEnv into your .bashrc file and set your system Python to 3.7.6?" 20 60 ;then
    cp ~/.bashrc ~/.bashrc_backup_before_python_script
    echo 'export PATH="${HOME}/.pyenv/bin:${PATH}"' >> ~/.bashrc
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc
    echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
    export PATH="${HOME}/.pyenv/bin:${PATH}"
    source ~/.bashrc

    echo "You can use pyenv install --list to list available Python versions."
    pyenv install 3.7.6
    pyenv global 3.7.6
else
    echo ""
    echo "Please either run the following manually to setup Python or use your own method."
    echo "Place the following lines in ~/.bashrc"
    echo 'export PATH="${HOME}/.pyenv/bin:${PATH}"'
    echo 'eval "$(pyenv init -)"'
    echo 'eval "$(pyenv virtualenv-init -)"'
    echo ""
    echo "Update your PATH: export PATH=\"${HOME}/.pyenv/bin:${PATH}\""
    echo "Evaluate .bashrc: source ~/.bashrc"
    echo ""
    echo "You can use pyenv install --list to list available Python versions."
    echo "Afterwards, please install Python 3.7.6: pyenv install 3.7.6"
    echo "You might want to consider setting your system Python to it as well: pyenv global 3.7.6"
fi
