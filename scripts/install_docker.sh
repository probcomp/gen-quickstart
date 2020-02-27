#!/usr/bin/env bash
if ! [[ -x "$(command -v docker)" ]]; then
    echo "Docker not found. Installing."
    sudo apt remove -y docker docker-engine docker.io
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    apt-key fingerprint 0EBFCD88
    sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
    sudo apt update && sudo apt install -y docker-ce
    sudo usermod -aG docker ${USER}
    docker -v
    echo "Please reboot to start using Docker with user rights."
    echo "If you need it before a restart, you can use: exec su - ${USER}"
else
    echo "Docker already installed. Skipping."
fi
