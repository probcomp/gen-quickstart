#!/usr/bin/env bash
ORIG_DIR=${PWD}
cd /tmp
git clone https://github.com/probcomp/gen-quickstart.git
cd gen-quickstart/
docker build -t gen:v0 .
cd ${ORIG_DIR}
docker run -d --name gen -p 8080:8080 -p 8090:8090 -p 8091:8091 -p 8092:8092 gen:v0

if whiptail --yesno "Gen container was started. Would you like to open it in a browser now?" 20 60 ;then
    xdg-open http://localhost:8080
else
    echo "Please open localhost:8080 to start interacting with Gen."
fi

# Optional: GUI Dialog
# if zenity --question --text="Gen container was started. Would you like to open it in a browser now?" --ok-label=Yes --cancel-label=No
# then
#     xdg-open http://localhost:8080
# else
#     echo "Please open localhost:8080 to start interacting with Gen."
# fi
