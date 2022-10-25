#!/bin/bash

if [ "$1" == "fetch" ]; then
    read -p "Do you want to fetch real_deployment from pi? (yes/no) " yn
    case $yn in
      yes )
        scp -r pi:~/Desktop/RPi  /home/tianchu/Documents/code_qy/puppybot-hw/
        echo "fetched the RPi folder from pi";;
      no )
        echo "not fetching the RPi folder from pi";;
      * ) echo invalid response;
        exit 1;;
    esac

elif [ "$1" == "push" ]; then
    read -p "Do you want to push RPi to pi? (yes/no) " yn
    case $yn in
      yes )
        scp -r ~/Documents/code_qy/puppybot-hw/RPi pi:~/Desktop/
        echo "pushed the RPi folder to pi";;
      no )
        echo "not pushing the RPi folder to pi";;
      * ) echo invalid response;
        exit 1;;
    esac

else
    echo "input illegal; should be one of 'fetch' and 'push'"
fi