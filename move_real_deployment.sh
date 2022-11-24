#!/bin/bash

if [ "$1" == "fetch" ]; then
    read -p "Do you want to fetch real_deployment from pi? (yes/no) " yn
    case $yn in
      yes )
        scp -r pi:~/Desktop/real_deployment  ~/Documents/code_qy/puppy-gym/
        echo "fetched the real_deployment folder from pi";;
      no )
        echo "not fetching the real_deployment folder from pi";;
      * ) echo invalid response;
        exit 1;;
    esac

elif [ "$1" == "push" ]; then
    read -p "Do you want to push real_deployment to pi? (yes/no) " yn
    case $yn in
      yes )
        scp -r ~/Documents/code_qy/puppy-gym/real_deployment pi:~/Desktop/
        scp -r ~/Documents/code_qy/puppy-gym/rl_utils pi:~/Desktop/
        echo "pushed the real_deployment folder to pi";;
      no )
        echo "not pushing the real_deployment folder to pi";;
      * ) echo invalid response;
        exit 1;;
    esac

else
    echo "input illegal; should be one of 'fetch' and 'push'"
fi