#!/bin/sh

currenttime=$(date "+%Y%m%d%H%M%S")

if [ ! -d log ]; then
    mkdir log
fi

echo "[Usage] $0 config_path [train|eval] gpunum"

if [ ! -e $1 ]; then
    echo "[ERROR] Configuration file: $1 does not exist!"
    exit 1
fi

expname=$(basename "$1" .yaml)
if [ ! -d "$expname" ]; then
    mkdir "$expname"
fi

echo "[INFO] Saving results to, or loading files from: $expname"

if [ "$3" = "" ]; then
    echo "[ERROR] Please enter the number of GPUs"
    exit 1
fi

gpunum=$3
echo "[INFO] Number of GPUs: $gpunum"
ntask=$((gpunum * 3))

PYTHONCMD="python -u main_gpt_all.py --config $1"

case "$2" in
    train)
        $PYTHONCMD --train ;;
    eval)
        $PYTHONCMD --eval ;;
    visgt)
        $PYTHONCMD --visgt ;;
    anl)
        $PYTHONCMD --anl ;;
    sample)
        $PYTHONCMD --sample ;;
    *)
        echo "Invalid option: $2" ;;
esac
