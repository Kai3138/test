#!/bin/sh

# 获取当前时间
currenttime=$(date "+%Y%m%d%H%M%S")

# 创建日志文件夹
if [ ! -d log ]; then
    mkdir log
fi

echo "[Usage] $0 config_path [train|eval] gpunum"

# 检查配置文件是否存在
if [ ! -e "$1" ]; then
    echo "[ERROR] Configuration file: $1 does not exist!"
    exit 1
fi

# 根据配置文件的基本名称创建一个文件夹
expname=$(basename "$1" .yaml)
if [ ! -d "$expname" ]; then
    mkdir "$expname"
fi

echo "[INFO] Saving results to, or loading files from: $expname"

# 设置 GPU 数量为 1
gpunum=1
echo "[INFO] Number of GPUs: $gpunum"
ntask=$((gpunum * 3))

# 设置 Python 命令
PYTHONCMD="python -u main.py --config $1"

# 根据命令参数执行相应的操作
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
