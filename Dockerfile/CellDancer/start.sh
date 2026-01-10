#!/bin/bash
# set -e: 當任何命令失敗時，腳本會立即退出
set -e

# 1. 以服務模式在後台啟動 SSH 守護程序
echo "Starting SSH service..."
service ssh start

# 2. 在前台運行 bash shell
# exec 命令會用 /bin/bash 進程替換掉當前的 shell 進程。
# 這很重要，因為它使得 bash 成為容器的主進程 (PID 1)，
# 這樣容器才能正確接收和處理來自 docker stop 的信號。
echo "Starting bash shell..."
exec bash