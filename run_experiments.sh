#!/bin/bash

# 执行第一个命令
bash experiment.sh --scenario S2 --method Ours --workers 2 --local-model
echo "第一个命令完成，暂停 2 分钟..."
sleep 120

# 执行第二个命令
bash experiment.sh --scenario S3 --method Ours --workers 2 --local-model
echo "第二个命令完成，暂停 2 分钟..."
sleep 120

# 执行第三个命令
bash experiment.sh --scenario S2 --method B3 --workers 2 --local-model
echo "第三个命令完成，暂停 2 分钟..."
sleep 120

# 执行第四个命令
bash experiment.sh --scenario S3 --method B3 --workers 2 --local-model
echo "全部命令执行完毕。"