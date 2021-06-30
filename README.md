
# 项目说明

pytorch分布式训练

## 单机器多卡
'''
sh train.sh 0 1
'''

## 多机器多卡

A机器运行

sh train.sh 0 2

B机器运行
sh train.sh 1 2

