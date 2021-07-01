
## 项目说明

pytorch分布式训练

## 环境

- torch==1.7.1

- python3.6.9

## 单机器多卡

sh train.sh 0 1

## 多机器多卡

A机器运行

sh train.sh 0 2

B机器运行

sh train.sh 1 2

## 采坑记录

- 两台机器的torch版本需要保持一致，我在实验过程中，一个使用torch1.2，一个使用1.7，导致两台机器跑步起来。

- 主机器的node_rank必须为0

