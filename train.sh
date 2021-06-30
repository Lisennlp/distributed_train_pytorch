# python -m torch.distributed.launch --nproc_per_node=4 --nnode=2 --node_rank=0 --master_addr=A_ip_address master_port=29500 main.py


    WORLD_SIZE=2
    # Change for multinode config
    MASTER_ADDR=192.168.53.8
    MASTER_PORT=29504
    node_rank=$1  # 机器编号
    nnodes=$2 # 总机器数量
    export OMP_NUM_THREADS=3
    # --no_strict \
    # CUDA_VISIBLE_DEVICES=0 python -W ignore \
    DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE --node_rank $node_rank --nnodes $nnodes --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
    CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch $DISTRIBUTED_ARGS main.py