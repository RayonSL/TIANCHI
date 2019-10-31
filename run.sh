#!/bin/bash
python -m torch.distributed.launch --nproc_per_node=3 --nnodes 4 --node_rank $1 --master_addr gpu13 tools/train.py
