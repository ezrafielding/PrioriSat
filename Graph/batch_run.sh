#! /bin/bash

python main_gan.py --mode train --data_dir ../datasets/NWPU-Captions/NWPU_Graphs_100/ --N 100 --max_len 514 --name 100Nodes

python main_gan.py --mode train --data_dir ../datasets/NWPU-Captions/NWPU_Graphs_50/ --N 50 --max_len 514 --name 50Nodes