#!/bin/bash
#SBATCH -p normal
#SBATCH -J 5s_proto5
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 100:00:00
#SBATCH --gres=dcu:2
#SBATCH -o /public/home/zhangzhong02/zpc/logs/5s_proto5.out
#SBATCH -e /public/home/zhangzhong02/zpc/logs/5s_proto5.err

module load compiler/devtoolset/7.3.1
module load compiler/rocm/dtk-22.04.2
module load mpi/hpcx/2.7.4/gcc-7.3.1
module load apps/anaconda3/5.2.0 

export PATH=/public/home/zhangzhong02/miniconda3/envs/dl/bin:$PATH
export PATH=/opt/hpc/software/mpi/hpcx/v2.7.4/gcc-7.3.1/bin/:$PATH
export LD_LIBRARY_PATH=/public/home/zhangzhong02/miniconda3/envs/dl/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:"*****"

python  main_proto.py --dataset_root dataset --config config/proto_5way_5shot_resnet12_msd.py --num_gpu 2 --mode train
