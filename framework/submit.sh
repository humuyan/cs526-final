#!/bin/bash
#SBATCH --nodes=1
#SBATCH --output=/pscratch/sd/z/zjia/hmy_workspace/sbatch_logs/%j.log
#SBATCH --error=/pscratch/sd/z/zjia/hmy_workspace/sbatch_logs/%j.log
#SBATCH --exclusive
#SBATCH --qos=regular
#SBATCH --time=24:00:00
#SBATCH --constraint gpu
#SBATCH --account=m4138_g

source /pscratch/sd/z/zjia/hmy_workspace/env.sh
python calc.py $1 a100 $2
