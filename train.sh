#!/bin/bash
#SBATCH --partition day
#SBATCH --mem 50GB
#SBATCH --time 6:00:00
#SBATCH --job-name train_dance_5joints
#SBATCH --output logs/train_nri_dance_5joints.log

source ~/.bashrc
conda activate nri
# module load CUDA/10.1.105
# python -u train.py --suffix _springs5 --epochs 10 --num-atoms 5 --dims 4 --edge-types 2 --save-folder logs/springs5_gpu/
python -u train.py --suffix _dance_5joints --epochs 500 --num-atoms 5 --dims 6 --edge-types 1 --save-folder logs/dance_5joints/
