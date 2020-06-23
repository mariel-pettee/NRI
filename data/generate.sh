#!/bin/bash
#SBATCH --partition day
#SBATCH --mem 10GB
#SBATCH --time 3:00:00
#SBATCH --job-name gen_short
#SBATCH --output generate_short.log

source ~/.bashrc
conda activate pytorch
python -u generate_dataset.py --num-train 10000 --num-valid 2000 --num-test 2000
