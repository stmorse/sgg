#!/bin/tcsh
#SBATCH --job-name=nri
#SBATCH -N 1 -n 4
#SBATCH --gpus=1
#SBATCH -t 12:00:00

cd /sciclone/geograd/stmorse/gg
module load python/3.12.7
source venv/bin/activate.csh

python -u train.py --subfolder science --suffix _rc --dims 15 --timesteps 48 >& train.log
