#!/bin/tcsh
#SBATCH --job-name=sgg
#SBATCH -N 1 -n 4
#SBATCH --mem=16G
#SBATCH --time=8:00:00

# Load python and venv
cd /sciclone/geograd/stmorse/sgg
module load python/3.12.7
svenv

# Run the Python script
python -u src/1_graph.py --start_year 2011 --end_year 2011 >& train.log