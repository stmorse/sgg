#!/bin/tcsh
#SBATCH --job-name=sgg_meta
#SBATCH -N 1 -n 4
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Load python and venv
cd /sciclone/geograd/stmorse/sgg
module load python/3.12.7
svenv

# Run subtopic clustering
python -u src/0_metadata.py \
  --start_year 2012 --end_year 2012 \
  --start_month 4 --end_month 12 \
  >& logs/metadata.log

