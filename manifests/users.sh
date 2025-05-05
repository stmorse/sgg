#!/bin/tcsh
#SBATCH --job-name=sgg_u
#SBATCH -N 1 -n 4
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Load python and venv
cd /sciclone/geograd/stmorse/sgg
module load python/3.12.7
svenv

# Run subtopic clustering
python -u src/3_users.py \
  --subreddit science \
  --start_year 2012 --end_year 2012 \
  --q 0.75 --period 6 \
  >& logs/users.log

