"""
Summary:
Stores CSV of topic participation for top q users for each period.

Details:
% = reddit directory

- Loads metadata (%/metadata), labels (%/subreddit/<name>/labels)
- Joins metadata + labels, filters by top users (total posts)
- Pivots to get users -> participation by label for each period (in months)
Saves:
  user_label_counts_{start}_{end}.csv (uncompressed) -> %/subreddit/<name>/users
"""

import argparse
import configparser
import os
import time

import numpy as np
import pandas as pd

from utils import iterate_periods

def get_users(
    meta_path: str,
    label_path: str,
    user_path: str,
    subreddit: str,
    start_year: int,
    end_year: int,
    start_month: int,
    end_month: int,
    q: float,
    period: int,
):
    t0 = time.time()
    print(f'CPU count  : {os.cpu_count()}')
    print(f'Subreddit  : {subreddit}')
    print(f'Range      : {start_year}-{start_month:02} to {end_year}-{end_month:02}')
    print(f'Period     : {period} months\n')

    # Iterate over periods
    for (p_start_year, p_start_month, p_end_year, p_end_month) in iterate_periods(start_year, start_month, end_year, end_month, period):
        all_top_users_metadata = []
        # Compute the list of months for this period
        period_start_idx = p_start_year * 12 + (p_start_month - 1)
        period_end_idx = p_end_year * 12 + (p_end_month - 1)
        months_in_period = [m for m in range(period_start_idx, period_end_idx + 1)]
        
        for m in months_in_period:
            year = m // 12
            month = (m % 12) + 1
            print(f'Reading {year}-{month:02} ... ({time.time()-t0:.3f})')
            
            # load metadata for this subreddit
            print(f'> Loading metadata ... ({time.time()-t0:.3f})')
            filepath = os.path.join(meta_path, f'metadata_{year}-{month:02}.csv')
            try:
                metadata = pd.read_csv(filepath, compression='gzip')
            except Exception as e:
                print(f'Error reading {filepath}: {e}')
                continue
            
            metadata = metadata[metadata['subreddit'] == subreddit]
            print(f'> Loaded subreddit {subreddit} with {metadata.shape[0]} entries.')

            # load labels
            print(f'> Loading labels ... ({time.time()-t0:.3f})')
            labels_filepath = os.path.join(label_path, f'labels_{year}-{month:02}.npz')
            try:
                with open(labels_filepath, 'rb') as f:
                    labels = np.load(f)['labels']
            except Exception as e:
                print(f'Error reading {labels_filepath}: {e}')
                continue

            # Add labels as a column of metadata
            metadata['label'] = labels

            # Keep only the top users based on counts
            print(f'> Building top users ... ({time.time()-t0:.3f})')
            user_counts = metadata['author'].value_counts()
            top_users = user_counts[user_counts >= user_counts.quantile(1 - q)]
            # Filter metadata to include only top users
            top_users_metadata = metadata[metadata['author'].isin(top_users.index)]
            all_top_users_metadata.append(top_users_metadata)

        if not all_top_users_metadata:
            print("No metadata loaded for this period; skipping.")
            continue

        # Concatenate metadata for the period
        print(f'\nComplete period {p_start_year}-{p_start_month:02} to {p_end_year}-{p_end_month:02}. Saving ... ({time.time()-t0:.3f})')
        period_metadata = pd.concat(all_top_users_metadata)

        # Create a pivot table with users as rows and labels as columns
        user_label_counts = period_metadata.pivot_table(
            index='author', 
            columns='label', 
            aggfunc='size', 
            fill_value=0
        )

        # Name output file with period boundaries
        out_filename = f"user_counts_{p_start_year}-{p_start_month:02}_{p_end_year}-{p_end_month:02}.csv"
        user_label_counts.to_csv(os.path.join(user_path, out_filename))
    print(f'Complete. ({time.time()-t0:.3f})')


if __name__=="__main__":
    config = configparser.ConfigParser()
    config.read('config.ini')  #<- run from root directory
    g = config['general']

    parser = argparse.ArgumentParser()
    parser.add_argument('--subreddit', type=str, required=True)
    parser.add_argument('--start_year', type=int, required=True)
    parser.add_argument('--end_year', type=int, required=True)
    parser.add_argument('--start_month', type=int, default=1, required=False)
    parser.add_argument('--end_month', type=int, default=12, required=False)
    parser.add_argument('--q', type=float, default=0.1, required=False)
    parser.add_argument('--period', type=int, default=12, help="Period in months for aggregation")
    args = parser.parse_args()

    subpath = os.path.join(g['save_path'], args.subreddit)
    subdir = 'users'
    if not os.path.exists(os.path.join(subpath, subdir)):
        os.makedirs(os.path.join(subpath, subdir))
    
    get_users(
        meta_path=g['meta_path'],
        label_path=os.path.join(subpath, 'labels'),
        user_path=os.path.join(subpath, subdir),
        subreddit=args.subreddit,
        start_year=args.start_year,
        end_year=args.end_year,
        start_month=args.start_month,
        end_month=args.end_month,
        q=args.q,
        period=args.period,
    )