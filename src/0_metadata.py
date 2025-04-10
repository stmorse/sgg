import argparse
import configparser
import os
import time

import pandas as pd

import utils

METADATA = ['id', 'created_utc', 'parent_id',
            'subreddit', 'subreddit_id', 'author']

def get_metadata(
        data_path: str,
        save_path: str,
        start_year: int,
        end_year: int,
        start_month: int,
        end_month: int
    ):
    
    t0 = time.time()
    years = range(start_year, end_year+1)
    months = range(start_month, end_month+1)
    
    print(f'Reading data from path     : {data_path}')
    print(f'Saving metadata to path     : {save_path}\n')
    print(f'Saving {start_year}-{start_month} to {end_year}-{end_month}\n')
    
    for year, month in [(yr, mo) for yr in years for mo in months]:
        print(f'> Reading {year}-{month:02} ... ', end=' ')
        # iterator over metadata
        reader = utils.read_file(
            year, month,
            return_type='metadata',
            metadata=METADATA,
            data_path=data_path,
            chunk_size=10000,
        )

        # store metadata as list of dicts
        res = []

        # k will keep track of line num
        # and should align with embeddings
        # because we're using the same filter (no deleted authors)
        k = 0
        for chunk in reader:
            for entry in chunk:
                entry['idx'] = k
                k += 1 
                res.append(entry)

        # convert to DataFrame
        df = pd.DataFrame(res)

        # save to csv
        save_file = os.path.join(
            save_path,
            f'metadata_{year}-{month:02}.csv'
        )
        df.to_csv(
            save_file, 
            index=False,
            compression='gzip',
        )

        print(f'Complete with {len(res)} entries. ({time.time()-t0:.2f} s)')


if __name__=="__main__":
    print('Running metadata.py ...')

    config = configparser.ConfigParser()
    config.read('../config.ini')
    g = config['general']

    parser = argparse.ArgumentParser()
    parser.add_argument('--start-year', type=int, required=True)
    parser.add_argument('--end-year', type=int, required=True)
    parser.add_argument('--start-month', type=int, default=1, required=False)
    parser.add_argument('--end-month', type=int, default=12, required=False)
    args = parser.parse_args()
    
    get_metadata(
        data_path=g['data_path'],
        save_path=g['meta_path'],
        start_year=args.start_year,
        end_year=args.end_year,
        start_month=args.start_month,
        end_month=args.end_month,
    )