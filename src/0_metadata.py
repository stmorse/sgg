import argparse
import configparser
import os
import time

import pandas as pd

import utils

METADATA = [
    'id', 'created_utc', 'parent_id', 'subreddit', 'subreddit_id', 'author'
]

def get_metadata(
        data_path: str,
        save_path: str,
        start_year: int,
        end_year: int,
        start_month: int,
        end_month: int,
        chunk_size: int,
    ):
    
    t0 = time.time()
    
    print(f'Reading data from path  : {data_path}')
    print(f'Saving metadata to path : {save_path}\n')
    print(f'Saving {start_year}-{start_month} to {end_year}-{end_month}\n')
    
    for year, month in utils.iterate_months(start_year, start_month, end_year, end_month):
        print(f'Reading {year}-{month:02} ... ({time.time()-t0:.2f} s)')
        
        # iterator over metadata
        print(f'> Making iterator ... ({time.time()-t0:.2f} s)')
        reader = utils.read_file(
            year, month,
            return_type='metadata',
            metadata=METADATA,
            data_path=data_path,
            chunk_size=chunk_size,
        )

        # store metadata as list of dicts
        res = []

        # k will keep track of line num
        # and should align with embeddings
        # because we're using the same filter (no deleted authors)
        print(f'> Building ... ({time.time()-t0:.2f} s)')
        k = 0
        for chunk in reader:
            print(f'  Chunk {chunk_size} ... ({time.time()-t0:.2f} s)')
            for entry in chunk:
                entry['idx'] = k
                k += 1 
                res.append(entry)

        # convert to DataFrame
        df = pd.DataFrame(res)

        # save to csv
        print(f'> Saving ... ({time.time()-t0:.2f} s)')
        save_file = os.path.join(
            save_path,
            f'metadata_{year}-{month:02}.csv'
        )
        df.to_csv(
            save_file, 
            index=False,
            compression='gzip',
        )

        print(f'> Complete with {len(res)} entries. ({time.time()-t0:.2f} s)')


if __name__=="__main__":
    print('Running metadata.py ...')

    config = configparser.ConfigParser()
    config.read('config.ini')  #<- run from root project folder
    g = config['general']

    parser = argparse.ArgumentParser()
    parser.add_argument('--start_year', type=int, required=True)
    parser.add_argument('--end_year', type=int, required=True)
    parser.add_argument('--start_month', type=int, default=1, required=False)
    parser.add_argument('--end_month', type=int, default=12, required=False)
    parser.add_argument('--chunk_size', type=int, default=1000000, required=False)
    args = parser.parse_args()
    
    get_metadata(
        data_path=g['data_path'],
        save_path=g['meta_path'],
        start_year=args.start_year,
        end_year=args.end_year,
        start_month=args.start_month,
        end_month=args.end_month,
        chunk_size=args.chunk_size,
    )