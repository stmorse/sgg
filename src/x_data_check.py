import json
import os
import pandas as pd
import utils

sy, sm, ey, em = 2007, 1, 2011, 12
BASEPATH = '/sciclone/geograd/stmorse/reddit'
subreddit = 'science'

def count_entries_and_users():
    counts = [0 for _ in range(2)]
    authors_all = set()
    authors_sub = set()

    print('     Entries total | Entries sub | Users total | Users sub ')

    for yr, mo in utils.iterate_months(sy, sm, ey, em):
        metapath = os.path.join(BASEPATH, f'metadata/metadata_{yr}-{mo:02}.csv')
        metadata = pd.read_csv(metapath, compression='gzip')
        sr = metadata[metadata['subreddit']==subreddit]

        u_authors_all = set(metadata['author'].unique())
        u_authors_sub = set(sr['author'].unique())
        
        new_authors = u_authors_all - authors_all
        authors_all.update(new_authors)
        
        new_sub_authors = u_authors_sub - authors_sub
        authors_sub.update(u_authors_sub)

        n_entries_total = len(metadata)
        n_entries_sub   = len(sr)
        n_users_total   = len(u_authors_all)
        n_users_sub     = len(u_authors_sub)

        counts[0] += n_entries_total 
        counts[1] += n_entries_sub

        print(f'{yr}-{mo}: {n_entries_total} | {n_entries_sub} | {n_users_total} ({len(authors_all)}) | {n_users_sub} ({len(authors_sub)})')

    print(f'Grand totals: \nEntries: {counts[0]}, {counts[1]}\nAuthors: {len(authors_all)}, {len(authors_sub)}')

def count_entries_and_users_from_filtered():
    graph_file = f"graph_{sy}-{sm:02d}_{sy}-{sm+5:02d}_filtered.json"
    graph_path = os.path.join(BASEPATH, f'subreddit/{subreddit}/filtered', graph_file)

    print(f'Using path: {graph_path}')

    if os.path.exists(graph_path):
        with open(graph_path, 'r') as f:
            graph = json.load(f)
    else:
        return

    users = graph['user_to_idx'].keys()
    print(f'Checking entries of {len(users)} authors...')

    count = 0
    for yr, mo in utils.iterate_months(sy, sm, ey, em):
        metapath = os.path.join(BASEPATH, f'metadata/metadata_{yr}-{mo:02}.csv')
        metadata = pd.read_csv(metapath, compression='gzip')
        sr = metadata[metadata['subreddit']==subreddit]

        filtered_sr = sr[sr['author'].isin(users)]
        count += len(filtered_sr)
        print(f'{yr}-{mo}: {len(filtered_sr)}')

    print(f'Grand total entries: {count}')

if __name__=="__main__":
    count_entries_and_users_from_filtered()