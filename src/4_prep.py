"""
Inputs: co-reply graphs & user counts aggregated per period
Outputs: filtered graphs and filtered+normalized user features
(Filter: users must be active in each period)
Supports arbitrary-length month windows.
"""

# Global paths — edit these to match your setup
GRAPH_DIR         = '/sciclone/geograd/stmorse/reddit/subreddit/science/graphs'
FEATURES_DIR      = '/sciclone/geograd/stmorse/reddit/subreddit/science/users'
OUT_GRAPH_DIR     = '/sciclone/geograd/stmorse/reddit/subreddit/science/filtered'
OUT_FEATURES_DIR  = '/sciclone/geograd/stmorse/reddit/subreddit/science/filtered'
AUTHORS_META_FILE = '/sciclone/geograd/stmorse/reddit/subreddit/science/filtered/authors_intersection.csv'

import os
import json
import argparse

import numpy as np
import pandas as pd
from utils import iterate_months


def month_index(year, month):
    return year * 12 + (month - 1)

def idx_to_ym(idx):
    return idx // 12, (idx % 12) + 1


def ensure_dirs():
    os.makedirs(OUT_GRAPH_DIR, exist_ok=True)
    os.makedirs(OUT_FEATURES_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(AUTHORS_META_FILE), exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_year',  type=int, required=True)
    parser.add_argument('--start_month', type=int, default=1, required=False)
    parser.add_argument('--end_year',    type=int, required=True)
    parser.add_argument('--end_month',   type=int, default=12, required=False)
    parser.add_argument('--period',      type=int, required=True)
    args = parser.parse_args()

    # compute windows
    idx0    = month_index(args.start_year, args.start_month)
    idx_end = month_index(args.end_year,   args.end_month)
    windows = []
    i = idx0
    while i <= idx_end:
        j = min(i + args.period - 1, idx_end)
        y0, m0 = idx_to_ym(i)
        y1, m1 = idx_to_ym(j)
        windows.append((y0, m0, y1, m1))
        i += args.period

    # 1) gather authors per window from feature CSVs
    presence = {}
    for (y0,m0,y1,m1) in windows:
        fn = f'user_counts_{y0}-{m0:02d}_{y1}-{m1:02d}.csv'
        df = pd.read_csv(os.path.join(FEATURES_DIR, fn), usecols=['author'])
        presence[(y0,m0,y1,m1)] = set(df['author'])

    # 2) compute intersection
    global_authors = sorted(set.intersection(*presence.values()))
    global_u2i     = {u:i for i,u in enumerate(global_authors)}

    # 3) save meta CSV
    meta = pd.DataFrame({'author': global_authors})
    for w in windows:
        col = f'{w[0]}-{w[1]:02d}_{w[2]}-{w[3]:02d}'
        meta[col] = meta['author'].isin(presence[w]).astype(int)
    meta.to_csv(AUTHORS_META_FILE, index=False)
    print(f'Saved authors intersection to {AUTHORS_META_FILE}')

    # 4) filter graphs per window
    for (y0,m0,y1,m1) in windows:
        graph_fn = f'graph_{y0}-{m0:02d}_{y1}-{m1:02d}.json'
        with open(os.path.join(GRAPH_DIR, graph_fn)) as f:
            g = json.load(f)
        old_u2i = g['user_to_idx']
        idx2u   = {i:u for u,i in old_u2i.items()}

        src,dst = g['edge_index']
        new_src, new_dst, new_w = [],[],[]
        for i,(u,v) in enumerate(zip(src,dst)):
            au, av = idx2u[u], idx2u[v]
            if au in global_u2i and av in global_u2i:
                new_src.append(global_u2i[au])
                new_dst.append(global_u2i[av])
                new_w.append(g['edge_weight'][i])
        out = {'user_to_idx': global_u2i,
               'edge_index': [new_src, new_dst],
               'edge_weight': new_w}
        out_fn = f'graph_{y0}-{m0:02d}_{y1}-{m1:02d}_filtered.json'
        with open(os.path.join(OUT_GRAPH_DIR, out_fn), 'w') as f:
            json.dump(out, f)
        print(f'Filtered graph saved: {out_fn} ({len(global_authors)} users)')

    # 5) filter & normalize features per window
    for (y0,m0,y1,m1) in windows:
        fn = f'user_counts_{y0}-{m0:02d}_{y1}-{m1:02d}.csv'
        df = pd.read_csv(os.path.join(FEATURES_DIR, fn)).set_index('author')
        cols = df.columns.tolist()
        df2 = df.reindex(global_authors, fill_value=0)
        arr = df2[cols].values.astype(float)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms==0] = 1.0
        feat = arr / norms
        out_fn = f'features_{y0}-{m0:02d}_{y1}-{m1:02d}_filtered.npy'
        np.save(os.path.join(OUT_FEATURES_DIR, out_fn), feat)
        print(f'Filtered features saved: {out_fn} shape={feat.shape}')

if __name__ == '__main__':
    ensure_dirs()
    main()
