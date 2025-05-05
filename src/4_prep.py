"""
Inputs: co-reply graphs & user counts aggregated per period
Outputs: filtered graphs and filtered + normalized user features
(Filter: users must be active in >=T periods)
arbitrary-length month windows, with LOCF+decay for missing activity.
"""

# Global paths
GRAPH_DIR         = '/sciclone/geograd/stmorse/reddit/subreddit/science/graphs'
FEATURES_DIR      = '/sciclone/geograd/stmorse/reddit/subreddit/science/users'
OUT_GRAPH_DIR     = '/sciclone/geograd/stmorse/reddit/subreddit/science/filtered2'
OUT_FEATURES_DIR  = '/sciclone/geograd/stmorse/reddit/subreddit/science/filtered2'
AUTHORS_META_FILE = '/sciclone/geograd/stmorse/reddit/subreddit/science/filtered2/authors_intersection.csv'

import os
import json
import argparse

import numpy as np
import pandas as pd

from utils import iterate_periods

def main(args):
    # compute windows
    sy, sm, ey, em, p = (
        args.start_year, args.start_month, args.end_year, args.end_month,
        args.period
    )

    # gather authors present per window
    presence = {}
    for w in iterate_periods(sy, sm, ey, em, p):
        y0,m0,y1,m1 = w
        fn = f'user_counts_{y0}-{m0:02d}_{y1}-{m1:02d}.csv'
        df = pd.read_csv(os.path.join(FEATURES_DIR, fn), usecols=['author'])
        presence[w] = set(df['author'])

    # count across windows
    counts = {}
    for authors in presence.values():
        for u in authors:
            counts[u] = counts.get(u, 0) + 1
    # keep users active in at least T windows
    global_authors = sorted([u for u, c in counts.items() if c >= args.min_periods])
    global_u2i = {u: i for i, u in enumerate(global_authors)}

    # save authors meta CSV (binary flags)
    meta = pd.DataFrame({'author': global_authors})
    for w in iterate_periods(sy, sm, ey, em, p):
        col = f'{w[0]}-{w[1]:02d}_{w[2]}-{w[3]:02d}'
        meta[col] = meta['author'].isin(presence[w]).astype(int)
    meta.to_csv(AUTHORS_META_FILE, index=False)
    print(f'Saved authors (n={len(global_authors)}) to {AUTHORS_META_FILE}')

    # filter graphs per window
    for w in iterate_periods(sy, sm, ey, em, p):
        y0,m0,y1,m1 = w
        gfn = f'graph_{y0}-{m0:02d}_{y1}-{m1:02d}.json'
        with open(os.path.join(GRAPH_DIR, gfn)) as f:
            g = json.load(f)
        old2i = g['user_to_idx']; idx2u = {i:u for u,i in old2i.items()}

        src, dst = g['edge_index']; wts = g['edge_weight']
        new_src, new_dst, new_w = [], [], []
        for u,v,wt in zip(src, dst, wts):
            au, av = idx2u[u], idx2u[v]
            if au in global_u2i and av in global_u2i:
                new_src.append(global_u2i[au])
                new_dst.append(global_u2i[av])
                new_w.append(wt)
        out = {'user_to_idx': global_u2i,
               'edge_index': [new_src, new_dst],
               'edge_weight': new_w}
        out_fn = f'graph_{y0}-{m0:02d}_{y1}-{m1:02d}_filtered.json'
        with open(os.path.join(OUT_GRAPH_DIR, out_fn), 'w') as f:
            json.dump(out, f)
        print(f'Filtered graph {out_fn}: nodes={len(global_authors)}, edges={len(new_w)}')

    # filter & normalize features with LOCF+decay
    prev_feat = None
    for w in iterate_periods(sy, sm, ey, em, p):
        y0,m0,y1,m1 = w
        fn = f'user_counts_{y0}-{m0:02d}_{y1}-{m1:02d}.csv'
        df = pd.read_csv(os.path.join(FEATURES_DIR, fn)).set_index('author')
        cols = df.columns.tolist()
        # df2 zero-fills missing authors
        df2 = df.reindex(global_authors, fill_value=0)
        arr = df2[cols].values.astype(float)
        # normalize observed
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        active = norms[:,0] > 0
        norms[norms==0] = 1.0
        arr_norm = arr / norms

        if prev_feat is None:
            feat = arr_norm
        else:
            feat = arr_norm.copy()
            miss = ~active
            # group mean over observed
            if active.any():
                group_mean = arr_norm[active].mean(axis=0)
            else:
                group_mean = prev_feat.mean(axis=0)
            # LOCF + decay convex combination
            feat[miss] = args.decay * prev_feat[miss] + (1 - args.decay) * group_mean

        out_fn = f'features_{y0}-{m0:02d}_{y1}-{m1:02d}_filtered.npy'
        np.save(os.path.join(OUT_FEATURES_DIR, out_fn), feat)
        print(f'Filtered features {out_fn}: shape={feat.shape}')

        prev_feat = feat

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_year',   type=int, required=True)
    parser.add_argument('--start_month',  type=int, default=1)
    parser.add_argument('--end_year',     type=int, required=True)
    parser.add_argument('--end_month',    type=int, default=12)
    parser.add_argument('--period',       type=int, required=True)
    parser.add_argument('--min_periods',  type=int, required=True)  # min periods with activity to keep a user
    parser.add_argument('--decay',        type=float, default=0.9)
    args = parser.parse_args()

    os.makedirs(OUT_GRAPH_DIR, exist_ok=True)
    os.makedirs(OUT_FEATURES_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(AUTHORS_META_FILE), exist_ok=True)
    
    main(args)
