#!/usr/bin/env python3
"""
Prepare unioned graphs and feature matrices across a range of years,
so that every graph and feature file shares the same node ordering.
"""

# Global paths — edit these to match your setup
GRAPH_DIR         = '/sciclone/geograd/stmorse/reddit/subreddit/science/links'
FEATURES_DIR      = '/sciclone/geograd/stmorse/reddit/subreddit/science/users'
OUT_GRAPH_DIR     = '/sciclone/geograd/stmorse/reddit/subreddit/science/filtered'
OUT_FEATURES_DIR  = '/sciclone/geograd/stmorse/reddit/subreddit/science/filtered'
AUTHORS_META_FILE = '/sciclone/geograd/stmorse/reddit/subreddit/science/filtered/authors_union.csv'

import os
import json
import argparse

import numpy as np
import pandas as pd


def ensure_dirs():
    os.makedirs(OUT_GRAPH_DIR,    exist_ok=True)
    os.makedirs(OUT_FEATURES_DIR, exist_ok=True)
    meta_dir = os.path.dirname(AUTHORS_META_FILE)
    if meta_dir:
        os.makedirs(meta_dir, exist_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--start_year', type=int, required=True)
    p.add_argument('--end_year',   type=int, required=True)
    p.add_argument('--decay',      type=float, default=0.9)
    args = p.parse_args()

    years = list(range(args.start_year, args.end_year + 1))
    union_authors = set()
    presence = {}  # for the meta–CSV

    # 1) Gather all authors (from both graphs and feature CSVs)
    for yr in years:
        # features
        feat_csv = os.path.join(FEATURES_DIR, f'user_counts_{yr}.csv')
        df_feat = pd.read_csv(feat_csv, usecols=['author'])
        authors_f = set(df_feat['author'])
        presence[yr] = authors_f
        union_authors |= authors_f

        # graph
        graph_json = os.path.join(GRAPH_DIR, f'graph_{yr}.json')
        with open(graph_json, 'r') as f:
            g = json.load(f)
        authors_g = set(g['user_to_idx'].keys())
        union_authors |= authors_g

    # build global ordering
    global_authors = sorted(union_authors)
    global_u2i = {u: i for i, u in enumerate(global_authors)}

    # 2) Save authors meta CSV
    meta_df = pd.DataFrame({'author': global_authors})
    for yr in years:
        meta_df[str(yr)] = meta_df['author'].isin(presence[yr]).astype(int)
    meta_df.to_csv(AUTHORS_META_FILE, index=False)
    print(f"Saved authors union to {AUTHORS_META_FILE}")

    # 3) Remap each graph to the global node set
    for yr in years:
        inp = os.path.join(GRAPH_DIR, f'graph_{yr}.json')
        with open(inp, 'r') as f:
            g = json.load(f)

        old_u2i = g['user_to_idx']         # { author: old_idx }
        idx2user = {idx: user for user, idx in old_u2i.items()}

        src_old, dst_old = g['edge_index']
        src_new = [ global_u2i[idx2user[u]] for u in src_old ]
        dst_new = [ global_u2i[idx2user[v]] for v in dst_old ]

        new_g = {
            'user_to_idx': global_u2i,
            'edge_index': [src_new, dst_new],
            'edge_weight': g['edge_weight']
        }

        outp = os.path.join(OUT_GRAPH_DIR, f'graph_{yr}_union.json')
        with open(outp, 'w') as f:
            json.dump(new_g, f)
        print(f"Year {yr}: wrote unioned graph → {outp}")

    # 4) Reindex & L2‐normalize each feature CSV
    prev_feat = None
    for yr in years:
        df = pd.read_csv(os.path.join(FEATURES_DIR, f'user_counts_{yr}.csv')) \
               .set_index('author')
        cols = df.columns.tolist()
        df2 = df.reindex(global_authors, fill_value=0)
        arr = df2[cols].values.astype(float)

        # compute per-row L2-normalized vector for those with activity
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        # avoid div0
        active = norms[:,0] > 0
        norms[norms == 0] = 1.0
        arr_norm = arr / norms

        # LOCF with decay for rows with no activity
        if prev_feat is None:
            feat = arr_norm
        else:
            feat = arr_norm.copy()
            miss = ~active
            feat[miss] = prev_feat[miss] * args.decay

        outp = os.path.join(OUT_FEATURES_DIR, f'features_{yr}_union.npy')
        np.save(outp, feat)
        print(f"Year {yr}: features → {outp}  shape={feat.shape}")

        prev_feat = feat


if __name__ == '__main__':
    ensure_dirs()
    main()
