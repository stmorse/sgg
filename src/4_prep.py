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
    args = p.parse_args()

    years = list(range(args.start_year, args.end_year + 1))
    presence = {}  # for the meta–CSV

    # 1) Gather authors with observed participation (from feature CSVs only)
    for yr in years:
        feat_csv = os.path.join(FEATURES_DIR, f'user_counts_{yr}.csv')
        df_feat = pd.read_csv(feat_csv, usecols=['author'])
        presence[yr] = set(df_feat['author'])

    # 2) Compute intersection of authors present in all years
    global_authors = sorted(set.intersection(*presence.values()))
    global_u2i = {u: i for i, u in enumerate(global_authors)}

    # 3) Save authors meta CSV
    meta_df = pd.DataFrame({'author': global_authors})
    for yr in years:
        meta_df[str(yr)] = meta_df['author'].isin(presence[yr]).astype(int)
    meta_df.to_csv(AUTHORS_META_FILE, index=False)
    print(f"Saved authors intersection to {AUTHORS_META_FILE}")

    # 4) Remap each graph to the global node set
    for yr in years:
        inp = os.path.join(GRAPH_DIR, f'graph_{yr}.json')
        with open(inp, 'r') as f:
            g = json.load(f)

        old_u2i = g['user_to_idx']         # { author: old_idx }
        idx2user = {idx: user for user, idx in old_u2i.items()}

        src_old, dst_old = g['edge_index']
        filtered_edges = [(u, v) for u, v in zip(src_old, dst_old)
                          if idx2user[u] in global_u2i and idx2user[v] in global_u2i]
        src_new = [global_u2i[idx2user[u]] for u, v in filtered_edges]
        dst_new = [global_u2i[idx2user[v]] for u, v in filtered_edges]

        new_g = {
            'user_to_idx': global_u2i,
            'edge_index': [src_new, dst_new],
            'edge_weight': [g['edge_weight'][i] for i, (u, v) in enumerate(zip(src_old, dst_old))
                            if idx2user[u] in global_u2i and idx2user[v] in global_u2i]
        }

        outp = os.path.join(OUT_GRAPH_DIR, f'graph_{yr}_union.json')
        with open(outp, 'w') as f:
            json.dump(new_g, f)
        print(f"Year {yr}: wrote filtered graph → {outp}")

    # 5) Reindex & L2‐normalize each feature CSV
    for yr in years:
        df = pd.read_csv(os.path.join(FEATURES_DIR, f'user_counts_{yr}.csv')) \
               .set_index('author')
        cols = df.columns.tolist()
        df2 = df.reindex(global_authors, fill_value=0)
        arr = df2[cols].values.astype(float)

        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        feat = arr / norms

        outp = os.path.join(OUT_FEATURES_DIR, f'features_{yr}_union.npy')
        np.save(outp, feat)
        print(f"Year {yr}: features → {outp}  shape={feat.shape}")


if __name__ == '__main__':
    ensure_dirs()
    main()
