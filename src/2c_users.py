import argparse
import json
import os

import numpy as np
import pandas as pd

GRAPH_PATH = "/sciclone/geograd/stmorse/reddit/subreddit/science/links/graph_2007-2007.json"
FEAT0_PATH = "/sciclone/geograd/stmorse/reddit/subreddit/science/users/user_label_counts_2007.csv"
FEAT1_PATH = "/sciclone/geograd/stmorse/reddit/subreddit/science/users/user_label_counts_2008.csv"

def load_and_threshold_graph(path, threshold):
    with open(path) as f:
        g = json.load(f)
    u2i = g['user_to_idx']
    ei = np.array(g['edge_index'], dtype=int)  # [2, E]
    ew = np.array(g['edge_weight'], dtype=float)
    mask = ew >= threshold
    return u2i, ei[:, mask], ew[mask]


def load_features_csv(path):
    df = pd.read_csv(path)
    authors = set(df['author'])
    return df, authors


def build_final_graph(u2i_thresh, ei_thresh, ew_thresh, authors_final, graph_path):
    # remap authors → new idx
    authors_sorted = sorted(authors_final)
    new_u2i = {u: i for i, u in enumerate(authors_sorted)}
    # filter & remap edges
    src, dst, w = [], [], []
    idx2u = {v: k for k, v in u2i_thresh.items()}
    for (i, j), weight in zip(ei_thresh.T.tolist(), ew_thresh.tolist()):
        u, v = idx2u[i], idx2u[j]
        if u in new_u2i and v in new_u2i:
            src.append(new_u2i[u])
            dst.append(new_u2i[v])
            w.append(weight)
    out = {
        'user_to_idx': new_u2i,
        'edge_index': [src, dst],
        'edge_weight': w
    }
    base, ext = os.path.splitext(graph_path)
    out_file = f"{base}_filtered{ext}"
    with open(out_file, 'w') as f:
        json.dump(out, f)
    return new_u2i, out_file


def save_aligned_features(df, users_map, feat_path):
    # keep only final users
    df = df[df['author'].isin(users_map)]
    # order rows so row i matches node idx i
    ordered = sorted(users_map, key=lambda u: users_map[u])
    df = df.set_index('author').loc[ordered]
    # feats = df.drop(columns=['author']).values
    feats = df.values.astype('float64')

    l2_norms = np.linalg.norm(feats, ord=2, axis=1, keepdims=True)
    feats = feats / l2_norms
    
    base, ext = os.path.splitext(feat_path)
    out_file = f"{base}_filtered{ext}"
    np.save(out_file, feats)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--graph_json',       required=False, default=GRAPH_PATH)
    p.add_argument('--threshold',        type=float, required=True)
    p.add_argument('--features_t0_csv',  required=False, default=FEAT0_PATH)
    p.add_argument('--features_t1_csv',  required=False, default=FEAT1_PATH)
    args = p.parse_args()

    # 1) threshold graph
    u2i_thr, ei_thr, ew_thr = load_and_threshold_graph(args.graph_json, args.threshold)

    # 2) load CSVs
    df0, auth0 = load_features_csv(args.features_t0_csv)
    df1, auth1 = load_features_csv(args.features_t1_csv)

    # 3) intersect users: in graph & both CSVs
    final_authors = set(u2i_thr) & auth0 & auth1

    # 4) build & save filtered graph
    users_map, graph_file = build_final_graph(u2i_thr, ei_thr, ew_thr, final_authors, args.graph_json)
    print(f"Filtered graph → {graph_file} ({len(users_map)} nodes)")

    # 5) filter & save features aligned to graph idx
    save_aligned_features(df0, users_map, args.features_t0_csv)
    save_aligned_features(df1, users_map, args.features_t1_csv)


if __name__ == '__main__':
    main()

