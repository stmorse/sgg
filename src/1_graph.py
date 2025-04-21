#!/usr/bin/env python3
"""
Summary:
Create a graph of coincident commenting activity over arbitrary periods of months.
  
- Loops over windows of length `period` months between start_year/start_month and end_year/end_month.
- For each window, aggregates co-reply edges within that span.

Outputs one JSON per window named:
  graph_{y0}-{m0:02d}_{y1}-{m1:02d}.json
"""

import argparse
import bz2
import json
import os

import pandas as pd
from utils import iterate_months 

METAPATH   = '/sciclone/geograd/stmorse/reddit/metadata'
SAVEPATH   = '/sciclone/geograd/stmorse/reddit/graphs'
SAVEPATH_SR= '/sciclone/geograd/stmorse/reddit/subreddit'

def month_index(year, month):
    return year * 12 + (month - 1)

def idx_to_ym(idx):
    return idx // 12, (idx % 12) + 1

def flush_mapping(mapping, cur_year, cur_month):
    """Keep comments from the current and immediate previous month."""
    retain_year, retain_month = (
        (cur_year, cur_month - 1) if cur_month > 1 
        else (cur_year - 1, 12)
    )
    to_remove = [
        cid for cid, (_, _, y, m) in mapping.items() 
        if (y, m) < (retain_year, retain_month)
    ]
    for cid in to_remove:
        del mapping[cid]
    return len(to_remove)

def process_file(filepath, cur_year, cur_month, global_mapping, edges, subreddit=None):
    try:
        df = pd.read_csv(filepath, compression='gzip')
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return
    for _, row in df.iterrows():
        if subreddit and row.get('subreddit') != subreddit:
            continue
        author = row.get('author')
        if not author or author == "[deleted]":
            continue
        cid = row.get('id')
        parent = row.get('parent_id',"")
        # direct reply
        if parent.startswith("t1_"):
            pid = parent[3:]
            pe = global_mapping.get(pid)
            if pe:
                p_author, p_parent, _, _ = pe
                if p_author not in (None, "[deleted]", author):
                    key = tuple(sorted((author, p_author)))
                    edges[key] = edges.get(key, 0) + 1
                # grandparent
                if p_parent.startswith("t1_"):
                    gp = p_parent[3:]
                    gpe = global_mapping.get(gp)
                    if gpe:
                        gp_author = gpe[0]
                        if gp_author not in (None, "[deleted]", author):
                            key = tuple(sorted((author, gp_author)))
                            edges[key] = edges.get(key, 0) + 1
        # record this comment
        global_mapping[cid] = (author, parent, cur_year, cur_month)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start_year",  type=int, required=True)
    p.add_argument("--start_month", type=int, default=1, required=False)
    p.add_argument("--end_year",    type=int, required=True)
    p.add_argument("--end_month",   type=int, default=12, required=False)
    p.add_argument("--period",      type=int, required=True)
    p.add_argument("--subreddit",   type=str, default=None)
    args = p.parse_args()

    # compute flat indices
    idx0 = month_index(args.start_year, args.start_month)
    idx_end = month_index(args.end_year, args.end_month)

    windows = []
    i = idx0
    while i <= idx_end:
        j = min(i + args.period - 1, idx_end)
        y0, m0 = idx_to_ym(i)
        y1, m1 = idx_to_ym(j)
        windows.append((y0, m0, y1, m1))
        i += args.period

    for y0, m0, y1, m1 in windows:
        # output path
        fname = f"graph_{y0}-{m0:02d}_{y1}-{m1:02d}.json"
        out_dir = SAVEPATH_SR if args.subreddit else SAVEPATH
        if args.subreddit:
            out_dir = os.path.join(out_dir, args.subreddit, f"graphs")
        os.makedirs(out_dir, exist_ok=True)
        outpath = os.path.join(out_dir, fname)
        print(f"Building graph for {y0}-{m0:02d} → {y1}-{m1:02d}, saving to {outpath}")

        edges = {}
        global_mapping = {}

        # iterate through every month in this window
        for (yr, mo) in iterate_months(y0, m0, y1, m1):
            f = f"metadata_{yr}-{mo:02d}.csv"
            path = os.path.join(METAPATH, f)
            if not os.path.exists(path):
                print(f"  Missing {path}, skipping")
                continue
            print(f"  Processing {yr}-{mo:02d}")
            process_file(path, yr, mo, global_mapping, edges, args.subreddit)
            flushed = flush_mapping(global_mapping, yr, mo)
            print(f"    mapping size {len(global_mapping)}, edges {len(edges)}, flushed {flushed}")

        # convert to indices
        users = sorted({u for u,v in edges} | {v for u,v in edges})
        u2i = {u:i for i,u in enumerate(users)}
        src, dst, wts = [], [], []
        for (u,v), w in edges.items():
            src.append(u2i[u])
            dst.append(u2i[v])
            wts.append(w)

        out = {"user_to_idx": u2i, "edge_index":[src,dst], "edge_weight":wts}
        with open(outpath, "w") as fp:
            json.dump(out, fp)
        print(f"  Saved: {len(users)} nodes, {len(wts)} edges\n")

if __name__ == "__main__":
    main()
