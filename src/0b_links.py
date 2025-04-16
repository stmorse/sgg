"""
Create a graph of coincident commenting activity over a date range.

The `parent_id` field is structured "tx_yyyy". 
- "x=3" indicates a top-level comment
- "x=1" indicates a reply to a parent given by "yyyy".  
- This id ("yyyy") corresponds to the `id` field.

Note: we are only adding an edge for
- comments within two replies of each other (parent/grandparent)
- replies that happen within two months of each other ("flush_mapping")
"""

import argparse
import bz2
import json
import os
from collections import Counter

DATAPATH = '/sciclone/data10/twford/reddit/reddit/comments'
OUTPATH  = '/sciclone/geograd/stmorse/reddit/links'

def iterate_months(s_year, s_month, e_year, e_month):
    year, month = s_year, s_month
    while (year < e_year) or (year == e_year and month <= e_month):
        yield year, month
        month += 1
        if month > 12:
            month = 1
            year += 1

def flush_mapping(mapping, cur_year, cur_month):
    """Keep comments from the current and immediate previous month"""
    
    # figure out the retain boundary
    retain_year, retain_month = (
        (cur_year, cur_month - 1) if cur_month > 1 
        else (cur_year - 1, 12)
    )

    # build list of ids to remove
    to_remove = ([
        cid for cid, (author, parent_id, y, m) in mapping.items() 
        if (y, m) < (retain_year, retain_month)
    ])

    # remove from global mapping
    for cid in to_remove:
        del mapping[cid]

    # return how many we removed
    return len(to_remove)

def process_file(filepath, cur_year, cur_month, global_mapping, edges, 
                 subreddit=None):
    with bz2.BZ2File(filepath, "rb") as f:
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            # skip non-subreddit
            if subreddit is not None and data.get("subreddit") != subreddit:
                continue
            
            # skip deleted accounts
            author = data.get("author")
            if not author or author == "[deleted]":
                continue
            cid = data.get("id")
            if not cid:
                continue

            parent_str = data.get("parent_id", "")
            
            # If reply to a comment ("t1_"), try to add edge for direct reply
            if parent_str.startswith("t1_"):
                p_id = parent_str[3:]               # extract just id
                p_entry = global_mapping.get(p_id)  # get info for this id
                
                if p_entry:
                    # if we've seen the parent, add the edge
                    p_author = p_entry[0]
                    if p_author and p_author != "[deleted]" and p_author != author:
                        # sorted to avoid duplicates
                        key = tuple(sorted((author, p_author)))
                        edges[key] = edges.get(key, 0) + 1
                    
                    # also, check for grandparent and repeat
                    p_parent = p_entry[1]
                    if p_parent.startswith("t1_"):
                        gp_id = p_parent[3:]
                        gp_entry = global_mapping.get(gp_id)
                        if gp_entry:
                            gp_author = gp_entry[0]
                            if gp_author and gp_author != "[deleted]" and gp_author != author:
                                key = tuple(sorted((author, gp_author)))
                                edges[key] = edges.get(key, 0) + 1
            
            # Add this comment to global mapping with its originating month info.
            global_mapping[cid] = (author, parent_str, cur_year, cur_month)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_year", type=int, required=True)
    parser.add_argument("--end_year", type=int, required=True)
    parser.add_argument("--start_month", default=1, type=int)
    parser.add_argument("--end_month", default=12, type=int)
    parser.add_argument("--base_dir", type=str, default=DATAPATH)
    parser.add_argument("--out_dir", type=str, default=OUTPATH)
    parser.add_argument("--subreddit", type=str, default=None)
    args = parser.parse_args()

    # ------
    # Build edges from raw data
    # ------

    # (user1, user2) -> weight; sorted tuple ensures undirected pair uniqueness.
    edges = {}  

    # comment_id -> (author, parent_id, year, month)
    global_mapping = {}  

    for year, month in iterate_months(
        args.start_year, args.start_month, args.end_year, args.end_month):
        
        fname = f"RC_{year}-{month:02d}.bz2"
        filepath = os.path.join(args.base_dir, fname)
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}. Skipping.")
            continue

        # build global_mapping and edges for this file
        print(f"Processing {filepath}...")
        process_file(filepath, year, month, global_mapping, edges,
                     args.subreddit)

        # flush entries in global_mapping older than 1 month ago
        flushed = flush_mapping(global_mapping, year, month)
        print(f"Flushed {flushed} old mapping entries.")

    # ------
    # Convert to users, edges, weights
    # ------

    # Build a user index from edge pairs.
    users = set()
    for (u, v) in edges:
        users.add(u)
        users.add(v)
    user_to_idx = {user: idx for idx, user in enumerate(sorted(users))}

    src, dst, weights = [], [], []
    for (u, v), w in edges.items():
        src.append(user_to_idx[u])
        dst.append(user_to_idx[v])
        weights.append(w)

    outdata = {
        "user_to_idx": user_to_idx,
        "edge_index": [src, dst],
        "edge_weight": weights
    }

    fname = f"graph_{args.start_year}-{args.end_year}.json"
    outname = os.path.join(args.out_dir, fname)
    with open(outname, "w") as f:
        json.dump(outdata, f)
    print(
        f"Graph saved to {outname}. Nodes: {len(user_to_idx)}, " 
        f"edges: {len(weights)}.")

if __name__ == "__main__":
    main()
