import argparse
import bz2
import json
import os
from collections import Counter

DATAPATH = '/sciclone/data10/twford/reddit/reddit/comments/RC_2007-01.bz2'
OUTPATH  = '/sciclone/geograd/stmorse/reddit/links'

# Two-level undirected interaction: direct reply & grandparent chain
# Edge keys stored as sorted tuples (u, v) so that each pair is unique

def build_comment_mapping(filepath):
    # First pass: map comment id -> (author, parent_id)
    mapping = {}
    with bz2.BZ2File(filepath, "rb") as f:
        for line in f:
            try:
                data = json.loads(line)
                cid = data.get("id")
                mapping[cid] = (data.get("author"), data.get("parent_id"))
            except json.JSONDecodeError:
                continue
    return mapping

def build_edges(filepath, comment_map):
    edges = Counter()
    with bz2.BZ2File(filepath, "rb") as f:
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            u = data.get("author")
            
            # Skip invalid/deleted user
            if not u or u == "[deleted]":
                continue

            par_full = data.get("parent_id", "")
            if par_full.startswith("t1_"):
                par_id = par_full[3:]
                par_data = comment_map.get(par_id)
                if par_data:
                    p_author = par_data[0]
                    if p_author and p_author != "[deleted]" and p_author != u:
                        key = tuple(sorted((p_author, u)))
                        edges[key] += 1

                    # Grandparent: look up parent's parent if parent's id starts with t1_
                    grand_full = par_data[1]
                    if grand_full and grand_full.startswith("t1_"):
                        gp_id = grand_full[3:]
                        gp_data = comment_map.get(gp_id)
                        if gp_data:
                            gp_author = gp_data[0]
                            if gp_author and gp_author != "[deleted]" and gp_author != u:
                                key = tuple(sorted((gp_author, u)))
                                edges[key] += 1
    return edges

def map_users(edges):
    # Get a sorted list of unique users from edge pairs
    users = set()
    for u, v in edges:
        users.add(u)
        users.add(v)
    return {user: idx for idx, user in enumerate(sorted(users))}

def build_edge_list(edges, user_to_idx):
    src, dst, weights = [], [], []
    for (u, v), w in edges.items():
        src.append(user_to_idx[u])
        dst.append(user_to_idx[v])
        weights.append(w)
    return [src, dst], weights

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", default=DATAPATH, required=False)
    parser.add_argument("--outpath", default=OUTPATH, required=False)
    args = parser.parse_args()
    
    print("Mapping comments...")
    comment_map = build_comment_mapping(args.datapath)
    print(f"Mapped {len(comment_map)} comments.")

    print("Building edges...")
    edges = build_edges(args.datapath, comment_map)
    print(f"Found {len(edges)} unique user pairs.")

    user_to_idx = map_users(edges)
    edge_index, edge_weight = build_edge_list(edges, user_to_idx)
    
    print(f"Graph stats: {len(user_to_idx)} nodes, {len(edge_weight)} edges.")
    
    # Save graph data to JSON (could be adapted to npz or Torch's save format)
    outdata = {
        "user_to_idx": user_to_idx,
        "edge_index": edge_index,
        "edge_weight": edge_weight
    }
    outpath = os.path.join(args.outpath, "graph.json")
    with open(outpath, "w") as f:
        json.dump(outdata, f)
    print("Graph saved to", outpath)

if __name__ == "__main__":
    main()
