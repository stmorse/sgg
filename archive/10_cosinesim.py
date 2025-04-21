import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

basepath = '/sciclone/geograd/stmorse/reddit/subreddit/science/filtered'

# def evaluate_cosine_graph(f0, graph_json):
#     """
#     Given features (f0) and co-reply graph_json,
#     computes cosine similarities as predicted edges
#     and evaluates against the actual edges (binary labels).
#     """
#     n = f0.shape[0]

#     print(f'Features shape: {f0.shape}')

#     # Compute cosine similarity matrix (n x n)
#     print('Computing cosine similarity...')
#     sim_mat = cosine_similarity(f0)
    
#     # Load actual edges from graph_json
#     print('Loading edges...')
#     edge_idx = np.array(graph_json['edge_index'])
#     actual_edges = set(zip(edge_idx[0], edge_idx[1]))

#     # Create labels: 1 if edge exists, 0 if not
#     print('Creating labels...')
#     labels, preds = [], []
#     for i in range(n):
#         if i % 1000 == 0:
#             print(f'> {i}')
#         for j in range(i+1, n):
#             labels.append(1 if (i, j) in actual_edges or (j, i) in actual_edges else 0)
#             preds.append(sim_mat[i, j])

#     labels = np.array(labels)
#     preds = np.array(preds)

#     # Evaluate using AUC
#     print(f'Computing AUC...\n')
#     auc = roc_auc_score(labels, preds)
#     print(f"Cosine similarity prediction AUC: {auc:.4f}")

#     return auc

# def evaluate_cosine_graph(f0, graph_json, neg_sample_ratio=1.0, random_state=42):
#     """
#     Compute cosine similarities for actual graph edges plus sampled negatives,
#     evaluating via ROC-AUC without full n² memory usage.

#     neg_sample_ratio: ratio of negative (non-edge) samples to positive edges.
#     """
#     np.random.seed(random_state)
#     n = f0.shape[0]

#     edge_idx = np.array(graph_json['edge_index'])
#     actual_edges = set(zip(edge_idx[0], edge_idx[1]))

#     num_pos = len(actual_edges)
#     num_neg = int(num_pos * neg_sample_ratio)

#     # Positive samples
#     pos_pairs = np.array(list(actual_edges))

#     # Negative samples: random pairs not in edges
#     neg_pairs = set()
#     while len(neg_pairs) < num_neg:
#         i = np.random.randint(0, n, size=num_neg * 2)
#         j = np.random.randint(0, n, size=num_neg * 2)
#         pairs = {(a,b) for a,b in zip(i,j) if a < b and (a,b) not in actual_edges and (b,a) not in actual_edges}
#         neg_pairs.update(pairs)
#         if len(neg_pairs) > num_neg:
#             neg_pairs = set(list(neg_pairs)[:num_neg])

#     neg_pairs = np.array(list(neg_pairs))

#     # Compute cosine similarities just for these pairs
#     def cos_sim(u, v):
#         return np.sum(u*v, axis=1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-10)

#     pos_sim = cos_sim(f0[pos_pairs[:,0]], f0[pos_pairs[:,1]])
#     neg_sim = cos_sim(f0[neg_pairs[:,0]], f0[neg_pairs[:,1]])

#     labels = np.concatenate([np.ones(len(pos_sim)), np.zeros(len(neg_sim))])
#     preds = np.concatenate([pos_sim, neg_sim])

#     auc = roc_auc_score(labels, preds)
#     print(f"Cosine similarity prediction AUC: {auc:.4f}")

#     return auc

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def cosine_auc_filtered(features_csv, graph_json, min_posts=10, neg_sample_ratio=1.0, random_state=42):
    np.random.seed(random_state)

    # Load CSV
    df = pd.read_csv(features_csv)
    df['total_posts'] = df.drop(columns=['author']).sum(axis=1)
    df = df[df['total_posts'] >= min_posts]
    active_authors = set(df['author'])
    print(f'Active authors after {min_posts} filter: {len(active_authors)}')

    # Load Graph JSON
    with open(graph_json) as f:
        graph = json.load(f)

    user_to_idx = graph['user_to_idx']

    # Find intersection of active authors
    active_users = [u for u in active_authors if u in user_to_idx]
    print(f'Active users after filtering: {len(active_users)}')
    if len(active_users) < 2:
        print("Not enough active users after filtering.")
        return None

    active_idxs = np.array([user_to_idx[u] for u in active_users])

    # Extract features and normalize
    features = df.set_index('author').loc[active_users].values
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    features_norm = features / norms

    # Build mapping from old idx to filtered idx
    old_to_new = {old: new for new, old in enumerate(active_idxs)}

    # Filter edges
    edge_idx = np.array(graph['edge_index'])
    edges = [(u,v) for u,v in zip(*edge_idx) if u in active_idxs and v in active_idxs]
    edges_filtered = [(old_to_new[u], old_to_new[v]) for u,v in edges]

    num_pos = len(edges_filtered)
    print(f'Num edges after filter: {num_pos}')
    if num_pos == 0:
        print("No edges left after filtering.")
        return None

    # Sample negative edges
    num_neg = int(num_pos * neg_sample_ratio)
    n = len(active_users)
    pos_set = set(edges_filtered)
    neg_set = set()
    attempts = 0
    max_attempts = num_neg * 10
    while len(neg_set) < num_neg and attempts < max_attempts:
        i, j = np.random.randint(0, n, size=2)
        if i != j and (i,j) not in pos_set and (j,i) not in pos_set:
            neg_set.add((i,j))
        attempts += 1

    print(f'Neg set: {len(neg_set)}')
    if len(neg_set) < num_neg:
        print("Not enough negative samples found.")
        return None

    pos_pairs = np.array(list(pos_set))
    neg_pairs = np.array(list(neg_set))

    # Cosine similarity function
    def cos_sim(u, v):
        return np.sum(u*v, axis=1)

    pos_sim = cos_sim(features_norm[pos_pairs[:,0]], features_norm[pos_pairs[:,1]])
    neg_sim = cos_sim(features_norm[neg_pairs[:,0]], features_norm[neg_pairs[:,1]])

    labels = np.concatenate([np.ones(len(pos_sim)), np.zeros(len(neg_sim))])
    preds = np.concatenate([pos_sim, neg_sim])

    auc = roc_auc_score(labels, preds)
    print(f"Filtered cosine similarity AUC: {auc:.4f}")

    return auc


f0 = np.load(os.path.join(basepath, 'features_2007_union.npy'))
with open("/sciclone/geograd/stmorse/reddit/subreddit/science/filtered/graph_2007_union.json") as f:
    graph = json.load(f)
graph.keys()

aucs = []
for mp in np.arange(0, 105, 5):
    print(f'\n\n----\nMin posts: {mp}\n----')
    auc = cosine_auc_filtered(
        "/sciclone/geograd/stmorse/reddit/subreddit/science/users/user_counts_2007.csv",
        "/sciclone/geograd/stmorse/reddit/subreddit/science/links/graph_2007.json",
        min_posts=mp, 
        neg_sample_ratio=1.0)
    aucs.append(auc)

print(f'\n\nALL AUC: {aucs}')