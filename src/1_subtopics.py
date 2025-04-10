"""
For a given date range,
- pulls and saves all metadata ---> metadata
- find topic cluster for a specific subreddit ---> subreddit/<name>/<labels / models>
- finds tf-idf keywords for those clusters ---> subreddit/<name>/tfidf
"""

import argparse
import configparser
import gc
import heapq        # for top-k cluster members
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

import utils

def find_topics(
    data_path: str,
    embed_path: str,
    meta_path: str,
    label_path: str,
    model_path: str,
    tfidf_path: str,
    subreddit: str,
    n_clusters: int,
    start_year: int,
    end_year: int,
    start_month: int,
    end_month: int,
    top_k: int,
    top_m: int,
    max_df: float
):
    """
    1. load metadata of subreddit
    2. load embeddings of subreddit
    3. cluster embeddings
    4. save labels
    5. save top-k words
    6. compute tf-idf
    """  

    t0 = time.time()
    years = range(start_year, end_year+1)
    months = range(start_month, end_month+1)
    
    print(f'CPU count  : {os.cpu_count()}')
    print(f'Subreddit  : {subreddit}')
    print(f'Range      : {start_year}-{start_month} to {end_year}-{end_month}\n')

    # ------
    # First pass: train k-means model on subreddit
    # ------

    # initialize model
    model = MiniBatchKMeans(n_clusters=n_clusters)

    # save subreddit indices so we don't have to load metadata on second pass
    sr_idx = {}

    for year, month in [(yr, mo) for yr in years for mo in months]:
        print(f'Reading {year}-{month:02} ... ({time.time()-t0:.3f})')
        
        # load metadata for this subreddit
        print(f'> Loading metadata ... ({time.time()-t0:.3f})')
        metadata = pd.read_csv(
            os.path.join(meta_path, f'metadata_{year}-{month:02}.csv'),
            compression='gzip',
        )
        metadata = metadata[metadata['subreddit'] == subreddit]
        sr_idx[(year, month)] = metadata['idx'].values
        print(f'> Found {len(sr_idx[(year, month)])} entries ... ')

        # load embeddings for this month
        # TODO: adjust for zarr
        print(f'> Loading embeddings ... ({time.time()-t0:.3f})')
        embeddings = utils.load_embeddings(year, month, embed_path=embed_path)
        embeddings = embeddings[sr_idx[(year, month)]]

        # cluster embeddings
        print(f'> Fitting model ... ({time.time()-t0:.3f})')
        model.partial_fit(embeddings)

    # we now have a trained model
    # save cluster centers
    cluster_centers = model.cluster_centers_
    with open(os.path.join(model_path, f'cc_{start_year}_{end_year}.npz'), 'wb') as f:
        np.savez_compressed(f, cc=cluster_centers, allow_pickle=False)

    # ------
    # Second pass: save labels and top-k sentences
    # ------

    print(f'\nModel training complete. Labeling and computing tf-idf ... ({time.time()-t0:.3f})\n')

    # initialize corpus
    # this will store a list of the top-k sentences for each cluster
    corpus = [[] for _ in range(n_clusters)]

    # this will store top-k indices for each cluster
    top_k_indices = {i: np.array([]) for i in range(n_clusters)}

    # TODO: currently this grabs top-k for each month,
    #       but we should grab top-k over the entire date range

    for year, month in [(yr, mo) for yr in years for mo in months]:
        print(f'Reading {year}-{month:02} ... ({time.time()-t0:.3f})')

        # load embeddings for this month
        # TODO: adjust for zarr
        print(f'> Loading embeddings ... ({time.time()-t0:.3f})')
        embeddings = utils.load_embeddings(year, month, embed_path=embed_path)
        embeddings = embeddings[sr_idx[(year, month)]]

        # predict labels
        print(f'> Labeling ... ({time.time()-t0:.3f})')
        labels = model.predict(embeddings)

        # save labels
        with open(os.path.join(label_path, f'labels_{year}-{month:02}.npz'), 'wb') as f:
            np.savez_compressed(f, labels=labels, allow_pickle=False)
        
        # grab top-k embeddings for each cluster
        # NOTE: these are indices within the subreddit
        print(f'> Finding top-k comments ... ({time.time()-t0:.3f})')
        for i in range(n_clusters):
            dist_c = np.linalg.norm(embeddings - cluster_centers[i], axis=1)
            top_k_indices[i] = np.concatenate((
                top_k_indices[i], 
                np.argpartition(dist_c, top_k)[:top_k]
            ))

        # build corpus
        print(f'> Building corpus ... ({time.time()-t0:.3f})')
        reader = utils.read_file(
            year, month, 
            return_type='sentences', 
            data_path=data_path,
            chunk_size=10000,
        )
        k = 0
        for chunk in reader:
            for s in chunk:
                # check if this sentence is in the subreddit
                # NOTE: `k` is the month-level index, `k_idx` is the sr-level index
                k_idx = np.where(sr_idx[(year, month)] == k)[0]
                if k_idx.shape[0] > 0:
                    k_idx = k_idx[0]
                    c = int(labels[k_idx])
                    if k_idx in top_k_indices[c]:
                        corpus[c].append(s)
                k += 1  # increment regardless

    # complete with second pass

    # collapse corpus into a format for tf-idf vectorizer
    for i in range(n_clusters):
        corpus[i] = ' --- '.join(corpus[i])

    print(f'Computing tf-idf ... ({time.time()-t0:.3f})')
    vectorizer = TfidfVectorizer(
        input='content',
        max_df=max_df,
        # max_features=100,
        use_idf=True,
        smooth_idf=True
    )

    X = vectorizer.fit_transform(corpus)

    print(f'> Extracting top {top_m} keywords ... ({time.time()-t0:.3f})')
    keywords = []
    for i in range(n_clusters):
        max_idx = np.argsort(X[i,:].toarray().flatten())[::-1][:top_m]
        keyword = vectorizer.get_feature_names_out()[max_idx]
        keywords.append(keyword)

    print(f'> Saving output ... ({time.time()-t0:.3f})')
    output = {
        'range': [(start_year, start_month), (end_year, end_month)],
        'full': {
            'scores': X,
            'feature_names': vectorizer.get_feature_names_out()
        },
        'tfidf': {
            i: {
                'sample_indices': top_k_indices[i],
                'keywords': keywords[i],
            } for i in range(n_clusters)
        }
    }

    with open(os.path.join(tfidf_path, f'tfidf_{start_year}-{end_year}.pkl'), 'wb') as f:
        pickle.dump(output, f)

    print('Garbage collection ...')
    gc.collect()

    print(f'Complete. ({time.time()-t0:.2f})')


if __name__=="__main__":
    config = configparser.ConfigParser()
    config.read('../config.ini')
    g = config['general']

    parser = argparse.ArgumentParser()
    parser.add_argument('--subpath', type=str, required=True)
    parser.add_argument('--subreddit', type=str, required=True)
    parser.add_argument('--start-year', type=int, required=True)
    parser.add_argument('--end-year', type=int, required=True)
    parser.add_argument('--start-month', type=int, default=1, required=False)
    parser.add_argument('--end-month', type=int, default=12, required=False)
    parser.add_argument('--n-clusters', type=int, required=True)
    parser.add_argument('--top-k', type=int, default=100, required=False)
    parser.add_argument('--top-m', type=int, default=20, required=False)
    parser.add_argument('--max-df', type=float, default=0.3, required=False)
    args = parser.parse_args()

    subpath = os.path.join(g['save_path'], args.subpath)
    
    for subdir in ['labels', 'models', 'tfidf']:
        if not os.path.exists(os.path.join(subpath, subdir)):
            os.makedirs(os.path.join(subpath, subdir))
    
    find_topics(
        data_path=g['data_path'],
        embed_path=g['embed_path'],
        meta_path=g['meta_path'],
        label_path=os.path.join(subpath, 'labels'),
        model_path=os.path.join(subpath, 'models'),
        tfidf_path=os.path.join(subpath, 'tfidf'),
        subreddit=args.subreddit,
        n_clusters=args.n_clusters,
        start_year=args.start_year,
        end_year=args.end_year,
        start_month=args.start_month,
        end_month=args.end_month,
        top_k=args.top_k,
        top_m=args.top_m,
        max_df=args.max_df
    )