#!/usr/bin/env python3
"""
Train edge-level GAE over arbitrary-month windows with
negative sampling & combined loss, using expanding-window
full-retrain CV and final test hold-out.
"""
import os
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, mean_squared_error

from models import EdgeEncoder, EdgeDecoder, FeatureDecoder
from utils import iterate_periods, sample_negative_edges

BASE = '/sciclone/geograd/stmorse/reddit/subreddit/science/filtered2'

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    # -----
    # BUILD DATA
    # -----

    # build window-pairs
    wins  = [w for w in iterate_periods(
        args.start_year, args.start_month,
        args.end_year,   args.end_month,
        args.period
    )]
    pairs = list(zip(wins, wins[1:]))
    
    # split train vs test
    train_pairs = pairs[:-args.num_test]
    test_pairs  = pairs[-args.num_test:]

    # preload train data
    # TODO: this is hacky, we can't do it this way with more data
    train_data = []
    for (y0,m0,y1,m1), (yn0,mn0,yn1,mn1) in train_pairs:
        tag0 = f"{y0}-{m0:02d}_{y1}-{m1:02d}"
        tag1 = f"{yn0}-{mn0:02d}_{yn1}-{mn1:02d}"

        g = json.load(open(os.path.join(BASE, f"graph_{tag0}_filtered.json")))
        ei_pos = torch.tensor(g['edge_index'], dtype=torch.long, device=device)
        ew = torch.tensor(g['edge_weight'], dtype=torch.float, device=device)
        pos_set = set(zip(ei_pos[0].tolist(), ei_pos[1].tolist()))
        n_pos   = ei_pos.size(1)
        x_t  = torch.tensor(
            np.load(os.path.join(BASE, f"features_{tag0}_filtered.npy")),
            dtype=torch.float, device=device)
        x_tp = torch.tensor(
            np.load(os.path.join(BASE, f"features_{tag1}_filtered.npy")),
            dtype=torch.float, device=device)
        
        train_data.append({
            'tag0': tag0, 'tag1': tag1,
            'ei_pos': ei_pos, 'ew': ew, 'pos_set': pos_set, 'n_pos': n_pos,
            'x_t': x_t, 'x_tp': x_tp
        })

    N, dim = train_data[0]['x_t'].shape

    # -----
    # TRAIN MODEL
    # -----

    # do we need this?
    encoder_final = edge_dec_final = feat_dec_final = None

    # do full retrain on an expanding set of training data
    # reporting validation on final window  
    # do M forward steps  
    for k in range(2, len(train_data)+1):
        subset = train_data[:k]
        last_tag1 = subset[-1]['tag1']
        print(f"\n=== TRAIN on t0->{last_tag1} ({k} windows) ===")

        # init fresh models
        enc = EdgeEncoder(in_channels=dim, hidden_channels=64,
                          edge_latent_dim=args.latent_dim).to(device)
        ed  = EdgeDecoder(edge_latent_dim=args.latent_dim).to(device)
        fd  = FeatureDecoder(edge_latent_dim=args.latent_dim, out_features=dim).to(device)
        
        opt = torch.optim.Adam(
            list(enc.parameters())+list(ed.parameters())+list(fd.parameters()), 
            lr=args.lr)
        crit_e = torch.nn.BCELoss()
        crit_f = torch.nn.MSELoss()

        # iterate epochs
        for ep in range(1, args.epochs+1):
            enc.train()
            ed.train()
            fd.train()
            opt.zero_grad()

            # iterate over windows in training set
            for d in subset[:-1]:
                # node encoding
                h = F.relu(enc.gcn1(d['x_t'], d['ei_pos'], d['ew']))
                h = F.relu(enc.gcn2(h,        d['ei_pos'], d['ew']))

                # pos/neg edge embeddings
                z_pos = enc.edge_mlp(torch.cat([h[d['ei_pos'][0]],
                                                h[d['ei_pos'][1]]], dim=1))
                neg_ei = sample_negative_edges(
                    N, d['pos_set'],
                    int(d['n_pos'] * args.neg_ratio),
                    device
                )
                z_neg = enc.edge_mlp(torch.cat([h[neg_ei[0]],
                                                h[neg_ei[1]]], dim=1))
                
                # concatenate edge embeddings, pos and negative
                # build label tensor of [111...000...] for [pos, neg]
                z_all = torch.cat([z_pos, z_neg], dim=0)
                labels = torch.cat([
                    torch.ones(d['n_pos'], device=device),
                    torch.zeros(z_neg.size(0), device=device)
                ], dim=0)

                # predict edges with edge decoder
                preds  = ed(z_all)

                # split for edge loss
                # TODO: this is holding out validation even for non-final windows
                # tr_m, _ = split_masks(preds.size(0), args.train_ratio, args.val_ratio)
                # loss_e = crit_e(preds[tr_m], labels[tr_m])

                # edge loss
                loss_e = crit_e(preds, labels)
                
                # feature loss (only use pos edges, reconstruct full features)
                feat_pred = fd(z_pos, d['ei_pos'], N)
                loss_f = crit_f(feat_pred, d['x_tp'])
                
                # combined
                loss = (args.alpha * loss_f + (1-args.alpha) * loss_e)
                # loss = loss_f + loss_e
                loss.backward()
            
            opt.step()

        # evaluate on last window only
        enc.eval()
        ed.eval() 
        fd.eval()
        with torch.no_grad():
            d = subset[-1]

            # encode
            h = F.relu(enc.gcn1(d['x_t'], d['ei_pos'], d['ew']))
            h = F.relu(enc.gcn2(h,        d['ei_pos'], d['ew']))
            z_pos = enc.edge_mlp(torch.cat([h[d['ei_pos'][0]],
                                            h[d['ei_pos'][1]]], dim=1))
            neg_ei = sample_negative_edges(
                N, d['pos_set'],
                int(d['n_pos'] * args.neg_ratio),
                device
            )
            z_neg = enc.edge_mlp(torch.cat([h[neg_ei[0]],
                                            h[neg_ei[1]]], dim=1))
            z_all = torch.cat([z_pos, z_neg], dim=0)
            labels = torch.cat([
                torch.ones(d['n_pos'], device=device),
                torch.zeros(z_neg.size(0), device=device)
            ], dim=0)
            preds = ed(z_all)
            
            # split
            # tr_m, va_m = split_masks(preds.size(0), args.train_ratio, args.val_ratio)
            
            # score edge predictions
            # auc_tr = roc_auc_score(labels[tr_m].cpu(), preds[tr_m].cpu())
            # auc_va = roc_auc_score(labels[va_m].cpu(), preds[va_m].cpu())
            auc_va = roc_auc_score(labels.cpu(), preds.cpu())
            
            # feature MSE
            feat_pred = fd(z_pos, d['ei_pos'], N)
            mse_f = mean_squared_error(
                d['x_tp'].cpu(), feat_pred.cpu()
            )

        print(f"[{d['tag0']}->{d['tag1']}] "
            #   f"Edge AUC Train={auc_tr:.4f}, Val={auc_va:.4f} | "
            f"Edge AUC={auc_va:.4f} | "
            f"Feat MSE={mse_f:.4f}")

        # keep final models
        if k == len(train_data):
            encoder_final = enc
            edge_dec_final = ed
            feat_dec_final = fd

    model_path = os.path.join(BASE, f"final_model_{last_tag1}.pth")
    torch.save({
        'enc': encoder_final.state_dict(),
        'ed': edge_dec_final.state_dict(),
        'fd': feat_dec_final.state_dict(),
        'args': vars(args)
    }, model_path)
    print(f"Models saved to {model_path}")

    # final test-set evaluation
    print("\n=== TEST SET EVALUATION ===")
    encoder_final.eval()
    edge_dec_final.eval()
    feat_dec_final.eval()
    for (y0,m0,y1,m1), (yn0,mn0,yn1,mn1) in test_pairs:
        tag0 = f"{y0}-{m0:02d}_{y1}-{m1:02d}"
        tag1 = f"{yn0}-{mn0:02d}_{yn1}-{mn1:02d}"
        
        # load test window
        g = json.load(open(os.path.join(BASE, f"graph_{tag0}_filtered.json")))
        ei_pos = torch.tensor(g['edge_index'], dtype=torch.long, device=device)
        ew = torch.tensor(g['edge_weight'], dtype=torch.float, device=device)
        pos_set = set(zip(ei_pos[0].tolist(), ei_pos[1].tolist()))
        n_pos   = ei_pos.size(1)
        x_t  = torch.tensor(np.load(os.path.join(BASE,
                      f"features_{tag0}_filtered.npy")),
                      dtype=torch.float, device=device)
        x_tp = torch.tensor(np.load(os.path.join(BASE,
                      f"features_{tag1}_filtered.npy")),
                      dtype=torch.float, device=device)
        
        with torch.no_grad():
            h = F.relu(encoder_final.gcn1(x_t, ei_pos, ew))
            h = F.relu(encoder_final.gcn2(h,   ei_pos, ew))
            z_pos = encoder_final.edge_mlp(torch.cat(
                [h[ei_pos[0]], h[ei_pos[1]]], dim=1))
            neg_ei = sample_negative_edges(
                N, pos_set,
                int(n_pos * args.neg_ratio),
                device
            )
            z_neg = encoder_final.edge_mlp(torch.cat(
                [h[neg_ei[0]], h[neg_ei[1]]], dim=1))
            z_all = torch.cat([z_pos, z_neg], dim=0)
            labels = torch.cat([
                torch.ones(n_pos, device=device),
                torch.zeros(z_neg.size(0), device=device)
            ], dim=0)
            preds = edge_dec_final(z_all)
            
            # score
            auc_test = roc_auc_score(labels.cpu(), preds.cpu())
            
            # feature MSE
            feat_pred = feat_dec_final(z_pos, ei_pos, N)
            mse_test = mean_squared_error(x_tp.cpu(),
                                          feat_pred.cpu())
        
        print(f"[TEST {tag0}->{tag1}] "
              f"Edge AUC Test={auc_test:.4f} | "
              f"Feat MSE={mse_test:.4f}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--start_year',  type=int,   required=True)
    p.add_argument('--start_month', type=int,   default=1)
    p.add_argument('--end_year',    type=int,   required=True)
    p.add_argument('--end_month',   type=int,   default=12)
    p.add_argument('--period',      type=int,   required=True)
    p.add_argument('--epochs',      type=int,   default=100)
    p.add_argument('--lr',          type=float, default=0.005)
    p.add_argument('--train_ratio', type=float, default=0.7)
    p.add_argument('--val_ratio',   type=float, default=0.15)
    p.add_argument('--neg_ratio',   type=float, default=1.0)
    p.add_argument('--alpha',       type=float, default=0.5)
    p.add_argument('--num_test',    type=int,   default=2)
    p.add_argument('--latent_dim',  type=int,   default=32)
    args = p.parse_args()
    
    main(args)