import numpy as np
import pandas as pd
import json
from scipy.spatial.distance import pdist

def main():
    print("Loading cached embeddings for Taxon-level Variance Analysis...")
    embs = np.load('outputs1/cached_embeddings_tf_efficientnetv2_m_in21k.npy')
    tr = pd.read_csv('outputs1/split_hard_constrained/train.csv')
    va = pd.read_csv('outputs1/split_hard_constrained/val.csv')
    te = pd.read_csv('outputs1/split_hard_constrained/test.csv')
    df = pd.concat([tr, va, te], ignore_index=True)

    species_list = sorted(df['label'].unique().tolist())
    stats_per_taxon = []

    for sp in species_list:
        sp_mask = (df['label'] == sp).values
        sp_indices = np.where(sp_mask)[0]
        sp_embs = embs[sp_indices]
        sp_blocks = df.iloc[sp_indices]['subfolder'].values
        
        unique_blocks = np.unique(sp_blocks)
        block_centroids = []
        intra_block_dists = []
        
        for b in unique_blocks:
            b_idx = np.where(sp_blocks == b)[0]
            b_e = sp_embs[b_idx]
            b_centroid = b_e.mean(axis=0)
            block_centroids.append(b_centroid)
            if len(b_idx) > 1:
                d_intra = np.linalg.norm(b_e - b_centroid, axis=1)
                intra_block_dists.extend(d_intra.tolist())
                
        intra_var = float(np.mean(intra_block_dists)) if len(intra_block_dists) > 0 else 0.0
        
        if len(block_centroids) > 1:
            block_centroids = np.array(block_centroids)
            inter_block_pdist = pdist(block_centroids, metric='euclidean')
            inter_var = float(np.mean(inter_block_pdist))
        else:
            inter_var = 0.0
            
        ratio = float(inter_var / intra_var) if intra_var > 0 else 0.0
        stats_per_taxon.append({
            'taxon': sp,
            'n_blocks': int(len(unique_blocks)),
            'intra_specimen_dist': intra_var,
            'inter_specimen_dist': inter_var,
            'variance_ratio': ratio
        })

    # Global pairwise distance sampling
    rng = np.random.RandomState(42)
    N = 10000

    # 1. Intra-specimen pairs
    intra_pairs_1, intra_pairs_2 = [], []
    for c in df['label'].unique():
        sub = df[df['label'] == c]
        for b in sub['subfolder'].unique():
            b_idx = sub[sub['subfolder'] == b].index.values
            if len(b_idx) >= 2:
                s1 = rng.choice(b_idx, size=min(100, len(b_idx)), replace=False)
                s2 = rng.choice(b_idx, size=min(100, len(b_idx)), replace=False)
                intra_pairs_1.extend(s1)
                intra_pairs_2.extend(s2)

    idx_sub = rng.choice(len(intra_pairs_1), size=min(N, len(intra_pairs_1)), replace=False)
    p1 = np.array(intra_pairs_1)[idx_sub]
    p2 = np.array(intra_pairs_2)[idx_sub]
    diff_intra = embs[p1] - embs[p2]
    dists_intra = np.linalg.norm(diff_intra, axis=1)
    dists_intra = dists_intra[dists_intra > 1e-6]

    # 2. Inter-specimen (same taxon) pairs
    inter_spec_1, inter_spec_2 = [], []
    for c in df['label'].unique():
        sub = df[df['label'] == c]
        blocks = sub['subfolder'].unique()
        if len(blocks) >= 2:
            for _ in range(300):
                b1, b2 = rng.choice(blocks, size=2, replace=False)
                i1 = rng.choice(sub[sub['subfolder'] == b1].index.values)
                i2 = rng.choice(sub[sub['subfolder'] == b2].index.values)
                inter_spec_1.append(i1)
                inter_spec_2.append(i2)

    diff_inter_spec = embs[np.array(inter_spec_1)] - embs[np.array(inter_spec_2)]
    dists_inter_spec = np.linalg.norm(diff_inter_spec, axis=1)

    # 3. Inter-taxon pairs
    inter_tax_1, inter_tax_2 = [], []
    taxa = df['label'].unique()
    for _ in range(N):
        t1, t2 = rng.choice(taxa, size=2, replace=False)
        i1 = rng.choice(df[df['label'] == t1].index.values)
        i2 = rng.choice(df[df['label'] == t2].index.values)
        inter_tax_1.append(i1)
        inter_tax_2.append(i2)

    diff_inter_tax = embs[np.array(inter_tax_1)] - embs[np.array(inter_tax_2)]
    dists_inter_tax = np.linalg.norm(diff_inter_tax, axis=1)

    df_stats = pd.DataFrame(stats_per_taxon)
    multi_block_df = df_stats[df_stats['n_blocks'] > 1]

    output_data = {
        'taxa_variance_statistics': stats_per_taxon,
        'summary': {
            'mean_intra_specimen_dist': float(multi_block_df['intra_specimen_dist'].mean()),
            'mean_inter_specimen_dist': float(multi_block_df['inter_specimen_dist'].mean()),
            'mean_variance_ratio': float(multi_block_df['variance_ratio'].mean()),
            'global_pairwise': {
                'intra_specimen_mean': float(np.mean(dists_intra)),
                'intra_specimen_std': float(np.std(dists_intra)),
                'inter_specimen_intra_taxon_mean': float(np.mean(dists_inter_spec)),
                'inter_specimen_intra_taxon_std': float(np.std(dists_inter_spec)),
                'inter_taxon_mean': float(np.mean(dists_inter_tax)),
                'inter_taxon_std': float(np.std(dists_inter_tax)),
                'ratio_inter_spec_to_inter_taxon': float(np.mean(dists_inter_spec) / np.mean(dists_inter_tax)),
                'ratio_intra_to_inter_specimen': float(np.mean(dists_intra) / np.mean(dists_inter_spec))
            }
        }
    }

    with open('outputs/specimen_variance_analysis.json', 'w') as f:
        json.dump(output_data, f, indent=2)

    df_stats.to_csv('outputs/taxon_specimen_variance.csv', index=False)
    print("SUCCESS: Saved outputs/specimen_variance_analysis.json and outputs/taxon_specimen_variance.csv")

if __name__ == '__main__':
    main()
