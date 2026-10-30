import numpy as np
import pandas as pd
import json
import time
import random
import os
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

def main():
    print("Loading cached embeddings and splits...")
    embs = np.load('outputs1/cached_embeddings_tf_efficientnetv2_m_in21k.npy')
    tr_df = pd.read_csv('outputs1/split_hard_constrained/train.csv')
    va_df = pd.read_csv('outputs1/split_hard_constrained/val.csv')
    te_df = pd.read_csv('outputs1/split_hard_constrained/test.csv')
    df = pd.concat([tr_df, va_df, te_df], ignore_index=True)
    path_to_idx = {p: i for i, p in enumerate(df['path'])}
    classes = sorted(df['label'].unique().tolist())

    cache_file = 'outputs/class_splits_cache.pkl'
    if os.path.exists(cache_file):
        print("Loading pre-computed splits cache from pickle...")
        with open(cache_file, 'rb') as f:
            class_splits_cache = pickle.load(f)
    else:
        print("Precomputing candidate splits for each class...")
        from split_protocols import SPLIT_METHODS
        candidate_solvers = {k: v for k, v in SPLIT_METHODS.items() if k not in ['PP0_Stratified_Random', 'PP6_Stratified_Random', 'PP11_DataSAIL_Image']}
        class_splits_cache = {c: {} for c in classes}
        for c in classes:
            sub_df = df[df['label'] == c].copy().reset_index(drop=True)
            sub_indices = [path_to_idx[p] for p in sub_df['path']]
            sub_embs = embs[sub_indices]
            for s_name, s_fn in candidate_solvers.items():
                try:
                    tr_s, va_s, te_s = s_fn(sub_df, sub_embs, seed=42)
                    class_splits_cache[c][s_name] = (
                        [path_to_idx[p] for p in tr_s['path']],
                        [path_to_idx[p] for p in va_s['path']],
                        [path_to_idx[p] for p in te_s['path']]
                    )
                except Exception:
                    pass
        os.makedirs('outputs', exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(class_splits_cache, f)

    print("Cache prepared. Available solvers per class:", {c: len(class_splits_cache[c]) for c in classes})

    def eval_cfg(cfg):
        tr_idx, va_idx, te_idx = [], [], []
        for c in classes:
            p_name = cfg[c]
            tr_c, va_c, te_c = class_splits_cache[c][p_name]
            tr_idx.extend(tr_c)
            va_idx.extend(va_c)
            te_idx.extend(te_c)
            
        tr_e = embs[tr_idx]
        va_e = embs[va_idx]
        
        # 1. MMD Train vs Val
        m_tr = tr_e.mean(axis=0)
        m_va = va_e.mean(axis=0)
        mmd = float(np.linalg.norm(m_tr - m_va))
        
        # 2. 1-NN on Val
        knn = KNeighborsClassifier(n_neighbors=1, metric='cosine', n_jobs=-1)
        knn.fit(tr_e, df.iloc[tr_idx]['label'])
        preds_va = knn.predict(va_e)
        y_true_va = df.iloc[va_idx]['label']
        rep = classification_report(y_true_va, preds_va, output_dict=True, zero_division=0)
        f1s = [v['f1-score'] for k, v in rep.items() if k in classes]
        hardest_f1 = float(min(f1s)) if f1s else 0.0
        macro_f1 = float(rep['macro avg']['f1-score'])
        acc = float(rep['accuracy'])
        
        # 3. DataSAIL loss proxy (sample 200 vectors)
        sub_tr_e = tr_e[::max(1, len(tr_e)//200)]
        sub_va_e = va_e[::max(1, len(va_e)//200)]
        sim_matrix = sub_tr_e @ sub_va_e.T
        sim_loss = float(np.sum(sim_matrix > 0.90))
        
        # Multi-objective fitness per Eq (3) in paper:
        # fitness = w1 * (sim_loss / 100) - w2 * (10 * mmd) - w3 * (10 * hardest_f1)
        fitness = 1.0 * (sim_loss / 100.0) - 0.5 * (10.0 * mmd) - 0.5 * (10.0 * hardest_f1)
        return {
            'fitness': fitness,
            'mmd': mmd,
            'hardest_f1': hardest_f1,
            'macro_f1': macro_f1,
            'acc': acc,
            'sim_loss': sim_loss
        }

    N_TRIALS = 1500
    print(f"Running Systematic Benchmark: Random Search vs Simulated Annealing ({N_TRIALS} iterations each)...")

    # --- RUN RANDOM SEARCH ---
    random.seed(42)
    np.random.seed(42)
    t0 = time.time()
    rs_best_fit = float('inf')
    rs_best_res = None
    rs_history = []

    for it in range(N_TRIALS):
        cand_cfg = {c: random.choice(list(class_splits_cache[c].keys())) for c in classes}
        res = eval_cfg(cand_cfg)
        if res['fitness'] < rs_best_fit:
            rs_best_fit = res['fitness']
            rs_best_res = res
        rs_history.append(rs_best_fit)

    t_rs = time.time() - t0
    hf1_rs = rs_best_res['hardest_f1'] * 100.0
    mf1_rs = rs_best_res['macro_f1'] * 100.0
    print(f"Random Search finished in {t_rs:.2f}s | Best Fitness: {rs_best_fit:.4f} | Hardest-F1: {hf1_rs:.2f}% | Macro-F1: {mf1_rs:.2f}% | MMD: {rs_best_res['mmd']:.4f}")

    # --- RUN SIMULATED ANNEALING ---
    random.seed(42)
    np.random.seed(42)
    t0 = time.time()
    curr_cfg = {c: random.choice(list(class_splits_cache[c].keys())) for c in classes}
    curr_res = eval_cfg(curr_cfg)
    curr_fit = curr_res['fitness']

    sa_best_cfg = curr_cfg.copy()
    sa_best_fit = curr_fit
    sa_best_res = curr_res
    sa_history = []

    # Calibrated temperature: initial temp = 1.0 (mean fitness change ~ 0.2 - 0.5)
    temp = 1.0
    cooling = 0.997

    for it in range(N_TRIALS):
        tgt_c = random.choice(classes)
        avail = list(class_splits_cache[tgt_c].keys())
        new_p = random.choice(avail)
        cand_cfg = curr_cfg.copy()
        cand_cfg[tgt_c] = new_p
        
        cand_res = eval_cfg(cand_cfg)
        cand_fit = cand_res['fitness']
        delta = cand_fit - curr_fit
        
        # Metropolis acceptance criterion for minimization
        if delta < 0 or random.random() < np.exp(-delta / max(1e-4, temp)):
            curr_cfg = cand_cfg
            curr_fit = cand_fit
            curr_res = cand_res
            if curr_fit < sa_best_fit:
                sa_best_fit = curr_fit
                sa_best_cfg = curr_cfg.copy()
                sa_best_res = curr_res
                
        temp = max(0.01, temp * cooling)
        sa_history.append(sa_best_fit)

    t_sa = time.time() - t0
    hf1_sa = sa_best_res['hardest_f1'] * 100.0
    mf1_sa = sa_best_res['macro_f1'] * 100.0
    print(f"Simulated Annealing finished in {t_sa:.2f}s | Best Fitness: {sa_best_fit:.4f} | Hardest-F1: {hf1_sa:.2f}% | Macro-F1: {mf1_sa:.2f}% | MMD: {sa_best_res['mmd']:.4f}")

    # Save results
    results = {
        'search_space': '11^18 (~ 5.56 x 10^18 states)',
        'iterations': N_TRIALS,
        'random_search': {
            'runtime_s': t_rs,
            'best_fitness': rs_best_fit,
            'hardest_f1': rs_best_res['hardest_f1'],
            'macro_f1': rs_best_res['macro_f1'],
            'mmd': rs_best_res['mmd'],
            'acc': rs_best_res['acc']
        },
        'simulated_annealing': {
            'runtime_s': t_sa,
            'best_fitness': sa_best_fit,
            'hardest_f1': sa_best_res['hardest_f1'],
            'macro_f1': sa_best_res['macro_f1'],
            'mmd': sa_best_res['mmd'],
            'acc': sa_best_res['acc'],
            'best_configuration': sa_best_cfg
        }
    }

    os.makedirs('outputs', exist_ok=True)
    with open('outputs/benchmark_sa_vs_random.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Successfully generated and saved outputs/benchmark_sa_vs_random.json")

if __name__ == '__main__':
    main()
