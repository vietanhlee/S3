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
    classes = sorted(df['label'].unique().tolist())

    cache_file = 'outputs/class_splits_cache.pkl'
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Cache file {cache_file} not found!")

    with open(cache_file, 'rb') as f:
        class_splits_cache = pickle.load(f)

    print("Splits cache loaded. Taxa count:", len(classes))

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
        
        # 3. DataSAIL loss proxy
        sub_tr_e = tr_e[::max(1, len(tr_e)//200)]
        sub_va_e = va_e[::max(1, len(va_e)//200)]
        sim_matrix = sub_tr_e @ sub_va_e.T
        sim_loss = float(np.sum(sim_matrix > 0.90))
        
        # Multi-objective fitness per Eq (3) in paper:
        fitness = 1.0 * (sim_loss / 100.0) - 0.5 * (10.0 * mmd) - 0.5 * (10.0 * hardest_f1)
        return {
            'fitness': fitness,
            'mmd': mmd,
            'hardest_f1': hardest_f1,
            'macro_f1': macro_f1,
            'acc': acc,
            'sim_loss': sim_loss
        }

    # Grid of hyperparameters to evaluate:
    # Initial Temperatures T0: [0.1, 0.5, 1.0, 2.5, 5.0]
    # Cooling Rates alpha: [0.990, 0.995, 0.997, 0.999]
    t0_list = [0.1, 0.5, 1.0, 2.5, 5.0]
    alpha_list = [0.990, 0.995, 0.997, 0.999]
    N_ITERS = 400  # fast and effective for grid sweep

    results_grid = []

    print(f"Starting Hyperparameter Sweep: {len(t0_list)} T0 values x {len(alpha_list)} alpha values ({N_ITERS} iters each)...")

    # Generate a fixed starting configuration for fair comparison across grid points
    random.seed(42)
    init_cfg = {c: random.choice(list(class_splits_cache[c].keys())) for c in classes}
    init_res = eval_cfg(init_cfg)
    init_fit = init_res['fitness']

    sweep_start = time.time()

    for t0 in t0_list:
        for alpha in alpha_list:
            t_pt_start = time.time()
            random.seed(42)
            np.random.seed(42)

            curr_cfg = init_cfg.copy()
            curr_fit = init_fit
            curr_res = init_res

            best_cfg = curr_cfg.copy()
            best_fit = curr_fit
            best_res = curr_res

            temp = t0
            accepted_moves = 0
            worse_accepted = 0

            for it in range(N_ITERS):
                tgt_c = random.choice(classes)
                avail = list(class_splits_cache[tgt_c].keys())
                new_p = random.choice(avail)
                cand_cfg = curr_cfg.copy()
                cand_cfg[tgt_c] = new_p

                cand_res = eval_cfg(cand_cfg)
                cand_fit = cand_res['fitness']
                delta = cand_fit - curr_fit

                accept = False
                if delta < 0:
                    accept = True
                else:
                    prob = np.exp(-delta / max(1e-4, temp))
                    if random.random() < prob:
                        accept = True
                        worse_accepted += 1

                if accept:
                    accepted_moves += 1
                    curr_cfg = cand_cfg
                    curr_fit = cand_fit
                    curr_res = cand_res
                    if curr_fit < best_fit:
                        best_fit = curr_fit
                        best_cfg = curr_cfg.copy()
                        best_res = curr_res

                temp = max(1e-4, temp * alpha)

            pt_runtime = time.time() - t_pt_start
            acceptance_rate = (accepted_moves / N_ITERS) * 100.0

            grid_entry = {
                'T0': t0,
                'alpha': alpha,
                'final_temp': temp,
                'acceptance_rate_pct': round(acceptance_rate, 2),
                'worse_accepted_count': worse_accepted,
                'best_fitness': round(best_fit, 4),
                'hardest_f1': round(best_res['hardest_f1'] * 100.0, 2),
                'macro_f1': round(best_res['macro_f1'] * 100.0, 2),
                'val_acc': round(best_res['acc'] * 100.0, 2),
                'mmd': round(best_res['mmd'], 4),
                'runtime_s': round(pt_runtime, 2)
            }
            results_grid.append(grid_entry)
            print(f"T0={t0:4.1f} | alpha={alpha:5.3f} -> Fit={best_fit:6.4f} | HardestF1={best_res['hardest_f1']*100:5.2f}% | MacroF1={best_res['macro_f1']*100:5.2f}% | AccRate={acceptance_rate:4.1f}% | Time={pt_runtime:4.1f}s")

    total_sweep_time = time.time() - sweep_start
    print(f"\nGrid sweep completed in {total_sweep_time:.2f}s across {len(results_grid)} configurations.")

    output_payload = {
        'description': 'Simulated Annealing hyperparameter sensitivity analysis across initial temperatures (T0) and cooling schedules (alpha) on 18-class timber dataset',
        'iterations_per_trial': N_ITERS,
        'search_space': '11^18 (~ 5.56 x 10^18 states)',
        'total_grid_runtime_s': round(total_sweep_time, 2),
        'grid_results': results_grid
    }

    out_path = 'outputs/sa_hyperparameter_sensitivity.json'
    with open(out_path, 'w') as f:
        json.dump(output_payload, f, indent=2)
    print(f"Results successfully saved to {out_path}")

if __name__ == '__main__':
    main()
