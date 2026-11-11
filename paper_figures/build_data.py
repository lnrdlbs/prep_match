import glob
import json
import os
import pickle
import shutil
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper_figures", "export", "data")
sys.path.insert(0, ROOT)

DATASETS = [
    "banknote", "dermatology", "diagnosis", "ecoli", "f_mnist_10k", "f_mnist_20k", "frogs",
    "glass", "ionosphere", "iris", "letter", "obesity", "pendigits", "segment", "usps", "wdbc",
]
ALGORITHMS = ["kmeans", "dbscan", "hdbscan", "agglomerative"]
KB_FILES = ["b", "m_1", "m_2", "s", "o", "ms_1", "ms_2", "mo_1", "mo_2", "so", "mso_1", "mso_2"]

if __name__ == "__main__":
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    for alg in ALGORITHMS:
        os.makedirs(os.path.join(OUT, f"prepmatch_{alg}"))
        for ds in DATASETS:
            files = glob.glob(os.path.join(ROOT, "eval_sigmod", f"{alg}_23_5", f"{ds}_genetic_big_k_23_p_5_all*.csv"))
            assert len(files) == 1, files
            shutil.copy2(files[0], os.path.join(OUT, f"prepmatch_{alg}"))

    rows = []
    for ds in DATASETS:
        with open(os.path.join(ROOT, "real_world_data", f"{ds}.pickle"), "rb") as f:
            rows.append({"dataset": ds, "n_samples": len(pickle.load(f).X_modified)})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "datasets.csv"), index=False)

    rs_files = []
    for ds in DATASETS:
        files = glob.glob(os.path.join(ROOT, "rs_21_5", f"random_iter_{ds}_105_length_(1, 3)_strat_no_seed_*.csv"))
        assert len(files) == 5, files
        rs_files += sorted(files)
    rs = pd.concat([pd.read_csv(f, usecols=["dataset_name", "seed", "ari_before", "best_ari"], dtype=str)
                    for f in rs_files], ignore_index=True)
    rs.to_csv(os.path.join(OUT, "random_search.csv"), index=False)

    rows = []
    for ds in DATASETS:
        path = os.path.join(ROOT, "tpe_results", f"{ds}.json")
        if os.path.exists(path):
            with open(path) as f:
                tpe = json.load(f)
            assert os.path.splitext(tpe["dataset_name"])[0].lower() == ds
            rows += [{"dataset_name": ds, "score": repr(r["score"])} for r in tpe["results"]]
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "tpe_autoclust.csv"), index=False)

    tpot = pd.read_csv(os.path.join(ROOT, "tpot_clustering_results", "tpot_clustering_results.csv"), dtype=str)
    tpot["dataset_name"] = tpot["dataset"].str.replace(".csv", "", regex=False)
    tpot = tpot[tpot["dataset_name"].isin(DATASETS)]
    assert len(tpot) == len(DATASETS)
    tpot[["dataset_name", "adjusted_rand_index"]].to_csv(os.path.join(OUT, "tpot_clustering.csv"), index=False)

    kb = pd.concat([pd.read_csv(os.path.join(ROOT, "res", "final", f"{name}.csv"),
                                usecols=["modified_ari", "best_fitness"], dtype=str).assign(file=name)
                    for name in KB_FILES], ignore_index=True)
    kb[["file", "modified_ari", "best_fitness"]].to_csv(os.path.join(OUT, "kb_results.csv"), index=False)

    files = sorted(glob.glob(os.path.join(ROOT, "output", "baselines", "*.csv")))
    rb = pd.concat([pd.read_csv(f, usecols=["random_score", "random_group_score"], dtype=str) for f in files],
                   ignore_index=True)
    rb.to_csv(os.path.join(OUT, "random_baselines.csv"), index=False)
    print("written to", OUT)
