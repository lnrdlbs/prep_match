import os
import sys
import glob
import copy
import json
import argparse

import pandas as pd
import numpy as np

#from scipy.spatial.distance import pdist
#from scipy.spatial.distance import squareform
from scipy.stats import skew, kurtosis, zscore

from sklearn.metrics.pairwise import nan_euclidean_distances
from sklearn.cluster import MeanShift, DBSCAN, OPTICS
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
#from sklearn.preprocessing import MinMaxScaler, RobustScaler

from autoclust import getErrorVaule
from autoclust import *

def extract_features(data: np.ndarray) -> dict:
    """
    Berechnet alle meta features für einen Datensatz data.

    :return: Ein Dictionary der form: meta_features = {"MF1_mean": value, ...}
    """
    meta_features = {}

    # Annahme: euklidische distanz (paarweise) 
    # Annahme: Falls werte fehlen wird das feature bei in den jenigen spalten ignoriert
    dist_matrix = nan_euclidean_distances(data)
    tri_upper_indices = np.triu_indices(len(dist_matrix), k=1)
    raw_distance = dist_matrix[tri_upper_indices]
   
    #Annahme: Min Max Skalierung zur normalisierung auf intervall [0,1]
    norm_distance = (raw_distance - raw_distance.min()) /(raw_distance.max() - raw_distance.min()) if raw_distance.max() >  raw_distance.min() else raw_distance

    #MF1 -Mf5
    meta_features = calculate_statisticMeasures(norm_distance)

    #MF6-MF15
    meta_features.update(calculate_histogram_measures(norm_distance))

    #M16 M19
    meta_features.update(calculate_zscoreMeasures(raw_distance))

    #MF20 - MF28

    # Annhame: Datenpunkten bei denen werte fehlen werden raussgeschmissen sonst stürtzen die clustering algorithmen ab
    data_clean = data[~np.isnan(data).any(axis=1)]
    meta_features.update(calculate_landmarkers(data_clean))

    return meta_features


def calculate_statisticMeasures(distance) -> dict :
    meta_features ={}
    measures = {
        "mean": np.mean,
        "var": np.var,
        "std": np.std,
        "skew": skew,
        "kurtosis": kurtosis
    }

    for i, (name, func) in enumerate(measures.items(), 1):
        meta_features[f"MF{i}_{name}"] = func(distance)

    return meta_features



def calculate_histogram_measures(distance) -> dict:
    meta_features = {}

    hist, _ = np.histogram(distance, bins=10, range=(0, 1))
    probs = hist / len(distance)

    for i in range(10):
        start_pct = i * 10
        end_pct = (i + 1) * 10
        
        meta_features[f"MF{6+i}_hist_{start_pct}_{end_pct}"] = probs[i]

    return meta_features


def calculate_zscoreMeasures(distance) -> dict:
    meta_features = {}
    abs_z = np.abs(zscore(distance))

    # Annahme: das sind die 4 intervalle
    bins = [0, 1, 2, 3, np.inf]
    hist, _ = np.histogram(abs_z, bins=bins)
    probs = hist / len(distance)
    
    labels = ["0_1", "1_2", "2_3", "3_plus"]

    for i in range(4):
        meta_features[f"MF{16+i}_zscore_bin_{labels[i]}"] = probs[i]

    return meta_features


def calculate_landmarkers (data) -> dict:
    
    meta_features={}
    
    current_feature = 20

    models = {"MeanShift": MeanShift(),
              "DBSCAN": DBSCAN(),
              "OPTICS": OPTICS()}
    
    metric_info = [
        (calinski_harabasz_score, "CHI"),
        (davies_bouldin_score, "DBI"),
        (silhouette_score, "SC")
    ]
    error_defaults = [getErrorVaule(m[0]) for m in metric_info]

    
    for name, model in models.items():

        try:
            labels = model.fit_predict(data)

            # Annahme: NoiseKlasse -1 wird mit dazugezählt
            n_cluster = len(set(labels))

            if n_cluster > 1:
                scores = [m[0](data, labels) for m in metric_info]

            else:
                # Annhame: falls nur ein cluster
                scores = error_defaults
                
        except Exception as e:
            print(e)
            # Annahme: falls exception wärend des clustering oder der evaluation bekommen alle schlechten score 
            scores = error_defaults

        for i, score in enumerate(scores):
            metric_name = metric_info[i][1]             
            meta_features[f"MF{current_feature}_{name}_{metric_name}"] = score
            current_feature += 1

    return meta_features


# Annhame: Für DBI haben wir error value 1e9 selbst gewählt (verzerrt ziemlich) -> wird mit max(real_dbi) + 2*std(real_dbi) "korrigiert", Die landmarker werden nicht auf [0,1] skaliert da es nicht im paper steht, haben dadurch deutlich mehr, unterschiedlich viel einfluss
def fix_dbi_errors_statistically(d_new_metafeatures: dict, knowledge_base_metafeatures: dict):
    """
    Ersetzt 1e9-Werte in DBI-Features durch max(real_dbi) + 2*std(real_dbi) -> (damit die distanzberechnung nicht unfassbar durch unsere eigene Annahme der 1e9 werte verzerrt wird.)
    Keine weitere Normalisierung oder Skalierung.
    """
    dbi_keys = ["MF21_MeanShift_DBI", "MF24_DBSCAN_DBI", "MF27_OPTICS_DBI"]
    all_dicts = [d_new_metafeatures] + list(knowledge_base_metafeatures.values())
    
    for key in dbi_keys:
        vals = np.array([d[key] for d in all_dicts], dtype=float)
        
        error_mask = np.isclose(vals, 1e9)
        real_vals = vals[~error_mask]
        
        if len(real_vals) > 0:

            penalty_value = np.max(real_vals) + (2 * np.std(real_vals))
            
            for d in all_dicts:
                if np.isclose(d[key], 1e9):
                    d[key] = float(penalty_value)
        else:
            #Falls der Key überall 1e9 ist (sehr unwahrscheinlich) dann 1, damit der rest nicht verzerrt wird
            for d in all_dicts:
                d[key] = 1



def get_rankedDistances_for_dnew_from_kb(d_new_metafeatures : dict, knowledge_base_metafeatures: dict, m : int = 50) -> list:
    """
    Berechnet die distanz zwischen den metafeatures von d_new zu den metafeatures der datensätze aus der kb.

    :param d_new_metafeatures: Dict mit Features von D_new
    :param knowledge_base_metafeatures: Dict der Form { "datensatzID": {features_dict} }
    :param m: Anzahl der zurückzugebenden Datensätze
    :return: Liste von standardmäßig m=50 Tupeln [(datensatzID, distanz_zu_dnew), ...] sortiert nach Ähnlichkeit
    """
    
    # damit dbi immer neu gefixt werden kann und ursprügliche dicts nicht geändert werden
    d_new_local = copy.deepcopy(d_new_metafeatures)
    kb_local = copy.deepcopy(knowledge_base_metafeatures)

    fix_dbi_errors_statistically(d_new_local, kb_local)

    rankedDistances = []
    
    keys = sorted(d_new_local.keys())
    new_vec = np.array([d_new_local[k] for k in keys]) # vektor d_new

    for dataset, dataset_features in kb_local.items():

        kb_vec = np.array([dataset_features[k] for k in keys]) # vektor kb
        
        # Annahme: Es wird die Euklidische Distanz als distanzmaß verwendet
        distance = np.linalg.norm(new_vec - kb_vec)
        
        rankedDistances.append((dataset, distance))

    rankedDistances.sort(key=lambda x: x[1])
    
    return rankedDistances[:m]



def load_knowledge_base(path: str) -> tuple:
    """
    Loads a knowledge_base JSON file consolidated by prepare_knowledge_base.py.

    :return: (meta_features_by_dataset, hofs_by_dataset)
        meta_features_by_dataset: {dataset_id: {meta_features}}  -> directly usable by get_rankedDistances_for_dnew_from_kb
        hofs_by_dataset: {dataset_id: [{"rank","ari","pipeline"}, ...]}  -> pipelines still as strings, only parsed
        against d_new's pset at warmstart/recommendation time (see get_PipleneInstantiations_for_dnew / get_recommended_pipelines)
    """
    with open(path, "r", encoding="utf-8") as f:
        knowledge_base = json.load(f)

    meta_features_by_dataset = {name: entry["meta_features"] for name, entry in knowledge_base.items()}
    hofs_by_dataset = {name: entry["hall_of_fame"] for name, entry in knowledge_base.items()}

    return meta_features_by_dataset, hofs_by_dataset


def _parse_pipeline_string(pipeline_str: str, pset):
    # Assumption: incompatible pipelines (eg. PCA n_components > n_features of d_new) should be
    # skipped instead of crashing, since the pset has feature-count-dependent terminals
    try:
        return creator.Individual.from_string(pipeline_str, pset)
    except TypeError:
        return None


def get_PipleneInstantiations_for_dnew(hofs : dict, rankedDistances: list, pset=None):
    """
    Returns the pipeline instantiations for the online-phase warmstart for a dataset d_new.

    :param hofs: A dictionary of the form: {dataset_id: hof, ...}. hof is either a list of real
        DEAP Individuals (legacy, in-memory mode) or a list of {"rank","ari","pipeline"} dicts
        (knowledge_base mode, see load_knowledge_base) - auto-detected per dataset_id from the
        type of the first entry.
    :rankedDistances: A list of the form  [(dataset_id, distance_to_dnew), ...]
    :param pset: Only needed in knowledge_base mode, to parse the pipeline strings back into Individuals
    :returns: List of Individuals
    """
    population = []
    skipped = []

    for dataset, _ in rankedDistances:

        hof = hofs.get(dataset)

        if not hof:
            continue

        if isinstance(hof[0], dict):
            # knowledge_base mode: pipelines are stored as strings, the first one that parses (rank order) is used
            for entry in hof:
                ind = _parse_pipeline_string(entry["pipeline"], pset)
                if ind is not None:
                    population.append(ind)
                    break
            else:
                skipped.append(dataset)
        else:
            # legacy mode: hof[0] is already a real DEAP Individual
            klon = copy.deepcopy(hof[0])
            del klon.fitness.values
            population.append(klon)

    if skipped:
        print(f"Warmstart: skipped {len(skipped)} dataset(s) (no compatible pipeline for d_new's feature count): {skipped}")

    return population


def get_recommended_pipelines(hofs: dict, rankedDistances: list, pset, k: int = 10) -> list:
    """
    Collects up to k unique, pset-compatible pipeline Individuals for the pure meta-learning
    recommendation (no subsequent GA evolution). Unlike get_PipleneInstantiations_for_dnew, this
    considers all hall_of_fame entries of a dataset (not just the best) before moving on to the
    next-most-similar dataset.

    :param hofs: {dataset_id: [{"rank","ari","pipeline"}, ...]}  (knowledge_base mode)
    :param rankedDistances: [(dataset_id, distance_to_dnew), ...]
    :param pset: pset of d_new, against which the pipeline strings are parsed
    :param k: maximum number of unique recommended pipelines
    :returns: list of up to k Individuals (unevaluated)
    """
    recommended = []
    seen = set()

    for dataset, _ in rankedDistances:
        hof = hofs.get(dataset)
        if not hof:
            continue

        for entry in hof:
            if len(recommended) >= k:
                return recommended

            ind = _parse_pipeline_string(entry["pipeline"], pset)
            if ind is None:
                continue

            ind_str = str(ind)
            if ind_str in seen:
                continue
            seen.add(ind_str)
            recommended.append(ind)

    return recommended


def evaluate_pipeline_candidates(individuals: list, pset, data, measure, is_offline: bool, target=None) -> list:
    """
    Evaluates each candidate directly (no GA evolution) via autoclust.evaluatePipeline.

    :returns: list of (pipeline_str, performance, length) tuples, sorted descending by performance
        (performance is always oriented by evaluatePipeline so that higher is better)
    """
    results = [
        (str(ind), *evaluatePipeline(ind, pset, data, measure, is_offline, target=target))
        for ind in individuals
    ]
    results.sort(key=lambda x: x[1], reverse=True)
    return results


EVAL_MEASURES = {
    "ari": (adjusted_rand_score, True),
    "silhouette": (silhouette_score, False),
    "calinski_harabasz": (calinski_harabasz_score, False),
    "davies_bouldin": (davies_bouldin_score, False),
}


def resolve_input_csv_paths(input_folder: str) -> list:
    """
    Recursively collects all *.csv file paths under input_folder, sorted by path.
    """
    csv_paths = []
    for root, _, files in os.walk(input_folder):
        for file_name in files:
            if file_name.lower().endswith(".csv"):
                csv_paths.append(os.path.join(root, file_name))
    csv_paths.sort()
    return csv_paths


def run_online_phase_for_dataset(dataset_path: str, kb_metafeatures: dict, kb_hofs: dict, args) -> dict:
    """
    Runs the online phase (ranking + warmstart GA, or recommend-only evaluation) for one d_new CSV
    against an already-loaded knowledge base. Raises ValueError on validation problems (eg. missing
    target column for --eval-measure ari) so callers can decide whether to abort (single-dataset
    mode) or skip-and-continue (batch mode).

    :returns: a JSON-serializable result dict (see code/KNOWLEDGE_BASE.md for the schema)
    """
    d_new_name = os.path.basename(dataset_path)
    df_new = pd.read_csv(dataset_path)

    has_target = args.target_column in df_new.columns
    y_true = df_new[args.target_column].values if has_target else None
    X = df_new.drop(columns=[args.target_column]).values if has_target else df_new.values

    print(f"D_new: {d_new_name}")
    d_new_features = extract_features(X)

    # exclude d_new itself from the ranking pool, in case it happens to share a name with a knowledge_base entry
    kb_metafeatures_local = {name: feats for name, feats in kb_metafeatures.items() if name != d_new_name}

    ranked_list = get_rankedDistances_for_dnew_from_kb(d_new_features, kb_metafeatures_local, m=args.m)

    print(f"\nMost similar datasets to '{d_new_name}':")
    print(f"{'Rank':<6} | {'Dataset':<25} | {'Distance':<10}")
    print("-" * 50)
    for rank, (name, dist) in enumerate(ranked_list, 1):
        print(f"{rank:<6} | {name:<25} | {dist:.4f}")

    reset_deap_creators()
    my_pset = createPset(n_features=X.shape[1])
    my_toolbox = createToolbox(my_pset, maxHeight=args.max_tree_height)

    result = {
        "dataset_name": d_new_name,
        "dataset_path": os.path.abspath(dataset_path),
        "mode": args.mode,
        "n_features": X.shape[1],
        "ranked_datasets": [{"dataset": name, "distance": dist} for name, dist in ranked_list],
    }

    if args.mode == "warmstart":
        new_pop = get_PipleneInstantiations_for_dnew(kb_hofs, ranked_list, pset=my_pset)
        print(f"the new generation consists of {len(new_pop)} individual(s)")

        print(f"starting online phase for {d_new_name}")

        hof_SC = runGeneticAlgorithm(my_toolbox, X, silhouette_score, False, args.time_limit_online, warmstartPop=new_pop)
        print(f"# pipelines SC {len(hof_SC)}")

        hof_DBI = runGeneticAlgorithm(my_toolbox, X, davies_bouldin_score, False, args.time_limit_online, warmstartPop=new_pop)
        print(f"# pipelines DBI {len(hof_DBI)}")

        hof_CHI = runGeneticAlgorithm(my_toolbox, X, calinski_harabasz_score, False, args.time_limit_online, warmstartPop=new_pop)
        print(f"# pipelines CHI {len(hof_CHI)}")

        lables = create_clustering_ensamble(my_toolbox, X, [hof_DBI, hof_CHI, hof_SC])

        print(lables)

        result["warmstart_population_size"] = len(new_pop)
        result["labels"] = [int(label) for label in lables]

    else:
        # --- recommend-only mode: pure meta-learning, no GA evolution ---
        measure, is_offline = EVAL_MEASURES[args.eval_measure]
        if is_offline and not has_target:
            raise ValueError(f"--eval-measure {args.eval_measure} requires a '{args.target_column}' column in {dataset_path}.")

        candidates = get_recommended_pipelines(kb_hofs, ranked_list, my_pset, k=args.k)
        print(f"Collected {len(candidates)} recommended pipeline(s) from the knowledge_base.")

        eval_results = evaluate_pipeline_candidates(candidates, my_pset, X, measure, is_offline, target=y_true)

        print(f"\n{'Rank':<6} | {'Score':<10} | {'Length':<6} | Pipeline")
        print("-" * 80)
        for rank, (pipeline_str, score, length) in enumerate(eval_results, 1):
            print(f"{rank:<6} | {score:<10.4f} | {length:<6} | {pipeline_str}")

        result["eval_measure"] = args.eval_measure
        result["k"] = args.k
        result["results"] = [
            {"rank": rank, "pipeline": pipeline_str, "score": score, "length": length}
            for rank, (pipeline_str, score, length) in enumerate(eval_results, 1)
        ]

    return result


if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))

    parser = argparse.ArgumentParser(
        description="Without --knowledge-base, the original demo (offline+online phase over "
                     "data/*.csv) runs unchanged. With --knowledge-base, a knowledge_base file "
                     "consolidated beforehand via prepare_knowledge_base.py is used for the online phase instead."
    )
    parser.add_argument("--knowledge-base", type=str, default=None,
        help="Path to a knowledge_base JSON file produced by prepare_knowledge_base.py. "
             "If not set, the script runs as before (data/*.csv demo).")
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument("--dataset-path", type=str, default=None,
        help="Path to d_new's CSV. Only relevant with --knowledge-base. Cannot be combined with --input-folder.")
    dataset_group.add_argument("--input-folder", type=str, default=None,
        help="Folder of d_new CSVs to evaluate in batch (searched recursively). Only relevant with "
             "--knowledge-base. Cannot be combined with --dataset-path.")
    parser.add_argument("--output-dir", type=str, default=os.path.join(repo_root, "online_phase_results"),
        help="Folder for per-dataset JSON result files in batch mode (default: online_phase_results). "
             "Only used with --input-folder. Each dataset's result is written as soon as that dataset "
             "finishes, and datasets whose result file already exists are skipped (and logged) rather "
             "than recomputed, so an interrupted batch run can simply be re-run to resume.")
    parser.add_argument("--target-column", type=str, default="target",
        help="Name of the target column in d_new, if present (default: 'target')")
    parser.add_argument("--m", type=int, default=50,
        help="Number of nearest datasets from the knowledge_base to consider (default: 50)")
    parser.add_argument("--mode", type=str, choices=["warmstart", "recommend-only"], default="warmstart",
        help="warmstart: GA online phase with knowledge_base pipelines as starting population (default). "
             "recommend-only: no GA evolution, instead up to --k pipelines from the knowledge_base "
             "are evaluated directly against --eval-measure.")
    parser.add_argument("--k", type=int, default=10,
        help="Number of recommended pipelines in recommend-only mode (default: 10)")
    parser.add_argument("--eval-measure", type=str, choices=list(EVAL_MEASURES.keys()), default="ari",
        help="Evaluation measure in recommend-only mode (default: 'ari', requires --target-column in d_new)")
    parser.add_argument("--time-limit-online", type=int, default=90,
        help="Time budget per GA run of the online phase in seconds (default: 90, only used in warmstart mode)")
    parser.add_argument("--max-tree-height", type=int, default=6,
        help="Maximum height of the generated GP pipeline trees (default: 6)")
    parser.add_argument("--output-path", type=str, default=None,
        help="If set: additionally write the result (labels or recommended pipelines) to this file. "
             "Only used with --dataset-path (single-dataset mode).")
    args = parser.parse_args()

    if args.knowledge_base is None:
        # --- Legacy Demo Modus: unverändertes Verhalten ---

        TimeLimit_offline = 30 # time of genetic algorithm run in seconds
        TimeLimit_online = 90
        MaxTreeHeight = 6

        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.abspath(os.path.join(current_dir, "..", "data"))
        csv_files = glob.glob(os.path.join(data_dir, "*.csv"))


        #kb aufbauen also meta features berechnen und offline phase durchlaufen lassen
        knowledge_base = {}
        hofs = {}

        for path in csv_files:
            file_name = os.path.basename(path)
            print(f"--- Extrahiere Features für: {file_name} ---")


            # meta feature calculation
            try:
                df = pd.read_csv(path)
                X = df.drop(columns=["target"]).values
                df_target = df["target"].values

                features = extract_features(X)

                knowledge_base[file_name] = features
                print(f"Erfolgreich hinzugefügt. ({len(features)} Features)")

            except Exception as e:
                print(f"Fehler beim mf von {file_name}: {e}")


            #offlie phase
            try:
                print(f"starte oflline phase für {file_name}")

                if "FitnessMulti" in creator.__dict__: del creator.FitnessMulti
                if "Individual" in creator.__dict__: del creator.Individual

                creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0)) #maximierung d. evaluationsmetrik (im paper steht minimizing aber ARI und CVIs maximieren alle ausser DBI (wird deswegen später negiert)), und minimierung der pipeline-Complexity (lenghth)
                creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)

                my_pset = createPset(n_features=X.shape[1])
                my_toolbox = createToolbox(my_pset, maxHeight=MaxTreeHeight)

                hof = runGeneticAlgorithm(my_toolbox, X, adjusted_rand_score , True, TimeLimit_offline, target=df_target)

                hofs[file_name] = hof

            except Exception as e:
                print(e)


        # distanzen zu kb berechnen
        if len(knowledge_base) >= 2:

            # de_new ist momentan einfach der erste datensatz im ordner
            d_new_name = os.path.basename(csv_files[0])


            print(f"D_new (Test-Datensatz): {d_new_name}")
            d_new_features = knowledge_base[d_new_name]

            kb_for_ranking = knowledge_base.copy()
            del kb_for_ranking[d_new_name]

            ranked_list = get_rankedDistances_for_dnew_from_kb(
                d_new_features,
                kb_for_ranking,
                m=50
            )
            print(f"\nTop ähnlichste Datensätze zu '{d_new_name}':")
            print(f"{'Platz':<6} | {'Datensatz':<25} | {'Distanz':<10}")
            print("-" * 50)

            for rank, (name, dist) in enumerate(ranked_list, 1):
                print(f"{rank:<6} | {name:<25} | {dist:.4f}")


        # die neuen instanzen für d_new berechnen

        new_pop = get_PipleneInstantiations_for_dnew(hofs, ranked_list)

        print(f"die neue generation besteht aus {len(new_pop)} anzahl individuen")


        # online phase für d_new:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(current_dir, "..", "data", d_new_name)
        d_new = pd.read_csv(path)
        X = d_new.drop(columns=["target"]).values

        print(f"starte online phase für {d_new_name}")

        if "FitnessMulti" in creator.__dict__: del creator.FitnessMulti
        if "Individual" in creator.__dict__: del creator.Individual

        creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0)) #maximierung d. evaluationsmetrik (im paper steht minimizing aber ARI und CVIs maximieren alle ausser DBI (wird deswegen später negiert)), und minimierung der pipeline-Complexity (lenghth)
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)

        my_pset = createPset(n_features=X.shape[1])
        my_toolbox = createToolbox(my_pset, maxHeight=MaxTreeHeight)


        hof_SC = runGeneticAlgorithm(my_toolbox, X, silhouette_score, False, TimeLimit_online, warmstartPop= new_pop)
        print(f"# piplines SC {len(hof_SC)}")

        hof_DBI = runGeneticAlgorithm(my_toolbox, X, davies_bouldin_score, False, TimeLimit_online, warmstartPop= new_pop)
        print(f"# piplines DBI {len(hof_DBI)}")

        hof_CHI = runGeneticAlgorithm(my_toolbox, X, calinski_harabasz_score, False, TimeLimit_online, warmstartPop= new_pop)
        print(f"# piplines CHI {len(hof_CHI)}")


        lables = create_clustering_ensamble(my_toolbox, X,  [hof_DBI, hof_CHI, hof_SC])

        print(lables)

    elif args.input_folder:
        # --- knowledge_base mode, batch over a folder ---
        kb_metafeatures, kb_hofs = load_knowledge_base(args.knowledge_base)

        output_dir = prepare_output_folder(args.output_dir)
        used_file_names = set()
        written, skipped_existing, failed = [], [], []

        dataset_paths = resolve_input_csv_paths(args.input_folder)
        print(f"Found {len(dataset_paths)} CSV dataset(s) in {args.input_folder}.")

        for dataset_path in dataset_paths:
            output_path = dataset_output_path(dataset_path, output_dir, used_file_names)

            if os.path.exists(output_path):
                skipped_existing.append(output_path)
                print(f"Skipping dataset because result already exists: {output_path}")
                continue

            try:
                result = run_online_phase_for_dataset(dataset_path, kb_metafeatures, kb_hofs, args)
                write_result_to_path(result, output_path)
                written.append(output_path)
                print(f"Result written: {output_path}")
            except Exception as e:
                failed.append(dataset_path)
                print(f"Skipping dataset due to unexpected error: {dataset_path} ({e})", file=sys.stderr)
                traceback.print_exc()
                continue

        print(f"\nResults written to {len(written)} file(s):")
        for output_path in written:
            print(f"- {output_path}")
        if skipped_existing:
            print(f"\nDatasets skipped because result files already existed: {len(skipped_existing)}")
            for output_path in skipped_existing:
                print(f"- {output_path}")
        if failed:
            print(f"\nDatasets skipped due to unexpected errors: {len(failed)}")
            for dataset_path in failed:
                print(f"- {dataset_path}")

    else:
        # --- knowledge_base mode, single dataset ---
        if not args.dataset_path:
            parser.error("either --dataset-path or --input-folder is required when --knowledge-base is set.")

        kb_metafeatures, kb_hofs = load_knowledge_base(args.knowledge_base)

        try:
            result = run_online_phase_for_dataset(args.dataset_path, kb_metafeatures, kb_hofs, args)
        except ValueError as e:
            parser.error(str(e))

        if args.output_path:
            if result["mode"] == "warmstart":
                pd.DataFrame({"label": result["labels"]}).to_csv(args.output_path, index=False)
                print(f"Labels written to: {args.output_path}")
            else:
                pd.DataFrame(result["results"]).to_csv(args.output_path, index=False)
                print(f"Result written to: {args.output_path}")