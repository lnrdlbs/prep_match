import argparse
import ast
import errno
import hashlib
import time
import csv
import json
import os
from pathlib import Path
import pickle
import re
from typing import Any
from numpy.random import MT19937, RandomState, SeedSequence
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from data import Data
from pymfe.mfe import MFE
keys_to_remove= ['nr_bin', 'nr_cat', 'nr_num', 'num_to_cat', 'sparsity']

def silent_remove(filename):
    try:
        os.remove(filename)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occurred

def check_columns_and_combine(folder_path, 
                              relevant_check: bool = True, 
                              target_init: str= 'dc', 
                              target_k: list = None, 
                              target_optimizer: str = 'genetic',
                              target_p: list = None):
    if not relevant_check:
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    else:
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv') and is_relevant_file(f, 
                                                                                                target_k=target_k,
                                                                                                target_init=target_init,
                                                                                                target_optimizer=target_optimizer,
                                                                                                target_p=target_p)]
    dataframes = []
    
    first_file = os.path.join(folder_path, csv_files[0])
    first_df = pd.read_csv(first_file)
    reference_columns = first_df.columns
    p_match = re.search(r'p_(\d+)', first_file)
    p_value = int(p_match.group(1)) # Extract the number after 'p_'
    first_df['p'] = p_value
    dataframes.append(first_df)
    
    for file in csv_files[1:]:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        p_match = re.search(r'p_(\d+)', file_path)
        if p_match:
            p_value = int(p_match.group(1)) # Extract the number after 'p_'
            
            df['p'] = p_value  
        else:
            print(file_path)
        
        # if not df.columns.equals(reference_columns):
        #     print(f"Columns do not match in file: {file}")
        #     print(f"Expected columns: {list(reference_columns)}")
        #     print(f"Found columns: {list(df.columns)}")
        #     return None
        dataframes.append(df)
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(combined_df['p'].unique())
    return combined_df


relevant_mf_keys = [
    "has_missing_values",
    "cov.mean",
    "cov.sd",
    "eigenvalues.mean",
    "eigenvalues.sd",
    "iq_range.mean",
    "iq_range.sd",
    "kurtosis.sd",
    "mad.mean",
    "mad.sd",
    "max.mean",
    "max.sd",
    "mean.sd",
    "median.mean",
    "median.sd",
    "min.sd",
    "nr_corr_attr",
    "nr_norm",
    "range.mean",
    "range.sd",
    "skewness.mean",
    "skewness.sd",
    "t_mean.sd",
    "var.mean",
    "var.sd",
    "t3",
    "t4",
    "wg_dist.sd",
]


def get_real_world_data_by_name(base_path: str, dataset_name):
    dataset_name = f"{dataset_name}.pickle"
    rw_path = os.path.join(base_path, dataset_name)
    with open(rw_path, "rb") as f:
        data = pickle.load(f)
    return data

def is_relevant_file2(    filename: str,
    target_k: list,
    target_optimizer: str,
    target_p: list,
):
    base_name = filename.replace(".csv", "")

    optimizer_match = re.search(r"_([A-Za-z0-9]+)_[A-Za-z0-9]+_k_", base_name)

    k_match = re.search(r"_k_(\d+)_p_", base_name)
    p_match = re.search(r"_p_(\d+)", base_name)

    if not (optimizer_match and k_match and p_match):
        return False

    # Extract values
    optimizer = optimizer_match.group(1)
    k = int(k_match.group(1))
    p = int(p_match.group(1))
    if (
        k in target_k
        and optimizer == target_optimizer
        and p in target_p
    ):
        return True
    return False


def is_relevant_file(
    filename: str,
    target_k: list,
    target_init: str,
    target_optimizer: str,
    target_p: list,
):
    base_name = filename.replace(".csv", "")

    optimizer_match = re.search(r"_([a-zA-Z0-9]+)_init_", base_name)
    init_match = re.search(r"_init_([a-zA-Z0-9]+)_k_", base_name)
    k_match = re.search(r"_k_(\d+)_p_", base_name)
    p_match = re.search(r"_p_(\d+)", base_name)

    if not (optimizer_match and init_match and k_match and p_match):
        return False

    # Extract values
    optimizer = optimizer_match.group(1)
    init = init_match.group(1)
    k = int(k_match.group(1))
    p = int(p_match.group(1))

    if (
        k in target_k
        and init == target_init
        and optimizer == target_optimizer
        and p in target_p
    ):
        return True
    return False


def write_time_dict(path: str, time_dict: dict):
    with open(path, "w") as f:
        json.dump(time_dict, f)


def get_filename_from_ml_run(result_dict: dict) -> str:
    name = get_filename(
        knowledge_base_name=result_dict["kb_name"],
        dataset_name=result_dict["dataset_name"],
        k=result_dict["k"],
        p=result_dict["nr_pipelines"],
        mf_groups = result_dict['mf_groups'],
        uuid=result_dict['uuid'],
        clustering_algo=result_dict['clustering']
    )
    return name


def get_filename(dataset_name: str, knowledge_base_name: str, k: int, p: int,  clustering_algo: str, mf_groups: list[str], uuid: str=None,) -> str:
    mf_groups_str = '_'.join(mf_groups)
    if dataset_name == 'blobs':
        name = f"{dataset_name}_{uuid}_{knowledge_base_name}_k_{k}_p_{p}_{mf_groups_str}_{clustering_algo}.csv"
    else: 
        name = f"{dataset_name}_{knowledge_base_name}_k_{k}_p_{p}_{mf_groups_str}_{clustering_algo}.csv"
    return name


def get_kb_name_from_filename(filename: str) -> str:
    name = Path(filename).stem
    if name == "reduced_second_stage":
        return "genetic_init_dc"
    elif name == "reduced_random_init":
        return "genetic_init_random"
    elif name == "random_imp_3":
        return "random_search"
    elif name == "random_help_3":
        return "random_search_help"
    elif name == "kb":
        return "genetic_big"
    else:
        raise ValueError(f"unknown kb name: {name}")


def arg_str2array(string):
    if string.startswith("[") and string.endswith("]"):
        strings_list = string[1:-1].split(",")
        return strings_list
    else:
        raise ValueError("Input should be in the form [a,b,c]")


def format_number_k(num):
    if num >= 1000:
        return f"{num/1000:.0f}k"
    else:
        return str(num)


def visualize_cluster_with_outliers(X, y, outliers, save_location: str = "tmp_output/"):

    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.scatter(outliers[:, 0], outliers[:, 1], color="red", marker="x")
    plt.savefig(save_location + f"{current_milli_time()}.png")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def current_milli_time():
    return round(time.time() * 1000)


def results_to_csv_file(results, result_path, group):

    results_dict_list = []
    for result in results:
        tmp_r = {}
        tmp_r["seed"] = result["seed"]
        tmp_r["shape"] = result["shape"]
        tmp_r["n_samples"] = result["n_samples"]
        tmp_r["n_clusters"] = result["n_clusters"]
        tmp_r["n_features"] = result["n_features"]
        tmp_r["cluster_std"] = result["cluster_std"]
        tmp_r["total_time"] = result["total_time"]
        tmp_r["pp_time"] = result["pp_time"]

        if group == "outlier":
            tmp_r["p_outlier"] = result["p_outlier"]
            tmp_r["outlier_method"] = result["outlier_method"]
            tmp_r["labels_outlier_fit"] = result["labels_outlier_fit"]
            tmp_r["labels_outlier_removed_fit"] = result["labels_outlier_removed_fit"]
            tmp_r["outlier_ari"] = result["outlier_ari"]
            tmp_r["outlier_nmi"] = result["outlier_nmi"]
            tmp_r["outlier_removed_ari"] = result["outlier_removed_ari"]
            tmp_r["outlier_removed_nmi"] = result["outlier_removed_nmi"]
            tmp_r["mf_base"] = result["mf_base"]
            tmp_r["mf_outlier"] = result["mf_outlier"]
            tmp_r["mf_outlier_removed"] = result["mf_outlier_removed"]

            if result["outlier_method"] == "lof":
                tmp_dict = {
                    "p_neighbors": result["p_neighbors"],
                    "algorithm": result["algorithm"],
                    "p_leaf_size": result["p_leaf_size"],
                    "metric": result["metric"],
                    "contamination": result["contamination"],
                    "n_jobs": result["n_jobs"],
                    "p": result["p"],
                }
                tmp_r["outlier_method_params"] = tmp_dict

            elif result["outlier_method"] == "iso_forest":
                tmp_dict = {
                    "n_estimators": result["n_estimators"],
                    "max_samples": result["max_samples"],
                    "contamination": result["contamination"],
                    "max_features": result["max_features"],
                    "bootstrap": result["bootstrap"],
                    "n_jobs": result["n_jobs"],
                    "warm_start": result["warm_start"],
                }
                tmp_r["outlier_method_params"] = tmp_dict
            else:
                raise ValueError(f'Unknown outlier method {result["outlier_method"]}')
        if group == "missing_value":
            tmp_r["p_missing_value"] = result["p_missing_value"]
            tmp_r["missing_value_method"] = result["missing_value_method"]
            tmp_r["labels_replaced_constant"] = result["labels_replaced_constant"]
            tmp_r["labels_replaced"] = result["labels_replaced"]
            tmp_r["replaced_constant_ari"] = result["replaced_constant_ari"]
            tmp_r["replaced_constant_nmi"] = result["replaced_constant_nmi"]
            tmp_r["replaced_ari"] = result["replaced_ari"]
            tmp_r["replaced_nmi"] = result["replaced_nmi"]
            tmp_r["mf_base"] = result["mf_base"]
            tmp_r["mf_missing_values"] = result["replaced_nmi"]
            tmp_r["mf_replaced_constant"] = result["mf_replaced_constant"]
            tmp_r["mf_replaced"] = result["mf_replaced"]

            if result["missing_value_method"] == "nni":
                tmp_dict = {
                    "p_neighbors": result["p_neighbors"],
                    "weights": result["weights"],
                    "metric": result["metric"],
                }
                tmp_r["missing_value_method_params"] = tmp_dict
            elif (
                result["missing_value_method"] == "mean"
                or result["missing_value_method"] == "median"
                or result["missing_value_method"] == "most_frequent"
            ):
                tmp_r["missing_value_method_params"] = {
                    "keep_empty_features": result["keep_empty_features"]
                }
            else:
                raise ValueError(f'Unknown outlier method {result["outlier_method"]}')
        if group == "scaling":
            tmp_r["multiplier_methods"] = result["multiplier_methods"]
            tmp_r["scaling_method"] = result["scaling_method"]
            tmp_r["labels_multiplied"] = result["labels_multiplied"]
            tmp_r["labels_scaled"] = result["labels_scaled"]
            tmp_r["multiplied_ari"] = result["multiplied_ari"]
            tmp_r["multiplied_nmi"] = result["multiplied_nmi"]
            tmp_r["scaled_ari"] = result["scaled_ari"]
            tmp_r["scaled_nmi"] = result["scaled_nmi"]
            tmp_r["mf_base"] = result["mf_base"]
            tmp_r["mf_multiplied"] = result["mf_multiplied"]
            tmp_r["mf_scaled"] = result["mf_scaled"]

            tmp_dict = {}
            if result["scaling_method"] == "robust_scaler":
                tmp_dict["with_centering"] = result["with_centering"]
                tmp_dict["with_scaling"] = result["with_scaling"]
                tmp_dict["unit_variance"] = result["unit_variance"]
            elif result["scaling_method"] == "standad_scaler":
                tmp_dict["with_mean"] = result["with_mean"]
                tmp_dict["with_std"] = result["with_std"]
            elif result["scaling_method"] == "power_transformer":
                tmp_dict["standadize"] = result["standadize"]
            elif result["scaling_method"] == "quantile_transformer":
                tmp_dict["n_quantiles"] = result["n_quantiles"]
                tmp_dict["output_distribution"] = result["output_distribution"]
                tmp_dict["ignore_implicit_zeros"] = result["ignore_implicit_zeros"]
            tmp_r["scaling_method_params"] = tmp_dict
        tmp_r["base_ari"] = result["base_ari"]
        tmp_r["base_nmi"] = result["base_nmi"]
        results_dict_list.append(tmp_r)

    df = pd.DataFrame(results_dict_list)
    # logger.debug('write_to_csv')
    df.to_csv(result_path)


def replace_nan_with_constant(X: ndarray, constant: int = 0) -> ndarray:
    X = np.nan_to_num(X, nan=constant)
    return X


def add_prefix_to_dict_keys(d: dict, prefix: str):
    return {f"{prefix}{k}": v for k, v in d.items()}


def get_numpy_rs(seed: int) -> RandomState:
    return RandomState(MT19937(SeedSequence(seed)))


def print_hof(hof, top_n=-1):

    i = 0
    if not top_n - 1:
        for el in hof:
            print_hof_single(el)
    else:
        for el in hof:
            print_hof_single(el)
            i = i + 1
            if i >= top_n:
                break


def print_hof_single(element):
    print(f"{element}: {element.fitness.values}")


def get_random_method(methods, current_methods):
    filtered = [m for m in methods if m not in current_methods]

    return np.random.choice(filtered)


def random_method_selection_possible(methods, current_methods):
    filtered = [m for m in methods if m not in current_methods]
    return len(filtered) > 0


def get_methods_of_pp_dict(pp_dict):
    methods = [key for subdict in pp_dict.values() for key in subdict.keys()]
    return methods


def get_best_params(hof, method_name):

    params = {}
    best_fitness = -1
    for candidate in hof:
        for method in candidate[0]:
            if method == method_name and candidate.fitness.values[0] > best_fitness:
                params = candidate[1][method]

    return params, best_fitness


def get_random_params(method, flat_methods_dict):
    params_dict = flat_methods_dict[method]
    params = {}
    for key, values in params_dict.items():
        params[key] = (
            np.random.choice(values)
            if isinstance(values, list)
            else np.random.randint(values.start, values.stop)
        )

    return params


def save_fitness_image(logbook):
    gen = logbook.select("gen")

    fit_mins = logbook.select("min")
    fit_max = logbook.select("max")

    _, ax1 = plt.subplots()
    line1 = ax1.plot(gen, fit_mins, "-b", label="Minimum Fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness", color="b")
    for tl in ax1.get_yticklabels():
        tl.set_color("b")

    ax2 = ax1.twinx()
    line2 = ax2.plot(gen, fit_max, "r-", label="Maximum, Fitness")
    ax2.set_ylabel("Fitness", color="r")
    for tl in ax2.get_yticklabels():
        tl.set_color("r")

    lns = line1 + line2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="center right")
    plt.savefig("output/test_deap.png")


def filename_contains_hash(filename: str, hash_value: str) -> bool:
    base = os.path.splitext(os.path.basename(filename))[0]
    tokens = re.split(r'[^0-9A-Za-z]+', base)
    return hash_value in tokens
# def get_optimization_file_name(d: Data):
#     return f"{d.metric}_mutpb_{d.mutpb}_cxpb{d.cxpb}_indpb{d.indpb}_seed_{d.seed}_time_budget_{d.time_budget}_samples_{d.n_samples}_features{d.n_features}_cluster_std_{d.cluster_std}_s_{d.has_scaling}_o_{d.has_outlier}_m_{d.has_missing_values}_generations_{d.n_generation}_population_size_{d.population_size}.csv"

def make_run_hash(*params: Any) -> str:
    blob = json.dumps(params, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]

def write_deap_result(file, data_object: Data):

    tmp_r = {}
    tmp_r["id"] = data_object.id
    tmp_r["uuid"] = data_object.uuid
    tmp_r["X_base"] = data_object.X_base
    tmp_r["y"] = data_object.y
    tmp_r["X_missing_values"] = data_object.X_missing_values
    tmp_r["y_missing_values"] = data_object.y_missing_values
    tmp_r["X_modified"] = data_object.X_modified
    tmp_r["y_modified"] = data_object.y_modified
    tmp_r["X_preprocessed"] = data_object.X_preprocessed
    tmp_r["y_preprocessed"] = data_object.y_preprocessed
    tmp_r["seed"] = data_object.seed
    tmp_r["n_samples"] = data_object.n_samples
    tmp_r["n_clusters"] = data_object.n_clusters
    tmp_r["n_features"] = data_object.n_features
    tmp_r["cluster_std"] = data_object.cluster_std
    tmp_r["has_scaling"] = data_object.has_scaling
    tmp_r["has_outlier"] = data_object.has_outlier
    tmp_r["multiplier_method"] = data_object.multiplier_method
    tmp_r["p_outlier"] = data_object.p_outlier
    tmp_r["p_missing_values"] = data_object.p_missing_values
    tmp_r["base_ari"] = data_object.base_ari
    tmp_r["modified_ari"] = data_object.modified_ari
    tmp_r["preprocessed_ari"] = data_object.preprocessed_ari
    tmp_r["generations"] = data_object.generations
    tmp_r["deap_min"] = data_object.deap_min
    tmp_r["deap_avg"] = data_object.deap_avg
    tmp_r["deap_max"] = data_object.deap_max
    tmp_r["best_pipeline"] = data_object.best_pipeline
    tmp_r["best_fitness"] = data_object.best_fitness

    tmp_r["mf_base"] = data_object.mf_base
    tmp_r["mf_modified"] = data_object.mf_modified
    tmp_r["mf_missing_values"] = data_object.mf_missing_values
    tmp_r["mf_preprocessed"] = data_object.mf_preprocessed

    tmp_r["top_5_pipeline"] = data_object.top_5_pipeline
    tmp_r["top_5_fitness"] = data_object.top_5_fitness

    tmp_r["stopped_by_budget"] = data_object.stopped_by_budget

    result = pd.DataFrame([tmp_r])

    if not os.path.exists(file):
        result.to_csv(file)
    else:
        result.to_csv(file, mode="a", header=False)

def dict_replace_nan_with_zero(d):
    return {k: (0 if pd.isna(v) else v) for k, v in d.items()}

def get_run_hash(data: Data):
    hash_str = make_run_hash(data.metric, data.mutpb, data.cxpb, data.indpb, data.seed, data.time_budget, data.n_samples, data.n_features, data.cluster_std, data.multiplier_method, data.p_outlier, data.p_missing_values, data.n_generation, data.population_size, data.shape, data.tourn_size)
    return hash_str
def write_deap_result2(d: Data, output_folder: str, overwrite:bool=True):
    
    hash_str = get_run_hash(d)
    output_path = os.path.join(output_folder, f'{d.uuid}_{hash_str}.csv')
    d.run_hash = hash_str
    if not os.path.exists(output_path) or overwrite:
        pd.Series(d.__dict__).to_frame().T.to_csv(output_path, index=False)
    else:
        raise FileExistsError()


    # tmp_r = {}
    # tmp_r["id"] = data_object.id
    # tmp_r["uuid"] = data_object.uuid
    # tmp_r["X_base"] = data_object.X_base
    # tmp_r["y"] = data_object.y
    # tmp_r["X_missing_values"] = data_object.X_missing_values
    # tmp_r["y_missing_values"] = data_object.y_missing_values
    # tmp_r["X_modified"] = data_object.X_modified
    # tmp_r["y_modified"] = data_object.y_modified
    # tmp_r["X_preprocessed"] = data_object.X_preprocessed
    # tmp_r["y_preprocessed"] = data_object.y_preprocessed
    # tmp_r["seed"] = data_object.seed
    # tmp_r["n_samples"] = data_object.n_samples
    # tmp_r["n_clusters"] = data_object.n_clusters
    # tmp_r["n_features"] = data_object.n_features
    # tmp_r["cluster_std"] = data_object.cluster_std
    # tmp_r["has_scaling"] = data_object.has_scaling
    # tmp_r["has_outlier"] = data_object.has_outlier
    # tmp_r["multiplier_method"] = data_object.multiplier_method
    # tmp_r["p_outlier"] = data_object.p_outlier
    # tmp_r["p_missing_values"] = data_object.p_missing_values
    # tmp_r["base_ari"] = data_object.base_ari
    # tmp_r["modified_ari"] = data_object.modified_ari
    # tmp_r["preprocessed_ari"] = data_object.preprocessed_ari
    # tmp_r["generations"] = data_object.generations
    # tmp_r["deap_min"] = data_object.deap_min
    # tmp_r["deap_avg"] = data_object.deap_avg
    # tmp_r["deap_max"] = data_object.deap_max
    # tmp_r["best_pipeline"] = data_object.best_pipeline
    # tmp_r["best_fitness"] = data_object.best_fitness

    # tmp_r["mf_base"] = data_object.mf_base
    # tmp_r["mf_modified"] = data_object.mf_modified
    # tmp_r["mf_missing_values"] = data_object.mf_missing_values
    # tmp_r["mf_preprocessed"] = data_object.mf_preprocessed

    # tmp_r["top_5_pipeline"] = data_object.top_5_pipeline
    # tmp_r["top_5_fitness"] = data_object.top_5_fitness

    # tmp_r["stopped_by_budget"] = data_object.stopped_by_budget

    # result = pd.DataFrame([tmp_r])

    # if not os.path.exists(file):
    #     result.to_csv(file)
    # else:
    #     result.to_csv(file, mode="a", header=False)


def get_data_objects_by_ids(data_folder: str, ids: list[int]):

    data_objects = []
    for data_id in ids:
        file_start = f"data_{data_id}_"
        for f_in_dir in os.listdir(data_folder):
            if f_in_dir.startswith(file_start):
                combined_path = os.path.join(data_folder, f_in_dir)
                with open(combined_path, "rb") as f:
                    obj = pickle.load(f)
                    data_objects.append(obj)

    return data_objects


def get_data_objects(data_folder: str):
    data_objects = []

    for f_in_dir in os.listdir(data_folder):
        combined_path = os.path.join(data_folder, f_in_dir)
        if Path(f_in_dir).suffix != ".pickle":
            continue
        with open(combined_path, "rb") as f:
            obj = pickle.load(f)
            data_objects.append(obj)
    return data_objects

def get_rw_for_optimizer(data_folder: str):
    data_objects: list[Data] = []
    for f_in_dir in os.listdir(data_folder):
        combined_path = os.path.join(data_folder, f_in_dir)
        if Path(f_in_dir).suffix != ".pickle":
            continue
        with open(combined_path, "rb") as f:
            obj = pickle.load(f)
            data_objects.append(obj)
    return data_objects
def get_data_object_by_uuid(data_folder: str, uuid):

    matching_files = [f for f in os.listdir(data_folder) if uuid in f]
    assert len(matching_files) == 1, f"{len(matching_files)}, {uuid}, {data_folder}"
    combined_path = os.path.join(data_folder, matching_files[0])
    with open(combined_path, "rb") as f:
        obj = pickle.load(f)
        return obj


def write_missing_value_imputed_to_pickle(data_object: Data, base_path):
    Path(base_path).mkdir(parents=True, exist_ok=True)
    with open(f"{base_path}/{data_object.uuid}.pickle", "wb") as f:
        pickle.dump(data_object, f)


def write_runtime_log(file_path: str, data_id: tuple, runtime: float):

    if not os.path.exists(file_path):
        with open(file_path, "w") as file:
            file.write("")

    with open(file_path, "a+") as f:
        writer = csv.writer(f)
        writer.writerow([data_id, runtime])


def write_checkpoint(checkpoint_path: str, cp: dict, data: Data):

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    full_path = f"{checkpoint_path}/{data.id}_{data.uuid}.pickle"
    with open(full_path, "wb") as f:
        pickle.dump(cp, f)


def read_cp(checkpoint_path, data: Data):
    full_path = os.path.join(checkpoint_path, f"{data.id}_{data.uuid}.pickle")
    with open(full_path, "rb") as f:
        cp = pickle.load(f)
    return cp


def get_ari_clustering(X,y,n_clusters, algorithm):
    if algorithm == 'kmeans':
        kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=876)
        labels = kmeans.fit_predict(X)
    elif algorithm == 'dbscan':
        dbscan = DBSCAN()
        labels = dbscan.fit_predict(X)
    elif algorithm == 'hdbscan':
        hdbscan = HDBSCAN()
        labels = hdbscan.fit_predict(X)
    elif algorithm=='agglomerative':
        agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
        labels = agglomerative.fit_predict(X)
    else:
        raise ValueError('Unknown Clustering Algorithm')
    ari = adjusted_rand_score(y, labels)
    return ari

def checkpoint_available(checkpoint_path: str, data: Data):
    full_path = f"{checkpoint_path}/{data.id}_{data.uuid}.pickle"
    return os.path.isfile(full_path)


def get_ari(X, y, n_clusters):
    try:
        kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=876)

        labels = kmeans.fit_predict(X)
        ari = adjusted_rand_score(y, labels)

    except ValueError as e:
        print(f"Error kmeans {e}")
        return 0
    else:
        return ari


def get_nmi(X, y, n_clusters):
    try:
        kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=876)

        labels = kmeans.fit_predict(X)
        nmi = normalized_mutual_info_score(y, labels)

    except ValueError as e:
        print(f"Error kmeans {e}")
        return 0
    else:
        return nmi

def get_ami(X, y, n_clusters):
    try:
        kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=876)

        labels = kmeans.fit_predict(X)
        nmi = adjusted_mutual_info_score(y, labels)

    except ValueError as e:
        print(f"Error kmeans {e}")
        return 0
    else:
        return nmi


def validate_individual(ind):
    methods = ind[0]
    methods_in_params = list(ind[1].keys())
    methods_in_oder = [e[1] for e in ind[2]]
    return set(methods) == set(methods_in_params) == set(methods_in_oder)


def parse_tuple(arg):
    try:
        return tuple(map(int, arg.split(",")))
    except ValueError:
        raise argparse.ArgumentTypeError("Tuple must be integers separated by a comma.")


def remove_keys(d, keys):
    return {k: v for k, v in d.items() if k not in keys}

def add_key(d, new_key, condition):
    if condition:
        d[new_key] = 1
    else:
        d[new_key] = 0
    return d
def filter_dict_by_keys(d, keys):
    return {k: v for k, v in d.items() if k in keys}

def remove_nan_values(d):
    return {k: v for k, v in d.items() if not pd.isna(v)}

def preprocess_object(data: Data, replace_nan_with_zero: bool =False):
    data.mf_base = remove_keys(data.mf_base, keys_to_remove)
    data.mf_modified = remove_keys(data.mf_modified, keys_to_remove)

    data.mf_modified = add_key(data.mf_modified, 'has_missing_values', data.p_missing_values >0)
    if replace_nan_with_zero:
        data.mf_modified = dict_replace_nan_with_zero(data.mf_modified)
    else:
        data.mf_modified = remove_nan_values(data.mf_modified)

    data.mf_modified_relevant = filter_dict_by_keys(data.mf_modified, relevant_mf_keys)
    data.X_modified = np.array(data.X_modified, dtype=np.float64)
    return data

def parse_metric_string(s):
    s = re.sub(r"\bnan\b", "None", s)
    return ast.literal_eval(s)


def preprocess_df(df):

    df['mf_modified'] = df['mf_modified'].apply(parse_metric_string)
    df['mf_base'] = df['mf_base'].apply(parse_metric_string)
    df['best_pipeline'] = df['best_pipeline'].apply(ast.literal_eval)

    df['mf_base'] = df['mf_base'].apply(lambda x: remove_keys(x, keys_to_remove))
    df['mf_modified'] = df['mf_modified'].apply(lambda x: remove_keys(x, keys_to_remove))
    df['top_5_pipeline'] = df['top_5_pipeline'].apply(ast.literal_eval)



    df['mf_modified'] = df.apply(
        lambda row: add_key(row['mf_modified'], 'has_missing_values', row['p_missing_values'] > 0),
        axis=1)


    df['mf_modified_relevant'] = df['mf_modified'].apply(lambda x: filter_dict_by_keys(x, relevant_mf_keys))
    return df


def filter_list_of_dicts(dicts, groups, drop_unknown=True):
    """
    Filter a list of meta-feature dictionaries by specified groups.

    Parameters
    ----------
    dicts : list[dict]
        Each dict maps meta-feature names to values.
    groups : set[str] | list[str]
        The meta-feature group names to keep (e.g. {"general", "statistical"}).
    drop_unknown : bool, default=True
        If True, drop features whose group is unknown.
        If False, keep them.

    Returns
    -------
    list[dict]
        List of filtered dictionaries.
    """
    # Build name -> group lookup once locally
    catalog = MFE.metafeature_description(print_table=False)
    name_to_group = {name: group for (group, name, *_ ) in catalog[0]}

    def _base_name(key):
        return key.split('.', 1)[0]

    def _key_to_group(key):
        return name_to_group.get(key) or name_to_group.get(_base_name(key))

    filtered_dicts = []
    for d in dicts:
        new_d = {}
        for k, v in d.items():
            g = _key_to_group(k)
            if g in groups:
                new_d[k] = v
            elif g is None and not drop_unknown:
                new_d[k] = v
        filtered_dicts.append(new_d)

    return filtered_dicts