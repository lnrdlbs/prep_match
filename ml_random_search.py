import argparse
import csv
import itertools
import os
from pathlib import Path
import pdb
import pickle
import random
import sys
import time

import numpy as np
import pandas as pd

from data import Data
from helper import (
    get_ari,
    get_data_objects,
    get_filename,
    parse_tuple,
    replace_nan_with_constant,
    str2bool,
    write_deap_result,
    write_runtime_log,
)

from helper import preprocess_object
from meta_features import get_all_meta_features
from pipeline_helper import (
    evaluate,
    pp_methods,
    pp_missing_value_methods,
    pp_methods_flat,
    mutate_pca_n_comb,
    mutate_iso_max_features,
    mutate_n_components,
    mutate_svd_solver,
    init_mv_hyperparameters,
    pp_methods_flat_include_mi,
)


def init_hyperparameters(method, data_object: Data, strategy: str):

    hps = {}
    if method in pp_methods['outlier'].keys():

        for param, values in pp_methods['outlier'][method].items():
            if (
                method == 'iso_forest'
                and param == 'max_features'
                and strategy == 'help'
            ):
                hps[param] = mutate_iso_max_features(data_object=data_object)
            hps[param] = (
                np.random.choice(values)
                if isinstance(values, list)
                else np.random.randint(values.start, values.stop)
            )

    elif method in pp_methods['scaling'].keys():
        if method == 'min_max' or method == 'max_abs':
            hps = {}
            return hps

        for param, values in pp_methods['scaling'][method].items():
            hps[param] = (
                np.random.choice(values)
                if isinstance(values, list)
                else np.random.randint(values.start, values.stop)
            )
    elif method in pp_methods['dim_red'].keys():
        for param, values in pp_methods['dim_red'][method].items():
            if method == 'pca':
                if param == 'n_components' and strategy == 'help':
                    hps[param] = mutate_pca_n_comb(data_object=data_object)
                elif param == 'svd_solver' and strategy == 'help':
                    hps[param] = mutate_svd_solver(hps=hps, method=method)
                else:
                    hps[param] = (
                        np.random.choice(values)
                        if isinstance(values, list)
                        else np.random.randint(values.start, values.stop)
                    )
            elif param == 'n_components' and strategy == 'help':
                hps[param] = mutate_n_components(data_object=data_object, method=method)
            else:
                hps[param] = (
                    np.random.choice(values)
                    if isinstance(values, list)
                    else np.random.randint(values.start, values.stop)
                )

    return hps


def main(
    d: Data,
    time_budget: int,
    max_iterations,
    pipeline_length_range,
    strategy: str,
    seed=876,
):
    np.random.seed(seed)
    result_dict = {}
    result_dict['kb_name'] = 'random'
    result_dict['seed'] = seed
    result_dict['dataset_name'] = d.shape
    d.X_modified = np.array(d.X_modified, dtype=np.float64)
    if np.isnan(d.X_modified).any():
        d.X_modified = replace_nan_with_constant(d.X_modified.copy(), constant=0)
    ari = get_ari(d.X_modified, d.y, d.n_clusters)
    print(f'Before {ari:0.2f}')
    result_dict['ari_before'] = ari

    budget_measure_start = time.time()
    stopped_by_budget = False
    max_ari = -1
    current_iteration = 0
    pls = []

    pp_time_start = time.time()
    best_pl = None
    while max_ari < 1.0 and current_iteration <= max_iterations:
        current_iteration = current_iteration + 1
        if time.time() - budget_measure_start >= 600:
            print(
                f'Time budget {time_budget} achieved after {current_iteration} iterations'
            )
            stopped_by_budget = True
            break

        length = np.random.randint(pipeline_length_range[0], pipeline_length_range[1])

        if strategy == 'help':
            if d.p_missing_values == 0:
                methods = np.random.choice(
                    list(pp_methods_flat.keys()), length, replace=False
                ).tolist()

            else:
                methods = []
                methods.append(np.random.choice(list(pp_missing_value_methods.keys())))
                methods = list(
                    itertools.chain(
                        *[
                            methods,
                            np.random.choice(
                                list(pp_methods_flat.keys()), length, replace=False
                            ),
                        ]
                    )
                )
            hps = {}
            for m in methods:
                if m in list(pp_missing_value_methods.keys()):
                    hp = init_mv_hyperparameters(method=m)
                else:
                    hp = init_hyperparameters(
                        method=m, data_object=d, strategy=strategy
                    )
                hps[m] = hp

            if d.p_missing_values == 0:
                np.random.shuffle(methods)
                order = [(i + 1, methods[i]) for i in range(len(methods))]
            else:
                first_el = methods[0]
                remaining_elements = methods[1:]
                np.random.shuffle(remaining_elements)
                shuffled_methods = [first_el] + remaining_elements
                order = [
                    (i + 1, shuffled_methods[i]) for i in range(len(shuffled_methods))
                ]
        else:
            hps = {}
            if d.p_missing_values > 0:
                d.X_modified = d.X_missing_values.copy()
                d.y_modified = d.y_missing_values.copy()
            methods = np.random.choice(
                list(pp_methods_flat_include_mi.keys()), length, replace=False
            ).tolist()
            for m in methods:
                if m in list(pp_missing_value_methods.keys()):
                    hp = init_mv_hyperparameters(method=m)
                else:
                    hp = init_hyperparameters(
                        method=m, data_object=d, strategy=strategy
                    )
                hps[m] = hp
            np.random.shuffle(methods)
            order = [(i + 1, methods[i]) for i in range(len(methods))]
        pipeline = [methods, hps, order]

        eval_time_start = time.time()
        d, invalid = evaluate(data_object=d, individual=pipeline)
        eval_time_end = time.time()

        if len(d.X_modified) * 0.5 > len(d.X_preprocessed):
            invalid = True
            d.preprocessed_ari = -1
        pl = (pipeline, d.preprocessed_ari, eval_time_end - eval_time_start)
        pls.append(pl)

        if max_ari <= d.preprocessed_ari:
            max_ari = d.preprocessed_ari
            best_pl = pl

    pp_time_end = time.time()
    result_dict['pp_time'] = pp_time_end - pp_time_start
    print(f'After {max_ari:0.2f}')
    print(f'Diff {max_ari-ari:0.2f}')
    result_dict['best_ari'] = max_ari
    result_dict['diff'] = max_ari - ari
    result_dict['best_pl'] = best_pl
    result_dict['pls'] = pls

    return result_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-sd',
        '--single_dataset',
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help='Use only single file',
    )
    parser.add_argument(
        '-d',
        '--datasets',
        type=str,
        required=False,
        help='Path to dataset folder (all files will be considered) works only of ds is not set',
    )
    parser.add_argument(
        '-dsp',
        '--single_dataset_path',
        type=str,
        required=False,
        help='Path to dataset file works only of ds is set',
    )
    parser.add_argument(
        '-o', '--output_folder', type=str, required=True, help='Output folder path'
    )
    parser.add_argument(
        '-oe',
        '--overwrite_existing',
        type=int,
        required=False,
        default=0,
        help='Overwrite existing results (param value =1), default is 0',
    )
    parser.add_argument(
        '-tb',
        '--time_budget',
        required=False,
        type=int,
        default=600,
        help='Time budget in seconds (default=600)',
    )
    parser.add_argument(
        '-mt',
        '--max_tries',
        type=int,
        required=False,
        default=50,
        help='Maximum pipelines to try',
    )
    parser.add_argument(
        '-pl',
        '--pipeline_length',
        required=False,
        type=parse_tuple,
        default=(1, 3),
        help=' pipeline length (min,max]',
    )
    parser.add_argument(
        '-s',
        '--strategy',
        type=str,
        default='help',
        help='Strategy, help, or full random',
    )
    parser.add_argument(
        '-nrs',
        '--number_seeds',
        type=int,
        default=0,
        help='Number of seeds that are random tried',
    )

    args = parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    file_paths = []
    if args.single_dataset:
        file_paths.append(args.single_dataset_path)
    else:
        file_paths = [
            os.path.join(args.datasets, f)
            for f in os.listdir(args.datasets)
            if f.endswith(".pickle") and os.path.isfile(os.path.join(args.datasets, f))
        ]

    assert len(file_paths) > 0, f'no dataset files found'

    if args.number_seeds == 0:
        seeds = [876]
    else:
        seeds = [random.randint(0, 100000) for _ in range(args.number_seeds)]

    for file_path in file_paths:
        name = Path(file_path).stem
        print(f'reading {name}')
        if name == 'time_dict' or name == 'rna' or name == 'f_mnist_70k':
            continue

        for seed in seeds:
            with open(file_path, 'rb') as f:
                data_object = pickle.load(f)
                data_object = preprocess_object(data_object)

            if args.overwrite_existing == 0:
                tmp_file_name = f'random_iter_{data_object.shape}_{args.max_tries}_length_{args.pipeline_length}_strat_{args.strategy}_seed_{seed}.csv'

                tmp_path = os.path.join(args.output_folder, tmp_file_name)
                if os.path.isfile(tmp_path):
                    print(f'skipping {tmp_file_name}')
                    continue

            data_object.p_missing_values = 0
            result_dict = main(
                data_object,
                time_budget=args.time_budget,
                max_iterations=args.max_tries,
                pipeline_length_range=args.pipeline_length,
                strategy=args.strategy,
                seed=seed,
            )
            result_file_name = f'random_iter_{result_dict["dataset_name"]}_{args.max_tries}_length_{args.pipeline_length}_strat_{args.strategy}_seed_{seed}.csv'
            result_path = os.path.join(args.output_folder, result_file_name)
            result_df = pd.DataFrame([result_dict])
            result_df.to_csv(result_path)
