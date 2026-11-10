import argparse
import ast
import copy
from functools import partial
from itertools import product

import builtins
import datetime as _dt
import json as _json
import pdb
import uuid as _uuid
import logging as _logging
import os
from typing import Any
import pandas as pd

import random
import time
import warnings
import numpy as np
import scipy
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import SGDOneClassSVM
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.neighbors import LocalOutlierFactor
from sklearn.random_projection import GaussianRandomProjection
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
)
from deap import creator, base, tools
from data import Data
from helper import (
    filename_contains_hash,
    get_best_params,
    get_data_object_by_uuid,
    get_data_objects,
    get_methods_of_pp_dict,
    get_random_method,
    get_random_params,
    get_run_hash,
    preprocess_object,
    random_method_selection_possible,
    replace_nan_with_constant,
    write_deap_result2,
    get_ari,
    get_nmi,
    get_ami,
)

from meta_features import get_all_meta_features
from smart_encoder import SmartEncoder

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--data_folder', type=str, required=True,
                    help='path to the folder of the input data')
parser.add_argument('-o', '--output_folder', type=str, required=True,
                    help='path to the folder of the output data')
parser.add_argument('-if', '--imputed_folder', type=str, required=False, default=None,
                    help='path to the folder where already imputed data is saved (input)')
parser.add_argument('-l', '--log_folder', type=str, required=True,
                    help='path to the log folder')
parser.add_argument('-e', '--eval', choices=['nmi', 'ami', 'ari'], default='ari', help='Evaluation metric')
parser.add_argument('-m', '--mutpb', type=float, default=[0.2], nargs='+',
                    help='Probability of mutation (if multiple values, one run per value)')
parser.add_argument('-c', '--cxpb', type=float, default=[0.5], nargs='+',
                    help='Probability of crossover (if multiple values, one run per value)')
parser.add_argument('-i', '--indpb', type=float, default=[0.2], nargs='+',
                    help='Probability of individual selected for mutation (if multiple values, one run per value)')
parser.add_argument('-s', '--seed', type=int, default=[876], nargs='+',
                    help='seed for reproducibility (if multiple values, one run per value)')
parser.add_argument('-is', '--init_strategy', choices=['random', 'dc_based'], default='dc_based',
                    help='pipeline initialization strategy')
parser.add_argument('-t', '--time_budget', type=int, default=[600], nargs='+',
                    help='time budget per dataset (if multiple values, one run per value)')
parser.add_argument('-g', '--generations', type=int, default=[50], nargs='+',
                    help='generations per dataset (if multiple values, one run per value)')
parser.add_argument('-p', '--population_size', type=int, default=[20], nargs='+',
                    help='population size per dataset (if multiple values, one run per value)')
parser.add_argument('-ts', '--tourn_size', type=int, default=[5], nargs='+',
                    help='Tournament selection sie (if multiple values, one run per value)')
parser.add_argument('-oe', '--overwrite_existing', type=int, default=0,
                    help='defines if existing files should be overwritten if not 1, the optimizer will skip already evaluated configs within the same output folder')

args = parser.parse_args()



_run_id = _uuid.uuid4().hex
_ts     = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

os.makedirs(args.log_folder, exist_ok=True)
_log_path = os.path.join(args.log_folder, f"{_ts}_{_run_id}.jsonl")

_logging.basicConfig(
    filename=_log_path,
    filemode="w",
    level=_logging.INFO,
    format="%(message)s",
)


_orig_print = builtins.print

def _dumps(payload: dict) -> str:
    return _json.dumps(payload, cls=SmartEncoder, ensure_ascii=False)

def _json_print(*print_args: Any, **kwargs: Any) -> None:
    """
    Replacement for built-in print():
    • writes a JSON object to the per-run file
    • keeps normal behavior on stdout/stderr
    """
    msg = " ".join(str(a) for a in print_args)
    _logging.info(
        _dumps(
            {
                "t"       : _dt.datetime.now().isoformat(timespec="seconds"),
                "category": "stdout",
                "message" : msg,
            }
        )
    )

    _orig_print(*print_args, **kwargs)

builtins.print = _json_print


def emit(tag: str, console:bool = False, **payload: Any) -> None:
    """
    Emit a typed event without disturbing stdout.

    Example:
        emit("fitness", generation=g, best=max(fits))
    """
    entry = {
        "t"       : _dt.datetime.now().isoformat(timespec="seconds"),
        "category": tag,
        **payload,
    }
    json_text = _dumps(entry)
    _logging.info(json_text)

    if console:
        _orig_print(json_text)

pp_methods = {
    "outlier": {
        "lof": {
            "p_neighbors": [0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "p_leaf_size": [1, 5, 10, 20, 30, 50],
            "p": [1, 2, 5, 10],
            "contamination": ["auto", 0.05, 0.1, 0.2, 0.3, 0.4],
        },
        "iso_forest": {
            "n_estimators": [10, 50, 100, 200, 500],
            "max_features": [2, 5, 10, 25, 35],
            "bootstrap": [True, False],
            "warm_start": [True, False],
        },
        "one_class_svm": {
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "gamma": ["scale", "auto"],
            "nu": [0.3, 0.5, 0.7],
            "shrinking": [True, False],
            "max_iter": [100, 500, 1000],
        },
        "elliptic": {
            "assume_centered": [True, False],
            "contamination": [0.1, 0.2, 0.3, 0.4, 0.5],
        },
        "sdg_one_class_svm": {
            "learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
            "warm_start": [True, False],
            "nu": [0.3, 0.5, 0.7],
            "max_iter": [100, 500, 1000],
            "eta0": [0.1, 0.2, 0.3],
        },
    },
    "scaling": {
        "power_transformer": {"standardize": [True, False]},
        "robust": {
            "with_centering": [True, False],
            "with_scaling": [True, False],
            "unit_variance": [True, False],
        },
        "standard_scaler": {"with_mean": [True, False], "with_std": [True, False]},
        "min_max": {"placeholder": ["placeholder"]},
        "quantile": {
            "p_quantiles": [0.3, 0.5, 0.7, 1],
            "output_distribution": ["uniform", "normal"],
        },
        "max_abs": {"placeholder": ["placeholder"]},
    },
    "dim_red": {
        "pca": {
            "n_components": ["mle", 2, 5, 10, 25, 35],
            "whiten": [True, False],
            "svd_solver": ["auto", "full", "arpack", "randomized"],
            "power_iteration_normalizer": ["auto", "QR", "LU"],
        },
        "grp": {"n_components": ["auto", 2, 5, 10, 25, 35], "eps": [0.01, 0.1, 0.3]},
        "lle": {
            "p_neighbors": [0.01, 0.05, 0.1],
            "n_components": [2, 5, 10, 25, 35],
            "eigensolver": ["auto", "arpack", "dense"],
            "max_iter": [50, 100, 500, 1000],
            "method": ["standard", "hessian", "modified", "ltsa"],
            "neighbors_algorithm": ["auto", "brute", "kd_tree", "ball_tree"],
        },
    },
}

pp_methods_flat = {
    "lof": {
        "p_neighbors": [0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "p_leaf_size": [1, 5, 10, 20, 30, 50],
        "p": [1, 2, 5, 10],
        "contamination": ["auto", 0.05, 0.1, 0.2, 0.3, 0.4],
    },
    "iso_forest": {
        "n_estimators": [10, 50, 100, 200, 500],
        "max_features": [2, 5, 10, 25, 35],
        "bootstrap": [True, False],
        "warm_start": [True, False],
    },
    "one_class_svm": {
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "gamma": ["scale", "auto"],
        "nu": [0.3, 0.5, 0.7],
        "shrinking": [True, False],
        "max_iter": [100, 500, 1000],
    },
    "elliptic": {
        "assume_centered": [True, False],
        "contamination": [0.1, 0.2, 0.3, 0.4, 0.5],
    },
    "sdg_one_class_svm": {
        "learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
        "warm_start": [True, False],
        "nu": [0.3, 0.5, 0.7],
        "max_iter": [100, 500, 1000],
        "eta0": [0.1, 0.2, 0.3],
    },
    "power_transformer": {"standardize": [True, False]},
    "robust": {
        "with_centering": [True, False],
        "with_scaling": [True, False],
        "unit_variance": [True, False],
    },
    "standard_scaler": {"with_mean": [True, False], "with_std": [True, False]},
    "min_max": {"placeholder": ["placeholder"]},
    "quantile": {
        "p_quantiles": [0.3, 0.5, 0.7, 1],
        "output_distribution": ["uniform", "normal"],
    },
    "max_abs": {"placeholder": ["placeholder"]},
    "pca": {
        "n_components": ["mle", 2, 5, 10, 25, 35],
        "whiten": [True, False],
        "svd_solver": ["auto", "full", "arpack", "randomized"],
        "power_iteration_normalizer": ["auto", "QR", "LU"],
    },
    "grp": {"n_components": ["auto", 2, 5, 10, 25, 35], "eps": [0.01, 0.1, 0.3]},
    "lle": {
        "p_neighbors": [0.01, 0.05, 0.1],
        "n_components": [2, 5, 10, 25, 35],
        "eigensolver": ["auto", "arpack", "dense"],
        "max_iter": [50, 100, 500, 1000],
        "method": ["standard", "hessian", "modified", "ltsa"],
        "neighbors_algorithm": ["auto", "brute", "kd_tree", "ball_tree"],
    },
}

creator.create("FitnessMax", base.Fitness, weights=(1.0,))

creator.create("Individual", list, fitness=creator.FitnessMax, id=None, parent_ids=list) # pylint: disable=no-member
toolbox = base.Toolbox()


def select_methods(data_object: Data, init_strategy: str = 'dc_based'):

    methods = []
    if init_strategy == 'dc_based':
        if data_object.has_outlier:
            methods.append(np.random.choice(list(pp_methods["outlier"].keys())))

        if data_object.has_scaling:
            methods.append(np.random.choice(list(pp_methods["scaling"].keys())))
        if not data_object.has_outlier and not data_object.has_scaling:
            tmp = [
                np.random.choice(list(pp_methods["outlier"].keys())),
                np.random.choice(list(pp_methods["scaling"].keys())),
                np.random.choice(list(pp_methods["dim_red"].keys())),
            ]
            methods.append(np.random.choice(tmp))
    elif init_strategy == 'random':

        length = np.random.randint(low=1, high=5)
        methods = np.random.choice(
            list(pp_methods_flat.keys()), length, replace=False
        ).tolist()
    else:
        raise ValueError(f'Unknown init strategy: {init_strategy}')
    return methods


def init_hyperparameters(method, data_object: Data):

    hps = {}
    if method in pp_methods["outlier"].keys():

        for param, values in pp_methods["outlier"][method].items():
            if method == "iso_forest" and param == "max_features":
                hps[param] = mutate_iso_max_features(data_object=data_object)
            hps[param] = (
                np.random.choice(values)
                if isinstance(values, list)
                else np.random.randint(values.start, values.stop)
            )

    elif method in pp_methods["scaling"].keys():
        if method == "min_max" or method == "max_abs":
            hps = {}
            return hps

        for param, values in pp_methods["scaling"][method].items():
            hps[param] = (
                np.random.choice(values)
                if isinstance(values, list)
                else np.random.randint(values.start, values.stop)
            )
    elif method in pp_methods["dim_red"].keys():
        for param, values in pp_methods["dim_red"][method].items():
            if method == "pca":
                if param == "n_components":
                    hps[param] = mutate_pca_n_comb(data_object=data_object)
                elif param == "svd_solver":
                    hps[param] = mutate_svd_solver(hps=hps, method=method)
                else:
                    hps[param] = (
                        np.random.choice(values)
                        if isinstance(values, list)
                        else np.random.randint(values.start, values.stop)
                    )
            elif param == "n_components":
                hps[param] = mutate_n_components(data_object=data_object, method=method)
            else:
                hps[param] = (
                    np.random.choice(values)
                    if isinstance(values, list)
                    else np.random.randint(values.start, values.stop)
                )

    return hps


def create_individual(data_object: Data, init_strategy: str = 'dc_based'):

    methods = select_methods(data_object=data_object, init_strategy=init_strategy)

    hps = {}
    for m in methods:
        hp = init_hyperparameters(method=m, data_object=data_object)
        hps[m] = hp

    np.random.shuffle(methods)

    order = [(i + 1, methods[i]) for i in range(len(methods))]
    ind = creator.Individual([methods, hps, order])# pylint: disable=no-member
    ind.id = str(_uuid.uuid4())
    ind.parent_id = []
    return ind


def main(
    data_object: Data,
    time_budget: int,
    init_strategy: str = 'dc_based',
):

    if data_object.metric == 'ari':
        data_object.base_ari = get_ari(
            data_object.X_base, data_object.y, data_object.n_clusters
        )
        emit('optimization', msg='base ari', ari=data_object.base_ari, console=True)
        data_object.modified_ari = get_ari(
            data_object.X_modified, data_object.y_modified, data_object.n_clusters
        )
        emit('optimization', msg='modified ari', ari=data_object.modified_ari, console=True)
    elif data_object.metric == 'ami':
        data_object.base_ari = get_ami(
            data_object.X_base, data_object.y, data_object.n_clusters
        )

        emit('optimization', msg='base ami', ami=data_object.modified_ari, console=True)
        data_object.modified_ari = get_ami(
            data_object.X_modified, data_object.y_modified, data_object.n_clusters
        )
        emit('optimization', msg='modified ami', ami=data_object.modified_ari, console=True)
    elif data_object.metric == 'nmi':
        data_object.base_ari = get_nmi(
            data_object.X_base, data_object.y, data_object.n_clusters
        )
        emit('optimization', msg='base nmi', nmi=data_object.modified_ari, console=True)
        data_object.modified_ari = get_nmi(
            data_object.X_modified, data_object.y_modified, data_object.n_clusters
        )
        emit('optimization', msg='modified nmi', nmi=data_object.modified_ari, console=True)

    
    toolbox.register("individual", partial(create_individual), data_object, init_strategy)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual) # pylint: disable=no-member
    toolbox.register("evaluate", partial(evaluate, data_object))
    toolbox.register("mate", partial(custom_crossover, data_object))
    # toolbox.register("mate", crossover_only_params)
    # toolbox.register("mutate", mutate_only_params, INDPB)

    print(f"has nan's {np.isnan(data_object.X_modified).any()}")

    hof = tools.HallOfFame(data_object.population_size)
    g = 0

    
    population = toolbox.population(n=data_object.population_size) # pylint: disable=no-member
    toolbox.register("mutate", partial(custom_mutate, hof, data_object))
    toolbox.register("select", tools.selTournament, tournsize=data_object.tourn_size)
    
    #toolbox.register("select", custom_select, g)
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()

    logbook.header = ["gen"] + stats.fields

    fitness = list(map(toolbox.evaluate, population)) # pylint: disable=no-member
    for ind, fit in zip(population, fitness):
        ind.fitness.values = (fit,)

    emit('optimization', msg='evaluated individuals', number_individuals=len(population), console=True )

    fits = [ind.fitness.values[0] for ind in population]
    hof.update(population)

    emit(tag='initial_pipelines', pipelines=population, console=True)
    budget_measure_start = time.time()
    stopped_by_budget = False
    while max(fits) < 1.0 and g < data_object.n_generation:
        if time.time() - budget_measure_start >= time_budget:
            emit('optimization', msg='time_budget achieved', time_budget=time_budget, done_generation=g, console=True)
            stopped_by_budget = True
            break
        g = g + 1
        emit('status', msg='start of generation', nr_generation=g, console=True)

        offspring = toolbox.select(population, len(population)) # pylint: disable=no-member
        offspring = list(map(toolbox.clone, offspring)) # pylint: disable=no-member

        hof_list = list(population)
        hof_list.sort(key= lambda ind: ind.fitness.values[0], reverse=True)
        rank_map ={ind.id: rank+1 for rank, ind in enumerate(hof_list)}
       

        # emit(
        #     "selection",
        #     generation=g,
        #     selected_ids=[ind.id for ind in offspring],
        #     selected_pipelines=[{ind.id: ind, 'fitness': ind.fitness.values[0], 'parent_ids': ind.parent_ids, 'rank':rank_map.get(ind.id, None) } for ind in offspring],
        #     console=True
        # )
        # for o in offspring:
        #     valid= validate_individual(o)
        #     if not valid == True:
        #         print('NOT VALID START')
        #         print(o)
        #         sys.exit(1)
        #     else: print('VALID START')

        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            if np.random.uniform(0, 1) < data_object.cxpb:
                toolbox.mate(child1, child2) # pylint: disable=no-member
                del child1.fitness.values
                del child2.fitness.values

        # for o in offspring:
        #     valid= validate_individual(o)
        #     if not valid == True:
        #         print('NOT VALID AFTER CROSS OVER')
        #         print(o)
        #         sys.exit(1)
        #     else: print('VALID CROSS OVER')

        for mutant in offspring:

            if np.random.uniform(0, 1) < data_object.mutpb:

                toolbox.mutate(mutant) # pylint: disable=no-member
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        # for o in offspring:
        #     valid= validate_individual(o)
        #     if not valid == True:
        #         print('NOT VALID AFTER MUTANT')
        #         print(o)
        #         sys.exit(1)
        #     else: print('VALID MUTANT')

        fitnesses = map(toolbox.evaluate, invalid_ind) # pylint: disable=no-member
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = (fit,)

        population[:] = offspring
        hof.update(population)
        record = stats.compile(population)
        logbook.record(gen=g, **record)
        emit('optimization', msg='current progress',
             generation=logbook[-1]['gen'], min=logbook[-1]['min'], max=logbook[-1]['max'], avg=logbook[-1]['avg'], console=True)
        fits = [ind.fitness.values[0] for ind in population]

    data_object.generations = logbook.select("gen")
    data_object.deap_min = logbook.select("min")
    data_object.deap_max = logbook.select("max")
    data_object.deap_avg = logbook.select("avg")

    if stopped_by_budget:
        data_object.stopped_by_budget = True
    else:
        data_object.stopped_by_budget = False

    if len(hof) > 0:
        data_object.best_pipeline = hof[0]
        data_object.best_fitness = hof[0].fitness.values[0]
    else:
        data_object.best_pipeline = None
        data_object.best_fitness = None
    if len(hof) >= 5:
        top_5 = hof[:5]
        data_object.top_5_pipeline = top_5
        data_object.top_5_fitness = [f.fitness.values[0] for f in top_5]
    else:
        data_object.top_5_pipeline = None
        data_object.top_5_fitness = None
    emit('status', msg='optimization for dataset done', console=True)
    return data_object

# def custom_select(g, population, k):
#     selected = tools.selTournament(population, k, tournsize=5)
#     emit(
#         "selection",
#         generation=g,
#         selected_ids=[ind.id for ind in selected],
#         selected_pipelines=[{ind.id: ind, 'parent_ids': ind.parent_ids} for ind in selected],
#         console=True
#     )
#     #pdb.set_trace()
#     return selected


def custom_mutate(hof, data_object: Data, individual):

    old_ind = copy.deepcopy(individual)
    mutation_operation = ''
    mutation_has_happened = False
    # decision if method, params, or oder mutate
    mutation_objective = np.random.randint(0, 3)

    if mutation_objective == 0:
        # decision if method add, change or remove
        # 0 = add, 1 = change, 2 = remove
        method_mutation_objective = -1
        if len(individual[0]) > 1:
            method_mutation_objective = mutation_objective = np.random.randint(0, 3)
        else:
            method_mutation_objective = mutation_objective = np.random.randint(0, 2)

        if method_mutation_objective == 0:
            # add if possible
            all_methods = get_methods_of_pp_dict(pp_methods)
            current_methods = individual[0]
            if random_method_selection_possible(
                all_methods, current_methods=current_methods
            ):
                new_method = get_random_method(all_methods, current_methods)
                params, current_fitness = get_best_params(hof, new_method)
                if current_fitness == -1:
                    params = get_random_params(new_method, pp_methods_flat)

                individual[0].append(new_method)
                individual[1][new_method] = params
                # for later understanding: this is an example individual
                # [['iso_forest'], {'iso_forest': {'n_estimators': 500, 'max_features': 2}},
                # [(1, 'iso_forest')]]

                individual[2].append((individual[2][-1][0] + 1, new_method))
                mutation_has_happened = True
                mutation_objective = 'method_add'
        elif method_mutation_objective == 1:
            # change random method
            all_methods = get_methods_of_pp_dict(pp_methods)
            method_to_change = np.random.choice(individual[0])
            if random_method_selection_possible(all_methods, current_methods=individual[0]):
                new_method = get_random_method(all_methods, individual[0])
                params = get_random_params(new_method, pp_methods_flat)
                individual[0] = [
                    new_method if item == method_to_change else item
                    for item in individual[0]
                ]
                del individual[1][method_to_change]
                individual[1][new_method] = params
                individual[2] = [
                    (num, new_method) if m == method_to_change else (num, m)
                    for num, m in individual[2]
                ]
                mutation_has_happened = True
                mutation_objective = 'method_change'
        elif method_mutation_objective == 2:
            if len(individual[0]) > 1:
                method_to_delete = np.random.choice(individual[0])
                individual[0].remove(method_to_delete)
                del individual[1][method_to_delete]
                individual[2] = [o for o in individual[2] if o[1] != method_to_delete]
                mutation_has_happened = True
                mutation_objective = 'method_remove'
    elif mutation_objective == 1:

        for method_name, params in individual[1].items():
            if method_name == "min_max" or method_name == "max_abs":
                # has no params
                continue
            else:
                for param_name, _ in params.items():
                    if np.random.uniform(0, 1) < data_object.indpb:
                        if method_name == "pca" and param_name == "n_components":
                            new_value = mutate_pca_n_comb(data_object=data_object)
                        elif method_name == "pca" and param_name == "svd_solver":
                            new_value = mutate_svd_solver(params, method_name)
                        elif (
                            method_name in ["grp", "lle"]
                            and param_name == "n_components"
                        ):
                            new_value = mutate_n_components(
                                data_object=data_object, method=method_name
                            )
                        elif (
                            method_name == "iso_forest" and param_name == "max_features"
                        ):
                            new_value = mutate_iso_max_features(data_object=data_object)
                        else:

                            valid_values = pp_methods_flat[method_name][param_name]
                            if isinstance(valid_values, list):
                                new_value = np.random.choice(
                                    [v for v in valid_values if v != params[param_name]]
                                )
                            else:
                                new_value = np.random.randint(
                                    valid_values.start, valid_values.stop - 1
                                )

                        params[param_name] = new_value
                        mutation_has_happened = True
                        mutation_objective = 'parameter_change'
            individual[1][method_name] = params

    elif mutation_objective == 2:
        if len(individual[2]) > 1:
            random_indices = np.random.choice(len(individual[2]), 2, replace=False)
            tmp_l = [list(tup) for tup in individual[2]]
            tmp_l[random_indices[0]][0], tmp_l[random_indices[1]][0] = (
                tmp_l[random_indices[1]][0],
                tmp_l[random_indices[0]][0],
            )
            individual[2] = [tuple(l) for l in tmp_l]
            mutation_has_happened = True
            mutation_objective = 'order change'
    if mutation_has_happened:
        emit('pipeline_change', type='mutation',  pipeline_id=individual.id, mutation_objective=mutation_objective, old_pipeline = old_ind, new_pipeline=individual, console=True)
    return individual


def custom_crossover(data_object: Data, ind1, ind2):
    old_pipeline_1 = copy.deepcopy(ind1)
    old_pipeline_2 = copy.deepcopy(ind2)
    crossover_has_happened = False
    parents = [old_pipeline_1.id, old_pipeline_2.id]
    methods_in_both = [item for item in ind1[0] if item in ind2[0]]
    if len(methods_in_both) == 0:
        return ind1, ind2

    for mb in methods_in_both:
        if mb == "min_max" or mb == "max_abs":
            continue
        if mb not in methods_in_both:
            continue
        for param in ind1[1][mb].keys():
            if np.random.uniform(0, 1) < data_object.cxpb:
                ind2[1][mb][param], ind2[1][mb][param] = (
                    ind2[1][mb][param],
                    ind1[1][mb][param],
                )
                crossover_has_happened = True
                ind1.parent_ids = parents
                ind2.parent_ids = parents
                ind1.id = str(_uuid.uuid4())
                ind2.id = str(_uuid.uuid4())
    # if crossover_has_happened:
    #     emit('pipeline_change', type='crossover',  parents=parents, children=[ind1.id, ind2.id], old_pipelines = [old_pipeline_1, old_pipeline_2], new_pipelines=[ind1, ind2], console=True)
    return ind1, ind2


def evaluate(data_object: Data, individual):
    #eval_start = time.time()

    data_object.X_preprocessed = data_object.X_modified.copy()
    data_object.y_preprocessed = data_object.y_modified.copy()
    individual[2] = sorted(individual[2], key=lambda x: x[0])
    invalid = False

    try:
        for el in individual[2]:
            method = el[1]

            hps = individual[1][method]
            if method == "lof":
                n_neighbors = int(hps["p_neighbors"] * len(data_object.X_preprocessed))
                if n_neighbors <= 0:
                    n_neighbors = 1
                leaf_size = int(hps["p_leaf_size"] * len(data_object.X_preprocessed))
                if hps["contamination"] == "auto":
                    contamination = "auto"
                else:
                    contamination = float(hps["contamination"])
                try:
                    operator = LocalOutlierFactor(
                        n_neighbors=n_neighbors,
                        leaf_size=leaf_size,
                        p=hps["p"],
                        algorithm=hps["algorithm"],
                        contamination=contamination,
                    )
                    is_inlier = operator.fit_predict(data_object.X_preprocessed)
                    data_object.X_preprocessed = data_object.X_preprocessed[
                        is_inlier == 1
                    ]
                    data_object.y_preprocessed = data_object.y_preprocessed[
                        is_inlier == 1
                    ]
                except ValueError as e:
                    emit('evaluation error', msg='lof error', exception=e, console=True)
                    invalid = True
            elif method == "iso_forest":
                if hps["max_features"] == "auto":
                    max_features = "auto"
                else:
                    max_features = int(hps["max_features"])

                try:
                    operator = IsolationForest(
                        n_estimators=hps["n_estimators"],
                        max_features=max_features,
                        bootstrap=hps["bootstrap"],
                        warm_start=hps["warm_start"],
                        random_state=876,
                    )
                    is_inlier = operator.fit_predict(data_object.X_preprocessed)

                    data_object.X_preprocessed = data_object.X_preprocessed[
                        is_inlier == 1
                    ]
                    data_object.y_preprocessed = data_object.y_preprocessed[
                        is_inlier == 1
                    ]
                except ValueError as e:
                    emit('evaluation error', msg='iso error', exception=e, console=True)
                    invalid = True

            elif method == "power_transformer":
                try:
                    operator = PowerTransformer(standardize=hps["standardize"])
                    data_object.X_preprocessed = operator.fit_transform(
                        data_object.X_preprocessed
                    )
                except scipy.optimize._optimize.BracketError as e:  # pylint: disable=protected-access
                    emit('evaluation error', msg='power transformer error', exception=e, console=True)
                    invalid = True

            elif method == "one_class_svm":
                try:
                    operator = OneClassSVM(
                        kernel=hps["kernel"],
                        gamma=hps["gamma"],
                        nu=hps["nu"],
                        shrinking=hps["shrinking"],
                        max_iter=hps["max_iter"],
                    )
                    is_inlier = operator.fit_predict(data_object.X_preprocessed)
                    data_object.X_preprocessed = data_object.X_preprocessed[
                        is_inlier == 1
                    ]
                    data_object.y_preprocessed = data_object.y_preprocessed[
                        is_inlier == 1
                    ]
                except ValueError as e:
                    emit('evaluation error', msg='one class svm error', exception=e, console=True)
                    invalid = True
            elif method == "elliptic":
                try:
                    operator = EllipticEnvelope(
                        assume_centered=hps["assume_centered"],
                        contamination=hps["contamination"],
                        random_state=876,
                    )
                    is_inlier = operator.fit_predict(data_object.X_preprocessed)
                    data_object.X_preprocessed = data_object.X_preprocessed[
                        is_inlier == 1
                    ]
                    data_object.y_preprocessed = data_object.y_preprocessed[
                        is_inlier == 1
                    ]
                except ValueError as e:
                    emit('evaluation error', msg='elliptic envelope error', exception=e, console=True)
                    invalid = True
                except UnboundLocalError as e:
                    emit('evaluation error', msg='internal elliptic envelope error', exception=e, console=True)
                    invalid = True

            elif method == "sdg_one_class_svm":
                try:
                    operator = SGDOneClassSVM(
                        learning_rate=hps["learning_rate"],
                        warm_start=hps["warm_start"],
                        nu=hps["nu"],
                        max_iter=hps["max_iter"],
                        eta0=hps["eta0"],
                        random_state=876,
                    )
                    is_inlier = operator.fit_predict(data_object.X_preprocessed)
                    data_object.X_preprocessed = data_object.X_preprocessed[
                        is_inlier == 1
                    ]
                    data_object.y_preprocessed = data_object.y_preprocessed[
                        is_inlier == 1
                    ]
                except ValueError as e:
                    emit('evaluation error', msg='sdg envelope error', exception=e, console=True)
                    invalid = True
            elif method == "robust":
                try:
                    operator = RobustScaler(
                        with_centering=hps["with_centering"],
                        with_scaling=hps["with_scaling"],
                        unit_variance=hps["unit_variance"],
                    )
                    data_object.X_preprocessed = operator.fit_transform(
                        data_object.X_preprocessed
                    )
                except ValueError as e:
                    emit('evaluation error', msg='robust scaling error', exception=e, console=True)
                    invalid = True

            elif method == "standard_scaler":
                try:
                    operator = StandardScaler(
                        with_mean=hps["with_mean"], with_std=hps["with_std"]
                    )
                    data_object.X_preprocessed = operator.fit_transform(
                        data_object.X_preprocessed
                    )
                except ValueError as e:
                    emit('evaluation error', msg='standard scaler error', exception=e, console=True)
                    invalid = True
            elif method == "min_max":
                try:
                    operator = MinMaxScaler()
                    data_object.X_preprocessed = operator.fit_transform(
                        data_object.X_preprocessed
                    )
                except ValueError as e:
                    emit('evaluation error', msg='min max error', exception=e, console=True)
                    invalid = True
            elif method == "quantile":
                n_quantiles = int(hps["p_quantiles"] * len(data_object.X_preprocessed))
                try:
                    operator = QuantileTransformer(
                        n_quantiles=n_quantiles,
                        output_distribution=hps["output_distribution"],
                        random_state=876,
                    )
                    data_object.X_preprocessed = operator.fit_transform(
                        data_object.X_preprocessed
                    )
                except ValueError as e:
                    emit('evaluation error', msg='quantile error', exception=e, console=True)
                    invalid = True
            elif method == "max_abs":
                try:
                    operator = MaxAbsScaler()
                    data_object.X_preprocessed = operator.fit_transform(
                        data_object.X_preprocessed
                    )
                except ValueError as e:
                    emit('evaluation error', msg='max abs error', exception=e, console=True)
                    invalid = True
            elif method == "pca":

                if hps["n_components"] == "mle":
                    n_components = "mle"
                else:
                    n_components = int(hps["n_components"])
                try:
                    operator = PCA(
                        n_components=n_components,
                        whiten=hps["whiten"],
                        svd_solver=hps["svd_solver"],
                        power_iteration_normalizer=hps["power_iteration_normalizer"],
                        random_state=876,
                    )
                    data_object.X_preprocessed = operator.fit_transform(
                        data_object.X_preprocessed
                    )
                except ValueError as e:
                    emit('evaluation error', msg='pca error', exception=e, console=True)
                    invalid = True
                except scipy.sparse.linalg._eigen.arpack.arpack.ArpackError:  # pylint: disable=protected-access
                    emit('evaluation error', msg='pca argpack error', exception=e, console=True)
                    invalid = True

            elif method == "grp":
                try:
                    if hps["n_components"] == "auto":
                        n_components = "auto"
                    else:
                        n_components = int(hps["n_components"])
                    operator = GaussianRandomProjection(
                        n_components=n_components, eps=hps["eps"], random_state=876
                    )

                    data_object.X_preprocessed = operator.fit_transform(
                        data_object.X_preprocessed
                    )
                except ValueError as e:
                    emit('evaluation error', msg='grp error', exception=e, console=True)
                    invalid = True
                except TypeError as e:
                    emit('evaluation error', msg='grp type error', exception=e, console=True)
                    invalid = True

            elif method == "lle":
                n_neighbors = int(hps["p_neighbors"] * len(data_object.X_preprocessed))
                try:
                    operator = LocallyLinearEmbedding(
                        n_neighbors=n_neighbors,
                        n_components=hps["n_components"],
                        eigen_solver=hps["eigensolver"],
                        max_iter=hps["max_iter"],
                        method=hps["method"],
                        neighbors_algorithm=hps["neighbors_algorithm"],
                        random_state=876,
                    )
                    data_object.X_preprocessed = operator.fit_transform(
                        data_object.X_preprocessed
                    )
                except ValueError as e:
                    emit('evaluation error', msg='lle error', exception=e, console=True)
                    invalid = True
            else:
                emit('evaluation error', msg='unknown method', method_name=method, console=True)
                raise ValueError(f"Eval: Unknown method: {method}")

        if np.isnan(data_object.X_preprocessed).any():
            emit('optimization error', msg='nan values introduced', individual=individual, console=True)
            invalid = True

        if len(data_object.X_preprocessed) < data_object.n_clusters:
            emit('optimization error', msg='to few samples', 
                 len_after_preprocessing=len(data_object.X_preprocessed), n_cluster=data_object.n_clusters, console=True)
            invalid = True
        if invalid:
            data_object.preprocessed_ari = "invalid"
            return 0
    except Exception as e:
        emit('evaluation error', msg='unknown error', exception=e, console=True)
        invalid = True
    # eval_end = time.time()
    # print(f'eval time: {eval_end-start}')
    if data_object.metric:
        data_object.preprocessed_ari = get_ari(
            data_object.X_preprocessed,
            data_object.y_preprocessed,
            data_object.n_clusters,
        )
    elif data_object.metric:
        data_object.preprocessed_ari = get_nmi(
            data_object.X_preprocessed,
            data_object.y_preprocessed,
            data_object.n_clusters,
        )
    elif data_object.metric:
        data_object.preprocessed_ari = get_ami(
            data_object.X_preprocessed,
            data_object.y_preprocessed,
            data_object.n_clusters,
        )
    return data_object.preprocessed_ari


def mutate_pca_n_comb(data_object: Data):
    max_com = min(data_object.n_samples, data_object.n_features)
    options = ["comp", "mle"]
    choice = np.random.choice(options, p=[0.8, 0.2])

    if choice == "comp":
        if max_com <= 2:
            return 2
        return np.random.randint(2, max_com)
    elif choice == "mle":
        return "mle"


def mutate_n_components(data_object: Data, method):
    valid_values = pp_methods["dim_red"][method]["n_components"]
    c = np.random.choice(valid_values)
    if c == "auto":
        return c
    c = int(c)
    if c > len(data_object.X_modified[0]):
        return len(data_object.X_modified[0])
    return c


def mutate_svd_solver(hps: dict, method: str):
    all_solvers = pp_methods["dim_red"][method]["svd_solver"].copy()
    if hps["n_components"] == "mle":
        return "auto"
    else:
        return random.choice(all_solvers)


def mutate_iso_max_features(data_object: Data):

    available_values = pp_methods["outlier"]["iso_forest"]["max_features"].copy()
    max_f = np.random.choice(available_values)

    #print(f"mut iso {max_f}, {data_object.n_features}, {len(data_object.X_modified[0])}")
    #if data_object.n_features < max_f:
    if len(data_object.X_modified[0]) <= max_f:
        max_f = len(data_object.X_modified[0])
    return max_f


def impute_values(data_object):

    method = data_object.best_pipeline[0]
    params = data_object.best_pipeline[1]
    if method == "knn":
        imputer = KNNImputer(n_neighbors=params["n_neighbors"])
    elif method == "simple":
        imputer = SimpleImputer(strategy=params["strategy"])
    else:
        raise ValueError(f"Invalid imputation method: {method}")

    data_object.X_modified = imputer.fit_transform(data_object.X_missing_values)
    return data_object


def prepare_data(data_object: Data, imputed_path: str):

    data_object_imputed = get_data_object_by_uuid(imputed_path, str(data_object.uuid))
    data_object_imputed = impute_values(data_object=data_object_imputed)
    data_object_imputed.generations = 0
    data_object_imputed.deap_min = 0
    data_object_imputed.deap_max = 0
    data_object_imputed.deap_avg = 0
    data_object_imputed.best_pipeline = 0
    data_object_imputed.best_fitness = 0
    data_object_imputed.top_5_fitness = []
    data_object_imputed.top_5_pipeline = []
    data_object_imputed.stopped_by_budget = False
    

    return data_object_imputed


if __name__ == "__main__":


    param_grid = {
        'mutpb': args.mutpb,
        'cxpb': args.cxpb,
        'indpb': args.indpb,
        'generations': args.generations,
        'population_size': args.population_size,
        'seed': args.seed,
        'time_budget': args.time_budget,
        'tourn_size': args.tourn_size
    }
    os.makedirs(args.output_folder, exist_ok=True)
    for values in product(*param_grid.values()):
        args_combo = dict(zip(param_grid.keys(), values))
        data_objects = get_data_objects(args.data_folder)
        i = 0
        
        for data in data_objects:
            i = i + 1
            data.metric = args.eval
            data.mutpb = args_combo['mutpb']
            data.cxpb = args_combo['cxpb']
            data.indpb = args_combo['indpb']
            data.n_generation = args_combo['generations']
            data.population_size = args_combo['population_size']
            data.seed = args_combo['seed']
            data.time_budget = args_combo['time_budget']
            data.init_strategy = args.init_strategy
            data.X_preprocessed = None
            data.mf_preprocessed = None
            data.mf_modified_relevant = None
            data.y_preprocessed = None
            data.base_ari = -2
            data.modified_ari = -2
            data.preprocessed_ari = -2
            data.generations = []
            data.deap_min = []
            data.deap_avg = []
            data.deap_max = []
            data.best_pipeline = []
            data.best_fitness = 0
            data.top_5_pipeline = []
            data.top_5_fitness = []
            data.runtime = -1
            data.run_hash = None
            data.tourn_size = args_combo['tourn_size']

            hash_to_check = get_run_hash(data)
            if args.overwrite_existing != 1:

                matched = False
                for fn in os.listdir(args.output_folder):
                    full_path = os.path.join(args.output_folder, fn)
                    if os.path.isfile(full_path) and filename_contains_hash(fn, hash_to_check):
                        matched = True
                if matched:
                    emit('skipping', msg=f'Skipping data with id: {data.uuid}, hash: {hash_to_check} and shape: {data.shape} because overwrite existing if False and a result exists in the output folder')
                    continue
            if data.p_missing_values > 0:
                if args.imputed_folder is None:
                    if data.shape == 'blobs':
                        raise ValueError('No imputed input folder provided but data with missing values')
                    else:
                        data.X_modified = replace_nan_with_constant(data.X_modified.copy(), constant=0)
                        data.X_base = replace_nan_with_constant(data.X_base.copy(), constant=0)
                else:
                    data = prepare_data(data, args.imputed_folder)
            if data.shape != 'blobs':
                data = preprocess_object(data)

            if isinstance(data.y, pd.Series):
                data.y= data.y.to_numpy(dtype=int)
            if isinstance(data.y_modified, pd.Series):
                data.y_modified= data.y_modified.to_numpy(dtype=int)
            if isinstance(data.y_missing_values, pd.Series):
                data.y_missing_values= data.y_missing_values.to_numpy(dtype=int)

            emit('data_before_preprocessing', data=data)
            start = time.time()
            print(f"-------------------------------------------------------\nDataset {i} of {len(data_objects)}\nId {data.id} {data.shape} {hash_to_check}\n-------------------------------------------------------")
            print(f'sample size: {data.n_samples}')
            data.time_budget = args_combo['time_budget']
            result_object = main(
                data_object=data,
                init_strategy=args.init_strategy,
                time_budget=args_combo['time_budget']
            )
            emit('status', msg='calculate meta features', console=True)
            try:
                result_object.mf_preprocessed = get_all_meta_features(
                    X=result_object.X_preprocessed
                )
            except ValueError as e:
                emit('error', msg='Error during meta feature calculation', exception=e, console=True)
                result_object.mf_preprocessed = []
            #write_deap_result("output/test_single.csv", data_object)
            end = time.time()
            data.runtime = end-start
            emit('status', 'write result')
            write_deap_result2( data, args.output_folder)
            emit('data_after_preprocessing', data=data)


