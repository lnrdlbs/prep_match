import argparse
import builtins
from itertools import product
from typing import Any
import warnings
import os
import time
import datetime as _dt
import json as _json
import uuid as _uuid
import logging as _logging
from functools import partial
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from deap import creator, base, tools
from data import Data
from helper import (
    filename_contains_hash,
    get_data_objects,
    get_run_hash,
    write_deap_result2,
    get_ari,
    get_ami,
    get_nmi,
    write_missing_value_imputed_to_pickle,
)

from meta_features import get_all_meta_features
from smart_encoder import SmartEncoder
warnings.filterwarnings("ignore", message="Early stopping criterion not reached.")


parser = argparse.ArgumentParser()

parser.add_argument('-d', '--data_folder', type=str, required=True,
                help='path to the folder of the input data')
parser.add_argument('-o', '--output_folder', type=str, required=True,
                help='path to the folder of the output data')
parser.add_argument('-l', '--log_folder', type=str, required=True,
                help='path to the log folder')

parser.add_argument('-io', '--imputed_output', type=str, required=True,
                help='path for the imputed data')

parser.add_argument('-e', '--eval', choices=['nmi', 'ami', 'ari'], default='ari', help='Evaluation metric')


parser.add_argument('-m', '--mutpb', type=float, default=[0.2], nargs='+',
                    help='Probability of mutation (if multiple values, one run per value)')
parser.add_argument('-c', '--cxpb', type=float, default=[0.5], nargs='+',
                    help='Probability of crossover (if multiple values, one run per value)')
parser.add_argument('-i', '--indpb', type=float, default=[0.2], nargs='+',
                    help='Probability of individual selected for mutation (if multiple values, one run per value)')
parser.add_argument('-ts', '--tourn_size', type=int, default=[5], nargs='+',
                    help='Tournament Size for selection process')
parser.add_argument('-s', '--seed', type=int, default=[876], nargs='+',
                    help='seed for reproducibility (if multiple values, one run per value)')
parser.add_argument('-t', '--time-budget', type=int, default=[600], nargs='+',
                    help='time budget per dataset (if multiple values, one run per value)')

parser.add_argument('-oe', '--overwrite_existing', type=int, default=0,
                    help='defines if existing files should be overwritten if not 1, ' \
                    'the optimizer will skip already evaluated configs within the same output folder')


parser.add_argument('-g', '--generations', type=int, default=[10], nargs='+',
                help='generations per dataset (if multiple values, one run per value)')
parser.add_argument('-p', '--population_size', type=int, default=[20], nargs='+',
                help='population size per dataset (if multiple values, one run per value)')
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
    "simple": {"strategy": ["mean", "median", "most_frequent"]},
    "knn": {"n_neighbors": range(1, 15)},
}

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax) # pylint: disable=no-member
toolbox = base.Toolbox()


def select_method():
    methods = list(pp_methods.keys())
    return np.random.choice(methods)


def init_hyperparameters(method):
    hps = {}
    for param, v in pp_methods[method].items():
        hps[param] = (
            np.random.choice(v)
            if isinstance(v, list)
            else np.random.randint(v.start, v.stop)
        )
    return hps


def create_individual():
    method = select_method()
    hps = init_hyperparameters(method=method)
    return creator.Individual([method, hps]) # pylint: disable=no-member


def main(data_object: Data):

    if data_object.metric == 'ari':
        data_object.base_ari = get_ari(
            data_object.X_base, data_object.y, data_object.n_clusters
        )
        emit('optimization', msg='base ari', ari=data_object.base_ari, console=True)

        data_object.modified_ari = get_ari(
            data_object.X_modified, data_object.y_modified, data_object.n_clusters
        )
        emit('optimization', msg='modified ari', ari=data_object.modified_ari, console=True)
    elif data_object.metric == 'nmi':
        data_object.base_ari = get_nmi(
            data_object.X_base, data_object.y, data_object.n_clusters
        )
        emit('optimization', msg='base nmi', nmi=data_object.modified_ari, console=True)

        data_object.modified_ari = get_nmi(
            data_object.X_modified, data_object.y_modified, data_object.n_clusters
        )
        emit('optimization', msg='modified nmi', nmi=data_object.modified_ari, console=True)
    elif data_object.metric == 'ami':
        data_object.base_ari = get_ami(
            data_object.X_base, data_object.y, data_object.n_clusters
        )
        emit('optimization', msg='base ami', ami=data_object.modified_ari, console=True)

        data_object.modified_ari = get_ami(
            data_object.X_modified, data_object.y_modified, data_object.n_clusters
        )
        emit('optimization', msg='modified ami', ami=data_object.modified_ari, console=True)
    else: raise ValueError(f'Unknown metric {data_object.metric}')


    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual) # pylint: disable=no-member
    toolbox.register("evaluate", partial(evaluate, data_object))
    # toolbox.register("mate", custom_crossover)
    toolbox.register("mate", crossover_only_params)
    toolbox.register("mutate", mutate_only_params, data_object.indpb)
    # toolbox.register("mutate", custom_mutate)
    toolbox.register("select", tools.selTournament, tournsize=data_object.tourn_size)

    hof = tools.HallOfFame(data_object.population_size)
    population = toolbox.population(n=data_object.population_size) # pylint: disable=no-member

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

    emit('optimization', msg='evaluated individuals', number_individuals=len(population), console=True)

    fits = [ind.fitness.values[0] for ind in population]
    hof.update(population)
    
    g = 0

    budget_measure_start = time.time()
    stopped_by_budget = False
    while max(fits) < 1.0 and g < data_object.n_generation:
        if time.time() - budget_measure_start >= data_object.time_budget:
            emit('optimization', msg='time budget achieved', generation=g,time_budget=data_object.time_budget, console=True)
            stopped_by_budget = True
            break
        g = g + 1
        emit('optimization', msg='starting generation', generation=g, console=True)

        offspring = toolbox.select(population, len(population)) # pylint: disable=no-member
        offspring = list(map(toolbox.clone, offspring)) # pylint: disable=no-member

        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            if np.random.uniform(0, 1) < data_object.cxpb:
                toolbox.mate(child1, child2) # pylint: disable=no-member
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if np.random.uniform(0, 1) < data_object.mutpb:
                toolbox.mutate(mutant) # pylint: disable=no-member
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

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


    data_object.stopped_by_budget = stopped_by_budget

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

    emit('status', msg='optimization for dataset done')
    return data_object


def mutate_only_params(indpb, individual):

    method, params = individual[0], individual[1]

    for param in params.keys():
        if np.random.uniform(0, 1) < indpb:

            valid_values = pp_methods[method][param]
            if isinstance(valid_values, list):
                new_value = np.random.choice(
                    [v for v in valid_values if v != params[param]]
                )
            else:
                new_value = np.random.randint(valid_values.start, valid_values.stop - 1)

            params[param] = new_value
    individual[1] = params
    return individual


def crossover_only_params(ind1, ind2):

    if ind1[0] != ind2[0]:
        return ind1, ind2

    method = ind1[0]
    param_keys = list(pp_methods[method].keys())
    if len(param_keys) == 1:
        ind1[1][param_keys[0]], ind2[1][param_keys[0]] = (
            ind2[1][param_keys[0]],
            ind1[1][param_keys[0]],
        )
    else:
        crossover_point = np.random.randint(1, len(param_keys) - 1)
        for key in param_keys[crossover_point:]:
            ind1[1][key], ind2[1][key] = ind2[1][key], ind1[1][key]

    return ind1, ind2

def evaluate(data_object: Data, individual):

    method = individual[0]
    params = individual[1]

    if method == "knn":
        imputer = KNNImputer(n_neighbors=params["n_neighbors"])
    elif method == "simple":
        imputer = SimpleImputer(strategy=params["strategy"])
    else:
        raise ValueError(f"Invalid imputation method: {method}")

    data_object.X_preprocessed = imputer.fit_transform(data_object.X_missing_values)
    data_object.y_preprocessed = data_object.y_missing_values.copy()

    if data_object.metric == 'ari':
        data_object.preprocessed_ari = get_ari(
            data_object.X_preprocessed,
            data_object.y_preprocessed,
            data_object.n_clusters,
        )
    elif data_object.metric == 'nmi':
        data_object.preprocessed_ari = get_nmi(
            data_object.X_preprocessed,
            data_object.y_preprocessed,
            data_object.n_clusters,
        )
    elif data_object.metric == 'ami':
        data_object.preprocessed_ari = get_ami(
            data_object.X_preprocessed,
            data_object.y_preprocessed,
            data_object.n_clusters,
        )
    return data_object.preprocessed_ari


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
    print(param_grid.values())
    for v in product(*param_grid.values()):
        args_combo = dict(zip(param_grid.keys(), v))

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
            data.tourn_size = args_combo['tourn_size']
            data.seed = args_combo['seed']
            data.time_budget = args_combo['time_budget']
            data.init_strategy = None
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
            i = i + 1
            if data.p_missing_values == 0:
                continue
            start = time.time()
            print(f"-------------------------------------------------------\nDataset {i} of {len(data_objects)}\nId {data.id} {data.shape} {hash_to_check}\n-------------------------------------------------------")
            result_object = main(data_object=data)
            emit('status', msg='calculate meta features', console=True)
            result_object.mf_preprocessed = get_all_meta_features(
                X=result_object.X_preprocessed
            )
            end = time.time()
            data.runtime = end-start
            write_missing_value_imputed_to_pickle(
                    data_object=result_object, base_path=args.imputed_output
                )
            write_deap_result2(data, args.output_folder)


