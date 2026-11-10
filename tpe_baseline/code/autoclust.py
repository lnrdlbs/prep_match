import traceback

import argparse
import json
import multiprocessing
import random
import re
import time
import numpy as np
import pandas as pd
import os
import sys
import operator

from deap import gp, creator, base, tools
from deap.gp import MetaEphemeral

from scipy.sparse.csgraph import connected_components

from sklearn.cluster import DBSCAN, KMeans, AffinityPropagation, AgglomerativeClustering, MeanShift, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA

from sklearn.metrics import adjusted_rand_score, silhouette_score, davies_bouldin_score, calinski_harabasz_score

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.manifold")
warnings.filterwarnings("ignore", message="Graph is not fully connected")
warnings.filterwarnings("ignore", message="Binning data failed")

#Annahme: internen aufbau der pipeline, garantiert valid pipelines über typen (klassen)
class Data(np.ndarray): pass
class Labels(np.ndarray): pass

#kmeans
class K_Clusters(int): pass
class InitType(str): pass
class MaxIter(int): pass 
class NInit(int): pass 
class Tol(float): pass 
class AlgorithmKM(str): pass 

#Standardscaler
class BoolType(str): pass

#Simple Imputer
class Strategy(str):pass

#PCA
class N_Components(int): pass

#DBSCAN
class EPS(float): pass 
class Min_Samples(int): pass
class Metric(str): pass
class Algorithm_DB(str): pass
class Leaf_Size(int): pass

# Affinity Propagation
class ConvIter(int): pass
class Damping(float): pass

# Agglomerative Clustering
class AffinityAC(str): pass
class Linkage(str): pass
class ComputeFullTree(str): pass

# MeanShift
class Bandwidth(float): pass

# Spectral Clustering
class AffinitySC(str): pass
class NNeighbors(int): pass
class AssignLabels(str): pass

# KNNImputer
class N_Neighbors_KNN(int): pass
class Weights(str): pass


def wrapperKmeans(data: Data, k: K_Clusters, init: InitType, m_iter: MaxIter, n_init: NInit, alg: AlgorithmKM, tol: Tol) -> Labels:
    # Annahme: sklearn hat die ursprünglichen parameter ["auto", "full", "elkan"] nicht mehr, full wurde zu lloyd. der default (auto) ist lloyd, auto und full ex nicht mehr
    # "auto" -> "lloyd"
    # "full" -> "lloyd"
    # "elkan" -> "elkan" 
    if alg == "full" or alg == "auto":
        actual_alg = "lloyd" 
    else:
        actual_alg = alg
    return KMeans(n_clusters=k, init=init, max_iter=m_iter, n_init=n_init, algorithm=actual_alg, tol=tol).fit_predict(data)

def wrapperDBSCAN(data: Data, eps: EPS, min_samples: Min_Samples, metric: Metric, algorithm: Algorithm_DB, leaf_size: Leaf_Size) -> Labels:
    return DBSCAN(eps=eps, min_samples=min_samples, metric=metric, algorithm=algorithm, leaf_size=leaf_size).fit_predict(data)

def wrapperAffinityPropagation(data: Data, m_iter: MaxIter, c_iter: ConvIter, damping: Damping) -> Labels:
    return AffinityPropagation(max_iter=m_iter, convergence_iter=c_iter, damping=damping).fit_predict(data)

def wrapperAgglomerative(data: Data, k: K_Clusters, aff: AffinityAC, link: Linkage, tree: ComputeFullTree) -> Labels:

    # Annahme: ward benötigt zwingend euclidean daher prüfung
    actual_aff = "euclidean" if link == "ward" else aff
    t = True if tree == "True" else (False if tree == "False" else "auto")

    # Annahme: neue sklearn version nutz metric statt affinity
    return AgglomerativeClustering(n_clusters=k, metric=actual_aff, linkage=link, compute_full_tree=t).fit_predict(data)

def wrapperMeanShift(data: Data, bin_s: BoolType, cl_all: BoolType, m_iter: MaxIter, band: Bandwidth) -> Labels:
    bs = True if bin_s == "True" else False
    ca = True if cl_all == "True" else False
    return MeanShift(bin_seeding=bs, cluster_all=ca, max_iter=m_iter, bandwidth=band).fit_predict(data)

def wrapperSpectral(data: Data, k: K_Clusters, n_init: NInit, aff: AffinitySC, n_neigh: NNeighbors, lab: AssignLabels) -> Labels:
    return SpectralClustering(n_clusters=k, n_init=n_init, affinity=aff, n_neighbors=n_neigh, assign_labels=lab).fit_predict(data)

def wrapperStandardScaler(data: Data, with_mean: BoolType, with_std: BoolType)-> Data:
    m = True if with_mean == "True" else False
    s = True if with_std == "True" else False
    return StandardScaler(with_mean=m, with_std=s).fit_transform(data)

def wrapperSimpleImputer(data:Data, strategy: Strategy) -> Data:
    return SimpleImputer(strategy=strategy).fit_transform(data)

def wrapperKNNImputer(data: Data, n_neigh: N_Neighbors_KNN, weights: Weights) -> Data:
    return KNNImputer(n_neighbors=n_neigh, weights=weights).fit_transform(data)

def wrapperPCA(data: Data, n_components : N_Components) ->Data:
    current_features = data.shape[1]
    # Suchraum in diesem Fall variabel mit [2,n_components], n_components kann sich zur Laufzeit ändern (bei zwei mal PCA) daher bei diesem Preprocessor explizit der check
    # Annahme: Entweder wenn möglich den Wunschwert oder das Maximum dessen (es findet keine reduktion statt)
    actual_n = min(n_components, current_features)
    # mindestens 2 Komponenten (laut Suchraum)
    if actual_n < 2:
        actual_n = 2
    return PCA(n_components=actual_n).fit_transform(data)


def createPset(n_features):
    pset = gp.PrimitiveSetTyped("AutoclustTest", [Data], Labels)

    pset.addPrimitive(wrapperKmeans, [Data, K_Clusters, InitType, MaxIter, NInit, AlgorithmKM, Tol], Labels, name="KMeans")
    pset.addPrimitive(wrapperDBSCAN, [Data, EPS, Min_Samples, Metric, Algorithm_DB, Leaf_Size], Labels, name="DBSCAN")
    pset.addPrimitive(wrapperAffinityPropagation, [Data, MaxIter, ConvIter, Damping], Labels, name="AffinityPropagation")
    pset.addPrimitive(wrapperAgglomerative, [Data, K_Clusters, AffinityAC, Linkage, ComputeFullTree], Labels, name="AgglomerativeClustering")
    pset.addPrimitive(wrapperMeanShift, [Data, BoolType, BoolType, MaxIter, Bandwidth], Labels, name="MeanShift")
    pset.addPrimitive(wrapperSpectral, [Data, K_Clusters, NInit, AffinitySC, NNeighbors, AssignLabels], Labels, name="SpectralClustering")
    
    pset.addPrimitive(wrapperStandardScaler, [Data, BoolType, BoolType], Data, name="StandardScaler")
    pset.addPrimitive(wrapperPCA, [Data, N_Components], Data, name="PCA")
    pset.addPrimitive(wrapperSimpleImputer, [Data, Strategy], Data, name="SimpleImputer")
    pset.addPrimitive(wrapperKNNImputer, [Data, N_Neighbors_KNN, Weights], Data, name="KNNImputer")


    # KMeans
    for i in ["k-means++", "random"]: 
        clean_name = i.replace("-", "_").replace("+", "plus")
        pset.addTerminal(i, InitType, name=f"init_{clean_name}")

    for m in range(50, 502): pset.addTerminal(m, MaxIter, name=f"max_it_{m}")

    for a in ["auto", "full", "elkan"]: pset.addTerminal(a, AlgorithmKM, name=f"alg_km_{a}")
    
    for k in range(2, 41): pset.addTerminal(k, K_Clusters, name=f"k_{k}")

    for n in range(2, 31): pset.addTerminal(n, NInit, name=f"n_init_{n}")

    # Annahme bei tol: das ist ne Menge [1e−5, 1e−1] und kein Wertebereich, weil 0.1 schritte machen keinen sinn bei Start 10^-5 bis 0.1 (ergibt eig nur 0.1)
    pset.addTerminal(1e-5, Tol, name="tol_1e_5")
    pset.addTerminal(1e-1, Tol, name="tol_1e_1")

    # Scaler / MeanShift Booleans
    for b in ["True", "False"]: pset.addTerminal(b, BoolType, name=f"bool_{b}")

    # DBSCAN
    for i in np.arange(0.1, 30.1, 0.1):
        val = round(i, 1)
        pset.addTerminal(val, EPS, name=f"eps_{str(val).replace('.', '_')}")

    for s in range(2, 21): pset.addTerminal(s, Min_Samples, name=f"min_samples_{s}")

    for m in ["euclidean", "l1", "l2", "manhattan"]: pset.addTerminal(m, Metric, name=f"metric_db_{m}")

    for a in ["auto", "ball_tree", "kd_tree", "brute"]: pset.addTerminal(a, Algorithm_DB, name=f"alg_db_{a.replace('_','')}")

    for l in range(2, 51): pset.addTerminal(l, Leaf_Size, name=f"leaf_{l}")

    # Affinity Propagation
    for c in range(2, 31): pset.addTerminal(c, ConvIter, name=f"conv_it_{c}")

    # Annhame: dadurch dass die floats im suchraum in eckigen klammern angegeben werden ist es eine menge mit zwei möglichkeiten, man kommt mit 0,1ser schritten sonst auch nicht auf 0.999
    pset.addTerminal(0.5, Damping, name="damping_0_5")
    pset.addTerminal(0.999, Damping, name="damping_0_999")

    # Agglomerative
    for a in ["euclidean", "l1", "l2", "manhattan", "cosine"]: pset.addTerminal(a, AffinityAC, name=f"aff_ac_{a}")
    for l in ["ward", "complete", "average", "single"]: pset.addTerminal(l, Linkage, name=f"link_{l}")
    for t in ["auto", "True", "False"]: pset.addTerminal(t, ComputeFullTree, name=f"tree_{t}")

    # MeanShift
    for b in np.arange(0.1, 2.6, 0.1):
        val = round(b, 1)
        pset.addTerminal(val, Bandwidth, name=f"band_{str(val).replace('.', '_')}")

    # Spectral
    for a in ["nearest_neighbors", "rbf"]: pset.addTerminal(a, AffinitySC, name=f"aff_sc_{a.replace('_','')}")
    for n in range(2, 41): pset.addTerminal(n, NNeighbors, name=f"n_neigh_{n}")
    for l in ["kmeans", "discretize"]: pset.addTerminal(l, AssignLabels, name=f"labels_{l}")

    # PCA
    for c in range(2, n_features + 1): pset.addTerminal(c, N_Components, name=f"n_components_{c}")

    # Simple Imputer
    for y in ["mean", "median", "most_frequent", "constant"]: pset.addTerminal(y, Strategy, name=f"strategy_{y}")

    # KNNImputer
    for n in range(5, 1001): pset.addTerminal(n, N_Neighbors_KNN, name=f"knn_n_{n}")
    for w in ["uniform", "distance"]: pset.addTerminal(w, Weights, name=f"weights_{w}")


    pset.renameArguments(ARG0="matrix")
    return pset



#Annahme: Es wrd die genGrow Methode verwendet und generate() in umgwandelt verwendet um error zu umgehen
def safe_generate(pset, min_, max_, condition, type_=None):
    # einach generate() methode von DEAP mit extra dass wenn mal kein element von dem gewürfelten (primitive/terminal) hat dann das andere erzwingt umfehler zu vermeiden
    """The tree is built from the root to the leaves, and it stops growing
    the current branch when the *condition* is fulfilled: in which case, it
    back-tracks, then tries to grow another branch until the *condition* is
    fulfilled again, and so on."""

    if type_ is None:
        type_ = pset.ret
    expr = []
    height = random.randint(min_, max_)
    stack = [(0, type_)]
    
    while len(stack) != 0:
        depth, type_ = stack.pop()
        
        # Prfung, ob es primitives und terminals mit dem type überhaupt gibt leer ist und erzwingen die andere Wahl
        if not pset.terminals[type_]:
            should_be_terminal = False  # Muss primitive sein
        elif not pset.primitives[type_]:
            should_be_terminal = True   # Muss terminal sein
        else:
            # Wenn beides da ist, entscheiden der Zufall und die Tiefe also orginale Logik
            should_be_terminal = condition(height, depth)

        if should_be_terminal:
            try:
                term = random.choice(pset.terminals[type_])
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError(f"The gp.generate function tried to add a terminal of type {type_}, but there is none available.").with_traceback(traceback)
            if type(term) is MetaEphemeral:
                term = term()
            expr.append(term)
        else:
            try:
                prim = random.choice(pset.primitives[type_])
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError(f"The gp.generate function tried to add a primitive of type {type_}, but there is none available.").with_traceback(traceback)
            expr.append(prim)
            for arg in reversed(prim.args):
                stack.append((depth + 1, arg))
    return expr

# Annahme: ARI hat ein bekanntes, erreichbares Optimum von 1.0 (perfekte Übereinstimmung mit dem Target);
# die Online-CVIs (SC/DBI/CHI) haben kein bekanntes Optimum, daher gilt dieser early-stop nur für die offline Phase (is_offline=True)
PERFECT_ARI_SCORE = 1.0 - 1e-9

# Annahme: Es wir genGrow() verwendet, also alle Blätter haben random Tiefe
def safe_genGrow(pset, min_, max_, type_=None):
    # Dasselbe wie grow() nur dass safeGenerate() aufgerufen wird.
    """ Generate an expression where each leaf might have a different depth
    between *min* and *max*."""


    def condition(height, depth):
            """Expression generation stops when the depth is equal to height
            or when it is randomly determined that a node should be a terminal.
            """
            return depth == height or \
                (depth >= min_ and random.random() < pset.terminalRatio)

    return safe_generate(pset, min_, max_, condition, type_)

def getEvalmethodName(method) ->str:
    return method.__name__

def getErrorVaule(measure) -> float:
    # Annahme: dass das so gehandelt wird
    clustering_worst_values = {
        "adjusted_rand_score": -0.5, # wertebereich [-0.5, 1] maximiert
        "silhouette_score": -1.0,    # wertebreich [-1, 1] maximiert
        "davies_bouldin_score": 1e9, # unendlich angenähert  #wertebereich [0, inf) minimiert
        "calinski_harabasz_score": 0.0 #wertebereich [0, inf) maximiert
    }
    
    return clustering_worst_values[getEvalmethodName(measure)]


def evaluatePipeline(individual, pset, data, measure, is_offline, target = None):
    pipeline_func = gp.compile(individual, pset)
    
    # Zähle nur die Primitives (Algorithmen)
    operators = [node for node in individual if isinstance(node, gp.Primitive)]
    length = len(operators) 

    try: 
        predicted_labels = pipeline_func(data)
        
        if is_offline: 
            performance = adjusted_rand_score(target, predicted_labels)
            
        else: # Online Evaluation
            #Annahme: kein CVI kommt mit nur einem cluster klar, Annhame: NoiseKlasse zB ind DBSCAN -1 wird mit dazugezählt, dbi kommt nicht mit alle dps haben ein eigenes cluster klar
            if len(set(predicted_labels)) <= 1 or len(set(predicted_labels)) >= len(data):
                score = getErrorVaule(measure)
                performance = -score if measure == davies_bouldin_score else score
                return performance, length 
            
            score = measure(data, predicted_labels)
            performance = -score if measure == davies_bouldin_score else score # DBI miminiert eig also wird negiert damits passt

    except Exception as e:
        #print(e)
        #traceback.print_exc() 

        # Annahme: Wenn pipeline nicht valid ist so schlechte werte zurückgeben dass aussortiert wird
        score = getErrorVaule(measure)
        performance = -score if measure == davies_bouldin_score else score
        return performance, length 
    
    return performance, length  

# Annahme: die Methoden werden aus deap so übernommen
def mutate(individual, pset, expr):
    mode = random.random()
    if mode < 1/3:
        # Point Mutation 
        return gp.mutNodeReplacement(individual, pset=pset) # tauscht ein random primitiv mit einem anderen aus, der dieselbe # an argumenten hat
    elif mode < 2/3:
        #  Insert Mutation
        return gp.mutInsert(individual, pset=pset)
    else:
        # Shrink Mutation
        return gp.mutShrink(individual)
    


def createToolbox(pset, maxHeight):

    toolbox = base.Toolbox()

    toolbox.pset = pset 

    # min=1 erzwingt, dass die Wurzel immer eine Clustering alg  ist
    toolbox.register("expr", safe_genGrow, pset=pset, min_=1, max_=maxHeight) # Max Wert ist einstellbar
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    

    toolbox.register("evaluate", evaluatePipeline, pset=pset)

    toolbox.register("select", tools.selNSGA2)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", mutate, pset=pset, expr=toolbox.expr)

    #steht nicht im paper, würde die maxHeight begrenzung auch für offsprings durchsetzen
    #toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxHeight))
    #toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxHeight))
    return toolbox




def generateOffspring_PaperDefinition(parents, toolbox):
    offspring = [toolbox.clone(p) for p in parents for _ in range(5)]
        
    # Annahme: individuals werden random gewählt
    mate_indices = random.sample(range(len(offspring)), 2) # Annhame: 5% von 50 sind 2
    idx1, idx2 = mate_indices[0], mate_indices[1]
    offspring[idx1], offspring[idx2] = toolbox.mate(offspring[idx1], offspring[idx2])
    del offspring[idx1].fitness.values, offspring[idx2].fitness.values

    unaffected_indices = [i for i in range(len(offspring)) if offspring[i].fitness.valid]
    num_to_mutate = int(len(unaffected_indices) * 0.90) # Annahme: 90% von 48 = 43
    
    for i in random.sample(unaffected_indices, num_to_mutate):
        offspring[i], = toolbox.mutate(offspring[i])
        del offspring[i].fitness.values

    return offspring

def gernerateOffspring_ProbabilityInterpretation(parents, toolbox, offspring_multiplier=5, cxpb=0.05, mutpb=0.9):
    offspring = [toolbox.clone(p) for p in parents for _ in range(offspring_multiplier)]
    affected_indices = set()

    # Annahme: Wird mit 5% WS gecrossoverd mit random partner aus pop und nicht 5% werden fest gecrossoverd
    for i in range(len(offspring)):
        if i not in affected_indices and random.random() < cxpb:
            potential_partners = [j for j in range(len(offspring)) 
                                  if j != i and j not in affected_indices]
            
            if potential_partners:
                partner_idx = random.choice(potential_partners)
                
                offspring[i], offspring[partner_idx] = toolbox.mate(offspring[i], offspring[partner_idx])
                
                affected_indices.add(i)
                affected_indices.add(partner_idx)
                
                del offspring[i].fitness.values
                del offspring[partner_idx].fitness.values

    #Annahme: uneffected haben eine 90% muatation Chance also es werden nicht fest 90% gemutated
    for i in range(len(offspring)):
        if i not in affected_indices:
            if random.random() < mutpb:
                offspring[i], = toolbox.mutate(offspring[i])
                
                del offspring[i].fitness.values

    return offspring

def extract_best_individuals(individuals, n_best):
    # Extrahiert die n_best Individuen, sortiert nach Performance (entweder CVI oder ARI)
    # In offline phase wird nur das beste individuum zurückgegeben und in online die besten 5
    candidates = list(individuals)
    candidates.sort(key=lambda x: x.fitness.values[0], reverse=True)
    top_n = candidates[:n_best]
    hof = tools.HallOfFame(len(top_n))
    hof.update(top_n)
    
    return hof


def update_ranked_hall_of_fame(hof, individuals, n_best):
    candidates = list(hof) + list(individuals)
    return extract_best_individuals(candidates, n_best)


def runGeneticAlgorithm(toolbox, data, measure, is_offline, time_limit_seconds, target=None, warmstartPop : list=None,
                         population_size=50, n_select=10, offspring_multiplier=5, cxpb=0.05, mutpb=0.9,
                         hall_of_fame_size=None):
    # random.seed(42)
    start_time = time.time()
    deadline = start_time + time_limit_seconds
    n_to_extract = hall_of_fame_size if hall_of_fame_size is not None else (1 if is_offline else 5)

    if is_offline:
        pop = toolbox.population(n=population_size)
    else:
        pop = []

        for ind in warmstartPop:
            # nochmal clonen damit bei den drei CVI durchläufen nicht überschrieben wird
            klon = toolbox.clone(ind)
            del klon.fitness.values
            pop.append(klon)

        # Annhame, falls aus irgendeinem grund weniger da sind, den rest mit population_size random individuals auffüllen
        if len(pop) < population_size:
            extra = population_size - len(pop)
            pop.extend(toolbox.population(n=extra))

        # Falls sie aus irgendeinem grund größer ist nur die population_size nehmen
        pop = pop[:population_size]

        print(f"Warmstart aktiv: {len(warmstartPop)} Seeds geladen.")
    
    global_pareto = tools.ParetoFront()
    
    print(f"Initialisiere Population für {getEvalmethodName(measure)}...")

    fitnesses = [toolbox.evaluate(ind, data=data, measure=measure, is_offline=is_offline, target=target)  for ind in pop]
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    global_pareto.update(pop)
    ranked_hof = extract_best_individuals(pop, n_to_extract)

    # Annahme: nur in der offline Phase (ARI als Measure) gibt es ein erreichbares Optimum (1.0), bei dem
    # eine Weitersuche keinen Mehrwert mehr bringt; die online-Struktur (CVI-Läufe, Warmstart, 3x-Aufruf) bleibt unverändert
    perfect_score_found = is_offline and ranked_hof[0].fitness.values[0] >= PERFECT_ARI_SCORE

    gen = 0
    while not perfect_score_found and time.time() < deadline:
        gen += 1

        parents = toolbox.select(pop, n_select)

        offspring = gernerateOffspring_ProbabilityInterpretation(parents, toolbox, offspring_multiplier=offspring_multiplier, cxpb=cxpb, mutpb=mutpb)

        for ind in offspring:
            # Annahme: vor jeder pipeline bewertung prüfen ob zeitlimit erreicht, da die dauer der bewertung von datensatzgr. abhängt
            if time.time() > deadline:
                break

            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind, data=data, measure=measure, is_offline=is_offline, target=target)

                if is_offline and ind.fitness.values[0] >= PERFECT_ARI_SCORE:
                    perfect_score_found = True
                    break


        valid_offspring = [ind for ind in offspring if ind.fitness.valid]

        if len(valid_offspring) == len(offspring):
            pop[:] = offspring
            global_pareto.update(offspring)
            ranked_hof = update_ranked_hall_of_fame(ranked_hof, offspring, n_to_extract)
        else:
            global_pareto.update(valid_offspring)
            ranked_hof = update_ranked_hall_of_fame(ranked_hof, valid_offspring, n_to_extract)
            break

        if perfect_score_found:
            elapsed = time.time() - start_time
            print(f"Perfekter ARI-Score (1.0) erreicht | Gen {gen:03d} | Zeit: {elapsed:.1f}s — stoppe vorzeitig.")

        if gen % 5 == 0 or gen == 1:
            elapsed = time.time() - start_time
            best_score = max(ind.fitness.values[0] for ind in global_pareto)
            print(f"Gen {gen:03d} | Zeit: {elapsed:.1f}s | Best {getEvalmethodName(measure)}: {best_score:.4f} | Pareto: {len(global_pareto)}")

    print(f"\n Test beendet nach {gen} Generationen ")
    
    final_hof = ranked_hof

    best_ind = final_hof[0]
    print(f"Beste Pipeline {getEvalmethodName(measure)}: {best_ind.fitness.values[0]:.4f}, Struktur: {best_ind}")
    
    return final_hof
    

def create_clustering_ensamble(toolbox, data : Data, hofs:list):
    """
    Erstellt ein einziges Clustering aus den insgesamt 15 verschiedenen Pipelines (jw. die 5 besten pro CVI-Durchlauf auf d_new).

    :param hofs: Ein set der form {hof_SC, hof_CHI, hof_DBI}
    :returns: Ein finales Clustering von d_new
    """

    # matrix erstellen

    N = len(data)
    co_matrix = np.zeros((N, N), dtype=np.uint8) 
    successful_pipelines = 0

    for hof in hofs:
        for pipeline in hof:

            pipeline_func = gp.compile(pipeline, toolbox.pset)

            try: 
                labels = pipeline_func(data)
                print(labels)
                # matrix mit labels updaten
                agreement_matrix = (labels[:, None] == labels[None, :]).astype(int)
                co_matrix += agreement_matrix.astype(np.uint8)
                successful_pipelines +=1

            except Exception as e:
                print(f"Pipeline im Ensemble übersprungen wegen Fehler: {e}")
                # Annhame: Wenn die Pipeline fehl schlägt, wird sie übersprungen
                continue 

    
    # Matrix normalisieren

    co_matrix = co_matrix / successful_pipelines
    # mit threshold das cluster erstellen
    adjacency_matrix = (co_matrix > 0.5).astype(int)
    
    n_components, final_labels = connected_components(
        csgraph=adjacency_matrix, 
        directed=False, 
        return_labels=True
    )

    return final_labels
        


def reset_deap_creators():
    if "FitnessMulti" in creator.__dict__: del creator.FitnessMulti
    if "Individual" in creator.__dict__: del creator.Individual

    creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0)) #maximierung d. evaluationsmetrik (im paper steht minimizing aber ARI und CVIs maximieren alle ausser DBI (wird deswegen später negiert)), und minimierung der pipeline-Complexity (lenghth)
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)


def run_offline_dataset(dataset_path: str, args):
    print(f"\n=== Dataset: {dataset_path} ===")
    df = pd.read_csv(dataset_path)
    if args.target_column not in df.columns:
        raise ValueError(f"Target column '{args.target_column}' not found in {dataset_path}")

    X = df.drop(columns=[args.target_column]).values
    y_true = df[args.target_column].values

    reset_deap_creators()

    my_pset = createPset(n_features=X.shape[1])
    my_toolbox = createToolbox(my_pset, maxHeight=args.max_tree_height)

    return runGeneticAlgorithm(
        my_toolbox, X, adjusted_rand_score, True, args.time_limit, y_true,
        population_size=args.population_size,
        n_select=args.n_select,
        offspring_multiplier=args.offspring_multiplier,
        cxpb=args.cxpb,
        mutpb=args.mutpb,
        hall_of_fame_size=args.hall_of_fame_size,
    )


def serialize_hall_of_fame(dataset_path: str, hof):
    ranked = sorted(hof, key=lambda ind: ind.fitness.values[0], reverse=True)
    hall_of_fame = [
        {
            "rank": rank,
            "ari": float(ind.fitness.values[0]),
            "pipeline": str(ind),
        }
        for rank, ind in enumerate(ranked, start=1)
    ]
    best = hall_of_fame[0] if hall_of_fame else None
    return {
        "dataset_name": os.path.basename(dataset_path),
        "dataset_path": dataset_path,
        "hall_of_fame": hall_of_fame,
        "best_pipeline": best["pipeline"] if best else None,
        "best_ari": best["ari"] if best else None,
    }


def resolve_dataset_paths(args, default_dataset_path: str):
    if args.input_folder:
        input_folder = os.path.abspath(os.path.expanduser(args.input_folder))
        if not os.path.isdir(input_folder):
            raise FileNotFoundError(f"Input folder not found: {input_folder}")
        dataset_paths = []
        for root, _, files in os.walk(input_folder):
            for file_name in files:
                if file_name.lower().endswith(".csv"):
                    dataset_paths.append(os.path.join(root, file_name))
        dataset_paths.sort()
        if not dataset_paths:
            raise FileNotFoundError(f"No CSV files found under input folder: {input_folder}")
        return dataset_paths

    dataset_path = args.dataset_path or default_dataset_path
    return [os.path.abspath(os.path.expanduser(dataset_path))]


def safe_filename_stem(name: str):
    stem = os.path.splitext(os.path.basename(str(name)))[0]
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return stem or "dataset"


def prepare_output_folder(output_folder: str):
    output_dir = os.path.abspath(os.path.expanduser(output_folder))
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def output_folder_from_deprecated_output_file(output_file: str):
    output_file = os.path.abspath(os.path.expanduser(output_file))
    output_dir = os.path.dirname(output_file)
    output_stem = safe_filename_stem(os.path.basename(output_file))
    return os.path.join(output_dir, output_stem)


def dataset_output_path(dataset_path: str, output_dir: str, used_file_names: set):
    dataset_name = os.path.basename(dataset_path)
    base_file_stem = safe_filename_stem(dataset_name)
    file_stem = base_file_stem
    suffix = 2
    while f"{file_stem}.json" in used_file_names:
        file_stem = f"{base_file_stem}_{suffix}"
        suffix += 1

    file_name = f"{file_stem}.json"
    used_file_names.add(file_name)
    return os.path.join(output_dir, file_name)


def result_output_path(result, output_dir: str, used_file_names: set):
    dataset_path = result.get("dataset_path") or result.get("dataset_name") or "dataset"
    return dataset_output_path(dataset_path, output_dir, used_file_names)


def write_result_to_path(result, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    return output_path


def write_result(result, output_dir: str, used_file_names: set):
    output_path = result_output_path(result, output_dir, used_file_names)
    return write_result_to_path(result, output_path)


def write_results(results, output_folder: str):
    output_dir = prepare_output_folder(output_folder)
    used_file_names = set()
    output_paths = [
        write_result(result, output_dir, used_file_names)
        for result in results
    ]
    return output_paths


def apply_memory_limit(max_memory_gb):
    # Annahme: O(n^2)-Algorithmen (Agglomerative/Spectral/KNNImputer) auf großen Datensätzen
    # können den Prozess sonst über den OOM-Killer abschießen, ohne dass Python das try/except
    # in evaluatePipeline überhaupt erreicht. RLIMIT_AS sorgt dafür, dass stattdessen ein
    # regulärer MemoryError geworfen wird.
    if max_memory_gb is None:
        return
    try:
        import resource
    except ImportError:
        print(
            f"Warning: --max-memory-gb is not supported on this platform (no 'resource' module); "
            "continuing without a memory limit.",
            file=sys.stderr,
        )
        return

    limit_bytes = int(max_memory_gb * (1024 ** 3))
    try:
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
        print(f"Applied virtual memory limit: {max_memory_gb} GB (RLIMIT_AS)")
    except (ValueError, OSError) as e:
        print(
            f"Warning: failed to apply memory limit ({e}); continuing without a memory limit.",
            file=sys.stderr,
        )


def _run_dataset_in_subprocess(dataset_path: str, args, output_path: str):
    # Annahme: macOS startet Kindprozesse via "spawn", ein frisches Interpreter-Image
    # erbt also keine globalen/Resourcen-Limits vom Elternprozess, daher hier erneut setzen.
    apply_memory_limit(args.max_memory_gb)
    try:
        hof = run_offline_dataset(dataset_path, args)
        result = serialize_hall_of_fame(dataset_path, hof)
        write_result_to_path(result, output_path)
    except MemoryError:
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run the AutoClust genetic algorithm pipeline search on one labeled CSV dataset or a folder of CSV datasets."
    )
    parser.add_argument("--time-limit", type=int, default=20,
        help="Time budget for the genetic algorithm run, in seconds (default: 20)")
    parser.add_argument("--max-tree-height", type=int, default=6,
        help="Maximum height of generated GP pipeline trees (default: 6)")
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--dataset-path", type=str, default=None,
        help="Path to one CSV dataset to load (default: '<script_dir>/../data/wine.csv')")
    input_group.add_argument("--input-folder", type=str, default=None,
        help="Folder containing CSV datasets to run recursively")
    parser.add_argument("--target-column", type=str, default="target",
        help="Name of the target/label column in the dataset CSV (default: 'target')")
    parser.add_argument("--population-size", type=int, default=50,
        help="Number of individuals in the GA population; also the warmstart fill/cap size (default: 50)")
    parser.add_argument("--n-select", type=int, default=10,
        help="Number of parents selected each generation via NSGA-II (default: 10)")
    parser.add_argument("--offspring-multiplier", type=int, default=5,
        help="Number of cloned offspring generated per selected parent each generation (default: 5)")
    parser.add_argument("--cxpb", type=float, default=0.05,
        help="Per-individual crossover probability during offspring generation (default: 0.05)")
    parser.add_argument("--mutpb", type=float, default=0.9,
        help="Per-individual mutation probability for unaffected offspring (default: 0.9)")
    parser.add_argument("--hall-of-fame-size", type=int, default=5,
        help="Number of ranked Hall of Fame pipelines to store per dataset (default: 5)")
    parser.add_argument("--output-folder", type=str, default="autoclust_results",
        help="Folder for per-dataset JSON result files (default: autoclust_results)")
    parser.add_argument("--max-memory-gb", type=float, default=None,
        help="Hard cap on process virtual memory in GB (Linux only, via RLIMIT_AS). "
             "Pipelines that would exceed it raise MemoryError instead of triggering the "
             "OS OOM killer. Default: no cap.")
    parser.add_argument("--hard-time-limit", type=int, default=None,
        help="Hard wall-clock cap, in seconds, on a single dataset's entire run "
             "(search + serialize + write), enforced via a subprocess that is "
             "forcibly terminated if it overruns. Unlike --time-limit, this can "
             "interrupt a single pipeline evaluation stuck in a long-running "
             "native call (e.g. AgglomerativeClustering/SpectralClustering/"
             "KNNImputer on a large dataset). Default: disabled (run in-process).")
    parser.add_argument("--output-file", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.hard_time_limit is not None and args.hard_time_limit <= args.time_limit:
        print(
            f"Warning: --hard-time-limit ({args.hard_time_limit}s) is <= --time-limit "
            f"({args.time_limit}s); the hard limit may fire before the soft limit gets "
            "a chance to finish even one generation cleanly.",
            file=sys.stderr,
        )

    apply_memory_limit(args.max_memory_gb)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_dataset_path = os.path.join(current_dir, "..", "data", "wine.csv")
    dataset_paths = resolve_dataset_paths(args, default_dataset_path)
    print(f"Running offline ARI optimization for {len(dataset_paths)} dataset(s).")

    output_folder = args.output_folder
    if args.output_file:
        output_folder = output_folder_from_deprecated_output_file(args.output_file)
        print(
            f"Warning: --output-file is deprecated; writing per-dataset files under {output_folder}. "
            "Use --output-folder instead.",
            file=sys.stderr,
        )

    output_dir = prepare_output_folder(output_folder)
    used_file_names = set()
    output_paths = []
    skipped_output_paths = []
    oom_skipped_datasets = []
    hard_timeout_skipped_datasets = []
    failed_datasets = []

    # Annahme: spawn-Context nur erzeugen, wenn --hard-time-limit aktiv ist, damit der
    # Default-Pfad (ohne dieses Flag) keinerlei Multiprocessing-Overhead bekommt.
    mp_ctx = multiprocessing.get_context("spawn") if args.hard_time_limit is not None else None

    for dataset_path in dataset_paths:
        output_path = dataset_output_path(dataset_path, output_dir, used_file_names)
        if os.path.exists(output_path):
            skipped_output_paths.append(output_path)
            print(f"Skipping dataset because result already exists: {output_path}")
            continue

        if args.hard_time_limit is None:
            try:
                hof = run_offline_dataset(dataset_path, args)
                result = serialize_hall_of_fame(dataset_path, hof)
                write_result_to_path(result, output_path)
                output_paths.append(output_path)
                print(f"Result written: {output_path}")
            except MemoryError:
                oom_skipped_datasets.append(dataset_path)
                print(f"Skipping dataset, exceeded --max-memory-gb limit: {dataset_path}", file=sys.stderr)
            except Exception as e:
                failed_datasets.append(dataset_path)
                print(f"Skipping dataset due to unexpected error: {dataset_path} ({e})", file=sys.stderr)
                traceback.print_exc()
            continue

        # Annahme: Nur eine eigene Kindprozess-Terminierung kann eine Pipeline stoppen,
        # die in einem blockierenden nativen Aufruf (zB AgglomerativeClustering) hängt,
        # da ein kooperativer Zeit-Check im Python-Code diesen Aufruf nicht unterbrechen kann.
        proc = mp_ctx.Process(
            target=_run_dataset_in_subprocess,
            args=(dataset_path, args, output_path),
        )
        proc.start()
        proc.join(args.hard_time_limit)

        if proc.is_alive():
            hard_timeout_skipped_datasets.append(dataset_path)
            print(
                f"Skipping dataset, exceeded --hard-time-limit of {args.hard_time_limit}s: {dataset_path}",
                file=sys.stderr,
            )
            proc.terminate()
            proc.join(5)
            if proc.is_alive():
                proc.kill()
                proc.join()
            continue

        if proc.exitcode == 0:
            output_paths.append(output_path)
            print(f"Result written: {output_path}")
        elif proc.exitcode == 2:
            oom_skipped_datasets.append(dataset_path)
            print(f"Skipping dataset, exceeded --max-memory-gb limit: {dataset_path}", file=sys.stderr)
        else:
            failed_datasets.append(dataset_path)
            print(
                f"Skipping dataset, subprocess failed with exit code {proc.exitcode}: {dataset_path}",
                file=sys.stderr,
            )

    print(f"\nResults written to {len(output_paths)} file(s):")
    for output_path in output_paths:
        print(f"- {output_path}")
    if skipped_output_paths:
        print(f"\nDatasets skipped because result files already existed: {len(skipped_output_paths)}")
        for output_path in skipped_output_paths:
            print(f"- {output_path}")
    if oom_skipped_datasets:
        print(f"\nDatasets skipped due to memory limit: {len(oom_skipped_datasets)}")
        for dataset_path in oom_skipped_datasets:
            print(f"- {dataset_path}")
    if hard_timeout_skipped_datasets:
        print(f"\nDatasets skipped due to hard time limit: {len(hard_timeout_skipped_datasets)}")
        for dataset_path in hard_timeout_skipped_datasets:
            print(f"- {dataset_path}")
    if failed_datasets:
        print(f"\nDatasets skipped due to unexpected errors: {len(failed_datasets)}")
        for dataset_path in failed_datasets:
            print(f"- {dataset_path}")

    #hof = runGeneticAlgorithm(my_toolbox, X, silhouette_score, False, TimeLimit)

    #hof = runGeneticAlgorithm(my_toolbox, X, davies_bouldin_score, False, TimeLimit)

    #hof = runGeneticAlgorithm(my_toolbox, X, calinski_harabasz_score, False, TimeLimit)
