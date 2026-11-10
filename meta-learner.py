import argparse
import os
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction import DictVectorizer
from sklearn.neighbors import NearestNeighbors
from data import Data

from helper import  filter_list_of_dicts, get_ari_clustering, get_filename, get_filename_from_ml_run, get_kb_name_from_filename, preprocess_df, preprocess_object, replace_nan_with_constant, str2bool
from eval_helper import evaluate

os.environ["OPENBLAS_NUM_THREADS"] = "10"


def correct_hps(d: Data, pipeline: list):
    np.random.seed(876)
    if 'iso_forest' in pipeline[0]:
        if pipeline[1]['iso_forest']['max_features'] > d.n_features:
            pipeline[1]['iso_forest']['max_features'] = np.random.randint(1, d.n_features)
    if 'pca' in pipeline[0]:
        if not pipeline[1]['pca']['n_components'] == 'mle':

            if isinstance(pipeline[1]['pca']['n_components'], str):
                pipeline[1]['pca']['n_components'] = int( pipeline[1]['pca']['n_components'])
            if pipeline[1]['pca']['n_components'] >= d.n_features:
                if pipeline[1]['pca']['svd_solver'] == 'arpack':
                    pipeline[1]['pca']['n_components'] = np.random.randint(1, min(d.n_features-1, d.n_samples-1))
                else:
                    pipeline[1]['pca']['n_components'] = np.random.randint(1, d.n_features)
    if 'lle' in pipeline[0]:
        if pipeline[1]['lle']['n_components'] > d.n_features:
            pipeline[1]['lle']['n_components'] = np.random.randint(1, d.n_features)
    return pipeline


def apply_ml(d: Data, kb: DataFrame, kb_name: str, k: int, pipelines: int, mf_groups: list[str], clustering_algorithm: str) -> dict:

    result_dict = {}
    result_dict['kb_name'] = kb_name
    result_dict['dataset_name'] = d.shape
    result_dict['clustering'] = clustering_algorithm
    d.X_modified = np.array(d.X_modified, dtype=np.float64)


    ari= get_ari_clustering(d.X_modified, d.y_modified, d.n_clusters, clustering_algorithm)
    print(f'Before {ari:0.2f}')
    result_dict['ari_before'] = ari

    kb_filtered = kb['mf_modified'].tolist()
    all_keys = [set(d.keys()) for d in kb_filtered]
    same_keys = all(k==all_keys[0] for k in all_keys)
    assert same_keys == True
    if not 'all' in mf_groups:
        kb_filtered = filter_list_of_dicts(kb_filtered, mf_groups)
    vec = DictVectorizer(sparse=False)
    meta_features_matrix = vec.fit_transform(kb_filtered)
    target_vector = vec.transform([d.mf_modified])
    
    target_vector[np.isposinf(target_vector)] = np.finfo(np.float64).max
    target_vector[np.isneginf(target_vector)] = np.finfo(np.float64).min

    knn_time_start = time.time()
    knn = NearestNeighbors(n_neighbors=1, metric='euclidean')

    knn.fit(meta_features_matrix)
    distances, indices = knn.kneighbors(target_vector, k)
    closest_index = indices[0][0]
    closest_record = kb.iloc[closest_index]
    closest_records = kb.iloc[indices[0]].reset_index(drop=True)
    knn_time_end = time.time()
    result_dict['knn_time'] = knn_time_end-knn_time_start
    best_after = -1
    total_bpl = None
    pls = []
    i = 0
    pp_time_start = time.time()
    for idx, cr in closest_records.iterrows():
        best_pipelines = cr['top_5_pipeline'][:pipelines]
        
        #ranking of pipeline within k
        j = 0
        for best_pl in best_pipelines:
            j+=1
            i = i+1
            print(f'pipeline {i}')
            best_pl = correct_hps(d, best_pl)
            eval_time_start = time.time()
            X,y, invalid = evaluate(d.X_modified.copy(), best_pl, d, d.y_modified)
            if invalid:
                print('INVALID')
                #continue
                ari_after = -1
            else:
                ari_after= get_ari_clustering(X, y, d.n_clusters, clustering_algorithm)
            eval_time_end = time.time()
            #idx is k, j is p, 
            pls.append((best_pl, ari_after, eval_time_end-eval_time_start,idx+1, distances[0][idx], j, len(X), np.unique(y).tolist()))
            if best_after <= ari_after:
                best_after = ari_after
                total_bpl = best_pl
    pp_time_end = time.time()
    result_dict['pp_time'] = pp_time_end - pp_time_start
    print(f'After {best_after:0.2f}')
    print(f'Diff {best_after-ari:0.2f}')
    result_dict['best_ari'] = best_after
    result_dict['diff'] = best_after-ari
    result_dict['best_pl'] = total_bpl
    result_dict['k'] = k
    result_dict['nr_pipelines'] = pipelines
    result_dict['pls'] = pls
    result_dict['uuid'] = d.uuid
    result_dict['mf_groups'] = mf_groups
    return result_dict




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-kb','--knowledge_base', type=str, required=True, help='Path to knowledge base')

    parser.add_argument('-sd','--single_dataset', type=str2bool, nargs='?', const=True, default=False,help='Use only single file')
    parser.add_argument('-d','--datasets', type=str, required=False, help='Path to dataset folder (all files will be considered) works only of ds is not set')
    parser.add_argument('-dsp','--single_dataset_path', type=str, required=False, help='Path to dataset file works only of ds is set')
    parser.add_argument('-o','--output_folder', type=str, required=True, help='Output folder path')
    parser.add_argument('-k', nargs='+', type=int, required=True, help='k neighbors')
    parser.add_argument('-p', '--pipelines', nargs='+', type=int, required=True, help='number of pipelines per neighbor')
    parser.add_argument('-oe', '--overwrite_existing', type=int, required=False, default=0, help='Overwrite existing results (param value =1), default is 0')
    parser.add_argument('-c', '--clustering', type=str, required=False, default='kmeans', help='Clustering algorithm used for evaluation, options are kmeans, dbscan, hdbscan and agglomerative')
    parser.add_argument('-mfg','--meta_feature_groups',
                        default=['all'],
                        nargs='+',
                        choices=('all', 'general', 'statistical', 'complexity'),
                        help='defines the used mf groups: all, general, statistical, complexity. Default is all, multiple options are possible')


    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    kb = pd.read_csv(args.knowledge_base)
    kb = preprocess_df(kb)

    file_paths = []
    if args.single_dataset:
        file_paths.append(args.single_dataset_path)
    else:
        file_paths = [os.path.join(args.datasets, f) for f in os.listdir(args.datasets) if f.endswith('.pickle') and os.path.isfile(os.path.join(args.datasets, f))]

    assert len(file_paths) > 0, 'no dataset files found'
    for file_path in file_paths:
        name = Path(file_path).stem
        print(f'reading {name}')
        if name == 'time_dict' or name == 'rna': continue
        with open(file_path, 'rb') as f:
            data_object = pickle.load(f)
            data_object = preprocess_object(data_object)
            kb_name = get_kb_name_from_filename(args.knowledge_base)
            for k in args.k:
                for p in args.pipelines:
                    if args.overwrite_existing == 0:
                        tmp_file_name = get_filename(dataset_name=data_object.shape,
                                                    knowledge_base_name=kb_name,
                                                    k=k,
                                                    p=p,
                                                    mf_groups=args.meta_feature_groups,
                                                    clustering_algo=args.clustering)

                        tmp_path = os.path.join(args.output_folder, tmp_file_name)

                        if os.path.isfile(tmp_path):
                            print(f'skipping {tmp_file_name}')
                            continue

                    result_dict = apply_ml(data_object, kb=kb, kb_name=kb_name, k=k, pipelines=p, mf_groups= args.meta_feature_groups, clustering_algorithm = args.clustering)
                    result_file_name = get_filename_from_ml_run(result_dict)
                    result_path = os.path.join(args.output_folder, result_file_name)
                    result_df = pd.DataFrame([result_dict])
                    result_df.to_csv(result_path)