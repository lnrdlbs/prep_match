import ast
from enum import Enum
import os
import random
import numpy as np
import pandas as pd
from pandas import DataFrame
import scipy
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge, SGDOneClassSVM
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, PowerTransformer, QuantileTransformer, RobustScaler
from sklearn.random_projection import GaussianRandomProjection
from sklearn.svm import OneClassSVM

from data import Data


pp_methods = {
    'outlier': {
        'lof': {
            'p_neighbors': [0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'p_leaf_size': [1,5,10,20,30,50],
            'p': [1,2,5,10],
            'contamination': ['auto', 0.05, 0.1, 0.2, 0.3, 0.4]
                },
        'iso_forest': {
            'n_estimators': [10,50,100,200,500],
            'max_features': [2,5,10,25,35],
            'bootstrap': [True, False],
            'warm_start': [True, False]

        },
        'one_class_svm': {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'gamma': ['scale', 'auto'], 'nu': [0.3,0.5,0.7], 'shrinking': [True, False], 'max_iter': [100,500,1000]},
        'elliptic': {'assume_centered': [True, False], 'support_fraction': [True, False], 'contamination': [0.1, 0.2, 0.3, 0.4, 0.5]},
        'sdg_one_class_svm': {'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'], 'warm_start': [True, False], 'nu': [0.3,0.5,0.7], 'max_iter': [100,500,1000], 'eta0': [0.1,0.2,0.3]},

        },
    'scaling': {
        'power_transformer': {'standardize': [True, False]},
        'robust': {
            'with_centering': [True, False], 
            'with_scaling': [True, False], 
            'unit_variance': [True, False]},
        'standard_scaler': {
            'with_mean': [True, False],
            'with_std': [True, False]
            },
        'min_max': {'placeholder': ['placeholder']},
        'quantile': {'p_quantiles': [0.3,0.5,0.7,1], 'output_distribution': ['uniform', 'normal']},
        'max_abs': {'placeholder': ['placeholder']}
        
    },
    'dim_red':
        {
            'pca': {'n_components': ['mle', 2,5,10,25,35], 'whiten': [True, False], 'svd_solver': ['auto', 'full', 'arpack', 'randomized'], 'power_iteration_normalizer': ['auto','QR', 'LU']},
            'grp': {'n_components': ['auto', 2,5,10,25,35], 'eps': [0.01, 0.1, 0.3]},
            'lle': {'p_neighbors':  [0.01, 0.05, 0.1], 'n_components': [2,5,10,25,35], 'eigensolver':['auto', 'arpack', 'dense'],  'max_iter':  [50,100,500,1000], 'method': ['standard', 'hessian', 'modified', 'ltsa'], 'neighbors_algorithm': ['auto', 'brute', 'kd_tree', 'ball_tree']},
        }
}

missing_value_methods = {
    'simple': {'strategy': ['mean', 'median', 'most_frequent']},
    'knn': {'n_neighbors': range(1,15)},
    #'iterative': {'estimator': ['bayesian', 'randomforest'], 'max_iter':range(5,20), 'initial_strategy': ['mean', 'median', 'most_frequent'], 'imputation_order': ['ascending', 'descending', 'roman', 'arabic', 'random'] },
}

def get_random_imputation():
    methods = list(missing_value_methods.keys())
    method = np.random.choice(methods)
    hps = {}
    for param, values in missing_value_methods[method].items():
        hps[param] = np.random.choice(values) if isinstance(values, list) else np.random.randint(values.start, values.stop)
    
    return method, hps


def select_methods(data_object: Data, length):
    
    if data_object == None:
        methods = []
        #if data_object.has_outlier:
            #methods.append(np.random.choice(list(pp_methods['outlier'].keys())))
            
        #if data_object.has_scaling:
        #methods.append(np.random.choice(list(pp_methods['scaling'].keys())))
        #if not data_object.has_outlier and not data_object.has_scaling: 
        tmp = [np.random.choice(list(pp_methods['outlier'].keys())), np.random.choice(list(pp_methods['scaling'].keys())), np.random.choice(list(pp_methods['dim_red'].keys())) ]
        for i in range(length):
            methods.append(np.random.choice(tmp))
        return methods
        
    else:
        methods = []
        if data_object.has_outlier:
            methods.append(np.random.choice(list(pp_methods['outlier'].keys())))
            
        if data_object.has_scaling:
            methods.append(np.random.choice(list(pp_methods['scaling'].keys())))
        if not data_object.has_outlier and not data_object.has_scaling: 
            tmp = [np.random.choice(list(pp_methods['outlier'].keys())), np.random.choice(list(pp_methods['scaling'].keys())), np.random.choice(list(pp_methods['dim_red'].keys())) ]
            methods.append(np.random.choice(tmp))
        return methods

def mutate_iso_max_features(data_object: Data): 
    
    available_values = pp_methods['outlier']['iso_forest']['max_features'].copy()
    max_f = np.random.choice(available_values)

    print(f'mut iso {max_f}, {data_object.n_features}')
    if data_object.n_features < max_f: 
        max_f = data_object.n_features
    return max_f


def mutate_pca_n_comb(data_object: Data):
    max_com = min(data_object.n_samples, data_object.n_features)
    options = ['comp', 'mle']
    choice = np.random.choice(options, p=[0.8,0.2])

    if choice == 'comp':
        if max_com <= 2: return 2
        return np.random.randint(2, max_com)
    elif choice == 'mle':
        return 'mle'

def mutate_svd_solver(hps: dict, method: str):
    all_solvers =  pp_methods['dim_red'][method]['svd_solver'].copy()
    if hps['n_components'] == 'mle':
        return 'auto'
    else:
        return random.choice(all_solvers)
    
def mutate_n_components(data_object: Data, method): 
    valid_values = pp_methods['dim_red'][method]['n_components']
    c = np.random.choice(valid_values)
    if c == 'auto': return c
    c = int(c)
    if c > data_object.n_features:
        return data_object.n_features
    else: return c
    
def init_hyperparameters(method, data_object: Data):
    
    hps = {}
    if  method in pp_methods['outlier'].keys():
         
        for param, values in pp_methods['outlier'][method].items():
            if method == 'iso_forest' and param == 'max_features':
                hps[param] = mutate_iso_max_features(data_object=data_object)
            hps[param] = np.random.choice(values) if isinstance(values, list) else np.random.randint(values.start, values.stop)
    
    elif method in pp_methods['scaling'].keys():
        if method == 'min_max' or method == 'max_abs':
                hps = {}
                return hps
        
        for param, values in pp_methods['scaling'][method].items():
            hps[param] = np.random.choice(values) if isinstance(values, list) else np.random.randint(values.start, values.stop)
    elif method in pp_methods['dim_red'].keys():
        for param, values in pp_methods['dim_red'][method].items():
            if method == 'pca':
                if param == 'n_components':
                    hps[param]= mutate_pca_n_comb(data_object=data_object)
                elif param == 'svd_solver':
                    hps[param]= mutate_svd_solver(hps=hps, method=method)
                else:
                    hps[param] = np.random.choice(values) if isinstance(values, list) else np.random.randint(values.start, values.stop)
            elif param == 'n_components':
                hps[param] = mutate_n_components(data_object=data_object, method=method)
            else:
                hps[param] = np.random.choice(values) if isinstance(values, list) else np.random.randint(values.start, values.stop)
        
    return hps

#def get_random_pipeline(data_object: Data):
def get_pipeline(rnd: bool, length: int, data_object):
    
    if rnd:
        methods = select_methods(data_object=None, length=length)
    else: 
        methods = select_methods(data_object=data_object, length=length)
    hps = {}
    for m in methods:
        hp = init_hyperparameters(method=m, data_object=data_object)
        hps[m] = hp
    
    np.random.shuffle(methods)
    
    order = [(i+1, methods[i]) for i in range(len(methods))]
    return methods, hps, order

def evaluate(X, individual, data_object, y): 

    
    #print(individual)
    individual[2]= sorted(individual[2], key=lambda x: x[0])
    invalid = False
    
    try:
        for el in individual[2]:
            method = el[1]

            
            hps = individual[1][method]
            if method == 'lof':
                n_neighbors = int(hps['p_neighbors'] * len(X))
                if n_neighbors <= 0: n_neighbors = 1
                leaf_size =  int(hps['p_leaf_size'] * len(X))
                if hps['contamination'] == 'auto': 
                    contamination = 'auto'
                else: contamination = float(hps['contamination'])
                try:
                    operator = LocalOutlierFactor(n_neighbors=n_neighbors, leaf_size=leaf_size, p=hps['p'], algorithm=hps['algorithm'], contamination=contamination)
                    is_inlier = operator.fit_predict(X)
                    X = X[is_inlier==1]
                    y = y[is_inlier==1]
                except ValueError as e: 
                    print(f'Error eval lof: {e}')
                    invalid = True
            elif method == 'iso_forest':
                if hps['max_features'] == 'auto': 
                    max_features = 'auto'
                else: max_features = int(hps['max_features'])
                
                try:
                    operator = IsolationForest(n_estimators=hps['n_estimators'], max_features=max_features, bootstrap=hps['bootstrap'], warm_start=hps['warm_start'], random_state=876)
                    is_inlier = operator.fit_predict(X)
                
                    X = X[is_inlier==1]
                    y = y[is_inlier==1]
                except ValueError as e:
                    print(f'Error eval iso: {e}')
                    invalid = True

            elif method == 'power_transformer': 
                try:
                    operator = PowerTransformer(standardize=hps['standardize'])
                    X = operator.fit_transform(X)
                except scipy.optimize._optimize.BracketError as e:
                    print(f'Error eval power: {e}')
                    invalid = True
                    
            elif method == 'one_class_svm':
                try:
                    operator = OneClassSVM(kernel=hps['kernel'], gamma=hps['gamma'], nu=hps['nu'], shrinking=hps['shrinking'], max_iter=hps['max_iter'])
                    is_inlier = operator.fit_predict(X)
                    X = X[is_inlier==1]
                    y = y[is_inlier==1]
                except ValueError as e:
                    print(f'Error eval one_class_svm: {e}')
                    invalid = True
            elif method == 'elliptic':
                try:
                    operator = EllipticEnvelope(assume_centered=hps['assume_centered'], contamination=hps['contamination'], random_state=876)
                    is_inlier = operator.fit_predict(X)
                    X = X[is_inlier==1]
                    y = y[is_inlier==1]
                except ValueError as e:
                    print(f'Error eval elliptic: {e}')
                    invalid = True
                except UnboundLocalError as e:
                    print(f'Internal Error eval elliptic: {e}')
                    invalid = True
                    
            elif method == 'sdg_one_class_svm':
                try:
                    operator = SGDOneClassSVM(learning_rate=hps['learning_rate'], warm_start=hps['warm_start'], nu=hps['nu'], max_iter=hps['max_iter'], eta0=hps['eta0'], random_state=876)
                    is_inlier = operator.fit_predict(X)
                    X = X[is_inlier==1]
                    y = y[is_inlier==1]
                except ValueError as e:
                    print(f'Error eval sdg  {e}')
                    invalid = True
            elif method == 'robust':
                try:
                    operator = RobustScaler(with_centering=hps['with_centering'], with_scaling=hps['with_scaling'], unit_variance=hps['unit_variance'])
                    X = operator.fit_transform(X)
                except ValueError as e:
                    print(f'Error eval robust  {e}')
                    invalid = True

            elif method == 'standard_scaler':
                try:
                    operator = StandardScaler(with_mean=hps['with_mean'], with_std=hps['with_std'])
                    X = operator.fit_transform(X)
                except ValueError as e:
                    print(f'Error eval standard  {e}')
                    invalid = True
            elif method == 'min_max':
                try:
                    operator = MinMaxScaler()
                    X = operator.fit_transform(X)
                except ValueError as e:
                    print(f'Error eval min_max  {e}')
                    invalid = True
            elif method == 'quantile':
                n_quantiles = int(hps['p_quantiles'] * len(X))
                try:
                    operator = QuantileTransformer(n_quantiles=n_quantiles, output_distribution=hps['output_distribution'], random_state=876)
                    X = operator.fit_transform(X)
                except ValueError as e:
                    print(f'Error eval quantile  {e}')
                    invalid = True
            elif method == 'max_abs':
                try:
                    operator = MaxAbsScaler()
                    X = operator.fit_transform(X)
                except ValueError as e:
                    print(f'Error eval max_abs  {e}')
                    invalid = True
            elif method == 'pca':
                
                if hps['n_components'] == 'mle':
                    n_components = 'mle'
                else:
                    n_components = int(hps['n_components'])
                try: 
                    operator = PCA(n_components=n_components, whiten=hps['whiten'], svd_solver=hps['svd_solver'],  power_iteration_normalizer=hps['power_iteration_normalizer'], random_state=876)
                    X = operator.fit_transform(X)
                except  ValueError as e:
                    print(f'Error eval pca {e}')
                    invalid = True
                except scipy.sparse.linalg._eigen.arpack.arpack.ArpackError:
                    print(f'Error eval pca argpack {e}')
                    invalid = True
                
            elif method == 'grp':
                try:
                    if hps['n_components'] == 'auto':
                        n_components = 'auto'
                    else:
                        n_components = int(hps['n_components'])
                    operator = GaussianRandomProjection(n_components=n_components, eps=hps['eps'], random_state=876)
            
                    X = operator.fit_transform(X)
                except  ValueError as e:
                    print(f'Error eval grp {e}')
                    invalid = True
                except TypeError as e:
                    print(f'Error eval grp type {e}')
                    invalid = True
                    
            elif method == 'lle':
                n_neighbors = int(hps['p_neighbors'] * len(X))
                try:
                    operator = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=hps['n_components'], eigen_solver=hps['eigensolver'], max_iter=hps['max_iter'], method=hps['method'], neighbors_algorithm=hps['neighbors_algorithm'], random_state=876)
                    X = operator.fit_transform(X)
                except ValueError as e:
                    print(f'Error eval lle {e}')
                    invalid = True
            else:
                raise ValueError(f'Eval: Unknown method: {method}')
        
        if  np.isnan(X).any():
            print(f'Nans introduced {individual}')
            invalid = True
        
        if len(X) < data_object.n_clusters:
            print(f'To few samples s: {len(X)}, c: {data_object.n_clusters}')
            invalid = True
    except Exception as e:
        invalid = True
        print(f'exception: {e}')
    return X, y, invalid
   

def evaluate_missing(X: np.ndarray, individual): 

    method = individual[0]
    params = individual[1]
    
    if method == 'knn':
        imputer = KNNImputer(n_neighbors=params['n_neighbors'])
    elif method == 'simple':
        imputer = SimpleImputer(strategy=params['strategy'])
    elif method == 'iterative':
        if params['estimator'] == 'bayesian':
            estimator = BayesianRidge()
        elif params['estimator'] == 'randomforest':
            estimator = RandomForestRegressor(n_jobs=20)
        else: 
            raise ValueError(f'Invalid estimator {params["estimator"]}')

        imputer = IterativeImputer(estimator=estimator, max_iter=params['max_iter'], initial_strategy=params['initial_strategy'], imputation_order=params['imputation_order'], random_state=data_object.seed)
    else: raise ValueError(f'Invalid imputation method: {method}')
    
    
    X = imputer.fit_transform(X)
    
    return X


class DfNames(Enum):
    MISSING = 1
    MISSING_SECOND = 2
    OUTLIER = 3
    SCALING = 4
    MISSING_OUTLIER = 5
    MISSING_OUTLIER_SECOND = 6
    MISSING_SCALING = 7
    MISSING_SCALING_SECOND = 8
    SCALING_OUTLIER = 9
    MISSING_SCALING_OUTLIER = 10
    MISSING_SCALING_OUTLIER_SECOND = 11
    BASE = 12
    ALL = 13
    
    
def get_data_dirs(enum_value: DfNames, path_prefix: str = ''):
    
    path = ''
    if enum_value == DfNames.BASE:
        path = 'data/base_v3/'
    elif enum_value == DfNames.MISSING:
        path = 'data/missing_v3/'
    elif enum_value == DfNames.MISSING_SECOND:
        path = 'data/imputed_v3/'
    elif enum_value == DfNames.OUTLIER:
        path = 'data/outlier_v3'
    elif enum_value == DfNames.SCALING:
        path = 'data/scaling_v3'
    elif enum_value == DfNames.MISSING_OUTLIER:
        path = 'data/missing_outlier_v3'
    elif enum_value == DfNames.MISSING_OUTLIER_SECOND:
        path = 'data/imputed_missing_outlier_v3/imputed_missing_outlier_v3'
    elif enum_value == DfNames.MISSING_SCALING:
        path = 'data/missing_scaling_v3'
    elif enum_value == DfNames.MISSING_SCALING_SECOND:
        path = 'data/imputed_missing_scaling_v3'
    elif enum_value == DfNames.SCALING_OUTLIER:
        path = 'data/scaling_outlier_v3'
    elif enum_value == DfNames.MISSING_SCALING_OUTLIER:
        path = 'data/missing_scaling_outlier_v3'
    elif enum_value == DfNames.MISSING_SCALING_OUTLIER_SECOND:
        path = 'data/imputed_missing_scaling_outlier_v3'    
    else:
        raise ValueError(f'Unknown value: {enum_value}')

    return os.path.join(path_prefix, path)


def get_config_paths(enum_value: DfNames, path_prefix: str = ''):
    
    path = ''
    if enum_value == DfNames.BASE:
        path = 'configs/config_data_base_v2.yaml'
    elif enum_value == DfNames.MISSING:
        path = 'configs/config_data_missing_v3.yaml'
    elif enum_value == DfNames.OUTLIER:
        path = 'configs/config_data_outlier_v3.yaml'
    elif enum_value == DfNames.SCALING:
        path = 'configs/config_data_scaling_v3.yaml'
    elif enum_value == DfNames.MISSING_OUTLIER:
        path = 'configs/config_data_missing_outlier_v3.yaml'
    elif enum_value == DfNames.MISSING_SCALING:
        path = 'configs/config_data_missing_scaling_v3.yaml'
    elif enum_value == DfNames.SCALING_OUTLIER:
        path =  'configs/config_data_scaling_outlier_v3.yaml'
    elif enum_value == DfNames.MISSING_SCALING_OUTLIER:
        path = 'configs/config_data_missing_scaling_outlier_v3.yaml'
    else:
        raise ValueError(f'Unknown value: {enum_value}')

    return os.path.join(path_prefix, path)





def get_name_by_enum(enum_value):
    if enum_value == DfNames.BASE:
        return 'base'
    if enum_value == DfNames.MISSING:
        return 'missing'
    if enum_value == DfNames.MISSING_SECOND:
        return 'missing_second'
    if enum_value == DfNames.OUTLIER:
        return 'outlier'
    if enum_value == DfNames.SCALING:
        return 'scaling'
    if enum_value == DfNames.MISSING_OUTLIER:
        return 'missing_outlier'
    if enum_value == DfNames.MISSING_OUTLIER_SECOND:
        return 'missing_outlier_second'
    if enum_value == DfNames.MISSING_SCALING:
        return 'missing_scaling'
    if enum_value == DfNames.MISSING_SCALING_SECOND:
        return 'missing_scaling_second'
    if enum_value == DfNames.SCALING_OUTLIER:
        return 'scaling_outlier_second'
    if enum_value == DfNames.MISSING_SCALING_OUTLIER:
        return 'missing_scaling_outlier'
    if enum_value == DfNames.MISSING_SCALING_OUTLIER_SECOND:
        return 'missing_scaling_outlier_second'    
    raise ValueError(f'Unknown value: {enum_value}')

def print_main_stats(df: DataFrame, name: str):
    print(f'Datasets: {len(df)}' )
    print(f'{name} mean base: {df["base_ari"].mean()}' )
    print(f'{name} median base: {df["base_ari"].median()}' )
    
    print(f'{name} mean modified: {df["modified_ari"].mean()}' )
    print(f'{name} median modified: {df["modified_ari"].median()}' )
    print(f'{name} mean optimized: {df["best_fitness"].mean()}' )
    print(f'{name} median optimized: {df["best_fitness"].median()}' )
    


def get_dfs_by_names(names: list, prefix = ''):
    res = {}
    
    if DfNames.BASE in names:
        res[DfNames.BASE] = pd.read_csv(os.path.join(prefix, 'res/final/b.csv'))
        
    if DfNames.MISSING in names:
        res[DfNames.MISSING] =pd.read_csv(os.path.join(prefix, 'res/final/m_1.csv'))
        
    if DfNames.MISSING_SECOND in names:
        res[DfNames.MISSING_SECOND] = pd.read_csv(os.path.join(prefix, 'res/final/m_2.csv'))
        
    if DfNames.OUTLIER in names:
        res[DfNames.OUTLIER] = pd.read_csv(os.path.join(prefix, 'res/final/o.csv'))
        
    if DfNames.SCALING in names:
        res[DfNames.SCALING] = pd.read_csv(os.path.join(prefix, 'res/final/s.csv'))
        
    if DfNames.MISSING_OUTLIER in names:
        res[DfNames.MISSING_OUTLIER] = pd.read_csv(os.path.join(prefix, 'res/final/mo_1.csv'))
        
    if DfNames.MISSING_OUTLIER_SECOND in names:
        res[DfNames.MISSING_OUTLIER_SECOND] = pd.read_csv(os.path.join(prefix, 'res/final/mo_2.csv'))
        
    if DfNames.MISSING_SCALING in names:
        res[DfNames.MISSING_SCALING] = pd.read_csv(os.path.join(prefix, 'res/final/mo_1.csv'))    
        
    if DfNames.MISSING_SCALING_SECOND in names:
        res[DfNames.MISSING_SCALING_SECOND] =  pd.read_csv(os.path.join(prefix, 'res/final/ms_2.csv'))
        
    if DfNames.SCALING_OUTLIER in names:
        res[DfNames.SCALING_OUTLIER] = pd.read_csv(os.path.join(prefix, 'res/final/so.csv'))
        
    if DfNames.MISSING_SCALING_OUTLIER in names:
        res[DfNames.MISSING_SCALING_OUTLIER] =  pd.read_csv(os.path.join(prefix, 'res/final/mso_1.csv'))
    if DfNames.MISSING_SCALING_OUTLIER_SECOND in names:
        res[DfNames.MISSING_SCALING_OUTLIER_SECOND] = pd.read_csv(os.path.join(prefix, 'res/final/mso_2.csv'))
    if DfNames.ALL in names:
        res[DfNames.BASE] = pd.read_csv(os.path.join(prefix, 'res/final/b.csv'))
        res[DfNames.MISSING] =pd.read_csv(os.path.join(prefix, 'res/final/m_1.csv'))
        res[DfNames.MISSING_SECOND] = pd.read_csv(os.path.join(prefix, 'res/final/m_2.csv'))
        res[DfNames.OUTLIER] = pd.read_csv(os.path.join(prefix, 'res/final/o.csv'))
        res[DfNames.SCALING] = pd.read_csv(os.path.join(prefix, 'res/final/s.csv'))
        res[DfNames.MISSING_OUTLIER] = pd.read_csv(os.path.join(prefix, 'res/final/mo_1.csv'))
        res[DfNames.MISSING_OUTLIER_SECOND] = pd.read_csv(os.path.join(prefix, 'res/final/mo_2.csv'))
        res[DfNames.MISSING_SCALING] = pd.read_csv(os.path.join(prefix, 'res/final/ms_1.csv'))    
        res[DfNames.MISSING_SCALING_SECOND] =  pd.read_csv(os.path.join(prefix, 'res/final/ms_2.csv'))
        res[DfNames.SCALING_OUTLIER] = pd.read_csv(os.path.join(prefix, 'res/final/so.csv'))
        res[DfNames.MISSING_SCALING_OUTLIER] =  pd.read_csv(os.path.join(prefix, 'res/final/mso_1.csv'))
        res[DfNames.MISSING_SCALING_OUTLIER_SECOND] = pd.read_csv(os.path.join(prefix, 'res/final/mso_2.csv'))
    
    return res

def extend_list(lst, length, increment=True):
    if not lst:
        return lst
    last_element = lst[-1]
    extended = lst + [last_element + i if increment else last_element for i in range(1, length - len(lst) + 1)]
    return extended[:length]


def preprocess_to_average_fitness(df: DataFrame):
    df['deap_max'] = df['deap_max'].apply(ast.literal_eval)
    df['generations'] = df['generations'].apply(ast.literal_eval)

    df = df[(df['deap_max'].apply(len) > 0) & (df['generations'].apply(len) > 0)]


    max_length = max(df['deap_max'].apply(len).max(), df['generations'].apply(len).max())

    df = df[['deap_max', 'generations']].copy()
    df['deap_max'] = df['deap_max'].apply(lambda x: extend_list(x, max_length, False))
    df['generations'] = df['generations'].apply(lambda x: extend_list(x, max_length, True))


    df = df.apply(pd.Series.explode).reset_index(drop=True)
    
    return df.groupby('generations')['deap_max'].mean()



def get_first_stage_by_name(enum_value: DfNames):
    
    if enum_value == DfNames.MISSING_SECOND:
        return DfNames.MISSING
    if enum_value == DfNames.MISSING_SCALING_SECOND:
        return DfNames.MISSING_SCALING
    if enum_value == DfNames.MISSING_OUTLIER_SECOND:
        return DfNames.MISSING_OUTLIER
    if enum_value == DfNames.MISSING_SCALING_OUTLIER_SECOND:
        return DfNames.MISSING_SCALING_OUTLIER
    
def get_second_stage_by_name(enum_value: DfNames):
    
    if enum_value == DfNames.MISSING:
        return DfNames.MISSING_SECOND
    if enum_value == DfNames.MISSING_SCALING:
        return DfNames.MISSING_SCALING_SECOND
    if enum_value == DfNames.MISSING_OUTLIER:
        return DfNames.MISSING_OUTLIER_SECOND
    if enum_value == DfNames.MISSING_SCALING_OUTLIER:
        return DfNames.MISSING_SCALING_OUTLIER_SECOND

    
if __name__ == '__main__':
    print_main_stats(pd.DataFrame())