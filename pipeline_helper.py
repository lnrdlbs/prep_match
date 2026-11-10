
import random
import time
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
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, PowerTransformer, QuantileTransformer, RobustScaler
from sklearn.random_projection import GaussianRandomProjection
from sklearn.svm import OneClassSVM
from data import Data
from helper import get_ari


pp_missing_value_methods = {
    'simple': {'strategy': ['mean', 'median', 'most_frequent']},
    'knn': {'n_neighbors': range(1,15)},
    #'iterative': {'estimator': ['bayesian', 'randomforest'], 'max_iter':range(5,20), 'initial_strategy': ['mean', 'median', 'most_frequent'], 'imputation_order': ['ascending', 'descending', 'roman', 'arabic', 'random'] },
}


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
        'elliptic': {'assume_centered': [True, False], 'contamination': [0.1, 0.2, 0.3, 0.4, 0.5]},
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

pp_methods_flat = {
    
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
    'elliptic': {'assume_centered': [True, False], 'contamination': [0.1, 0.2, 0.3, 0.4, 0.5]},
    'sdg_one_class_svm': {'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'], 'warm_start': [True, False], 'nu': [0.3,0.5,0.7], 'max_iter': [100,500,1000], 'eta0': [0.1,0.2,0.3]},

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
    'max_abs': {'placeholder': ['placeholder']},
    'pca': {'n_components': ['mle', 2,5,10,25,35], 'whiten': [True, False], 'svd_solver': ['auto', 'full', 'arpack', 'randomized'], 'power_iteration_normalizer': ['auto','QR', 'LU']},
            'grp': {'n_components': ['auto', 2,5,10,25,35], 'eps': [0.01, 0.1, 0.3]},
            'lle': {'p_neighbors':  [0.01, 0.05, 0.1], 'n_components': [2,5,10,25,35], 'eigensolver':['auto', 'arpack', 'dense'],  'max_iter':  [50,100,500,1000], 'method': ['standard', 'hessian', 'modified', 'ltsa'], 'neighbors_algorithm': ['auto', 'brute', 'kd_tree', 'ball_tree']},
}

pp_methods_flat_include_mi = {
    
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
    'elliptic': {'assume_centered': [True, False], 'contamination': [0.1, 0.2, 0.3, 0.4, 0.5]},
    'sdg_one_class_svm': {'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'], 'warm_start': [True, False], 'nu': [0.3,0.5,0.7], 'max_iter': [100,500,1000], 'eta0': [0.1,0.2,0.3]},

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
    'max_abs': {'placeholder': ['placeholder']},
    'pca': {'n_components': ['mle', 2,5,10,25,35], 'whiten': [True, False], 'svd_solver': ['auto', 'full', 'arpack', 'randomized'], 'power_iteration_normalizer': ['auto','QR', 'LU']},
            'grp': {'n_components': ['auto', 2,5,10,25,35], 'eps': [0.01, 0.1, 0.3]},
            'lle': {'p_neighbors':  [0.01, 0.05, 0.1], 'n_components': [2,5,10,25,35], 'eigensolver':['auto', 'arpack', 'dense'],  'max_iter':  [50,100,500,1000], 'method': ['standard', 'hessian', 'modified', 'ltsa'], 'neighbors_algorithm': ['auto', 'brute', 'kd_tree', 'ball_tree']},
    'simple': {'strategy': ['mean', 'median', 'most_frequent']},
    'knn': {'n_neighbors': range(1,15)},
}


def init_mv_hyperparameters(method):
    hps = {}
    for param, values in pp_missing_value_methods[method].items():
        hps[param] = np.random.choice(values) if isinstance(values, list) else np.random.randint(values.start, values.stop)
    return hps
def mutate_iso_max_features(data_object: Data): 
    
    available_values = pp_methods['outlier']['iso_forest']['max_features'].copy()
    max_f = np.random.choice(available_values)

    #print(f'mut iso {max_f}, {data_object.n_features}')
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

def mutate_n_components(data_object: Data, method): 
    valid_values = pp_methods['dim_red'][method]['n_components']
    c = np.random.choice(valid_values)
    if c == 'auto': return c
    c = int(c)
    if c > data_object.n_features:
        return data_object.n_features
    return c
def mutate_svd_solver(hps: dict, method: str):
    all_solvers =  pp_methods['dim_red'][method]['svd_solver'].copy()
    if hps['n_components'] == 'mle':
        return 'auto'
    else:
        return random.choice(all_solvers)

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


def evaluate(data_object: Data, individual, seed=876):
    
    
    np.random.seed(seed=seed)
    data_object.X_preprocessed = data_object.X_modified.copy()
    data_object.y_preprocessed = data_object.y_modified.copy()
    
    individual[2]= sorted(individual[2], key=lambda x: x[0])
    invalid = False
    
    try:
        for el in individual[2]:
            method = el[1]

            hps = individual[1][method]
            
            if method == 'knn':
                imputer = KNNImputer(n_neighbors=hps['n_neighbors'])
                data_object.X_preprocessed = imputer.fit_transform(data_object.X_missing_values)
                data_object.X_modified = data_object.X_preprocessed.copy()
                data_object.y_preprocessed = data_object.y_missing_values.copy()
                data_object.y_modified = data_object.y_missing_values.copy()
            elif method == 'simple':
                imputer = SimpleImputer(strategy=hps['strategy'])
                data_object.X_preprocessed = imputer.fit_transform(data_object.X_missing_values)
                data_object.X_modified = data_object.X_preprocessed.copy()
                data_object.y_preprocessed = data_object.y_missing_values.copy()
                data_object.y_modified = data_object.y_missing_values.copy()
            
            elif method == 'lof':
                n_neighbors = int(hps['p_neighbors'] * len(data_object.X_preprocessed))
                if n_neighbors <= 0: n_neighbors = 1
                leaf_size =  int(hps['p_leaf_size'] * len(data_object.X_preprocessed))
                if hps['contamination'] == 'auto': 
                    contamination = 'auto'
                else: contamination = float(hps['contamination'])
                try:
                    operator = LocalOutlierFactor(n_neighbors=n_neighbors, leaf_size=leaf_size, p=hps['p'], algorithm=hps['algorithm'], contamination=contamination)
                    is_inlier = operator.fit_predict(data_object.X_preprocessed)
                    data_object.X_preprocessed = data_object.X_preprocessed[is_inlier==1]
                    data_object.y_preprocessed = data_object.y_preprocessed[is_inlier==1]
                except ValueError as e: 
                    print(f'Error eval lof: {e}')
                    invalid = True
            elif method == 'iso_forest':
                if hps['max_features'] == 'auto': 
                    max_features = 'auto'
                else: 
                    if hps['max_features'] > len(data_object.X_preprocessed[0]):
                        max_features = np.random.randint(1, len(data_object.X_preprocessed[0]))
                    else:
                        max_features = int(hps['max_features'])
                
                try:
                    operator = IsolationForest(n_estimators=hps['n_estimators'], max_features=max_features, bootstrap=hps['bootstrap'], warm_start=hps['warm_start'], random_state=876)
                    is_inlier = operator.fit_predict(data_object.X_preprocessed)
                
                    data_object.X_preprocessed = data_object.X_preprocessed[is_inlier==1]
                    data_object.y_preprocessed = data_object.y_preprocessed[is_inlier==1]
                except ValueError as e:
                    print(f'Error eval iso: {e}')
                    invalid = True

            elif method == 'power_transformer': 
                try:
                    operator = PowerTransformer(standardize=hps['standardize'])
                    data_object.X_preprocessed = operator.fit_transform(data_object.X_preprocessed)
                except scipy.optimize._optimize.BracketError as e:
                    print(f'Error eval power: {e}')
                    invalid = True
                    
            elif method == 'one_class_svm':
                try:
                    operator = OneClassSVM(kernel=hps['kernel'], gamma=hps['gamma'], nu=hps['nu'], shrinking=hps['shrinking'], max_iter=hps['max_iter'])
                    is_inlier = operator.fit_predict(data_object.X_preprocessed)
                    data_object.X_preprocessed = data_object.X_preprocessed[is_inlier==1]
                    data_object.y_preprocessed = data_object.y_preprocessed[is_inlier==1]
                except ValueError as e:
                    print(f'Error eval one_class_svm: {e}')
                    invalid = True
            elif method == 'elliptic':
                try:
                    operator = EllipticEnvelope(assume_centered=hps['assume_centered'], contamination=hps['contamination'], random_state=876)
                    is_inlier = operator.fit_predict(data_object.X_preprocessed)
                    data_object.X_preprocessed = data_object.X_preprocessed[is_inlier==1]
                    data_object.y_preprocessed = data_object.y_preprocessed[is_inlier==1]
                except ValueError as e:
                    print(f'Error eval elliptic: {e}')
                    invalid = True
                except UnboundLocalError as e:
                    print(f'Internal Error eval elliptic: {e}')
                    invalid = True
                    
            elif method == 'sdg_one_class_svm':
                try:
                    operator = SGDOneClassSVM(learning_rate=hps['learning_rate'], warm_start=hps['warm_start'], nu=hps['nu'], max_iter=hps['max_iter'], eta0=hps['eta0'], random_state=876)
                    is_inlier = operator.fit_predict(data_object.X_preprocessed)
                    data_object.X_preprocessed = data_object.X_preprocessed[is_inlier==1]
                    data_object.y_preprocessed = data_object.y_preprocessed[is_inlier==1]
                except ValueError as e:
                    print(f'Error eval sdg  {e}')
                    invalid = True
            elif method == 'robust':
                try:
                    operator = RobustScaler(with_centering=hps['with_centering'], with_scaling=hps['with_scaling'], unit_variance=hps['unit_variance'])
                    data_object.X_preprocessed = operator.fit_transform(data_object.X_preprocessed)
                except ValueError as e:
                    print(f'Error eval robust  {e}')
                    invalid = True

            elif method == 'standard_scaler':
                try:
                    operator = StandardScaler(with_mean=hps['with_mean'], with_std=hps['with_std'])
                    data_object.X_preprocessed = operator.fit_transform(data_object.X_preprocessed)
                except ValueError as e:
                    print(f'Error eval standard  {e}')
                    invalid = True
            elif method == 'min_max':
                try:
                    operator = MinMaxScaler()
                    data_object.X_preprocessed = operator.fit_transform(data_object.X_preprocessed)
                except ValueError as e:
                    print(f'Error eval min_max  {e}')
                    invalid = True
            elif method == 'quantile':
                n_quantiles = int(hps['p_quantiles'] * len(data_object.X_preprocessed))
                try:
                    operator = QuantileTransformer(n_quantiles=n_quantiles, output_distribution=hps['output_distribution'], random_state=876)
                    data_object.X_preprocessed = operator.fit_transform(data_object.X_preprocessed)
                except ValueError as e:
                    print(f'Error eval quantile  {e}')
                    invalid = True
            elif method == 'max_abs':
                try:
                    operator = MaxAbsScaler()
                    data_object.X_preprocessed = operator.fit_transform(data_object.X_preprocessed)
                except ValueError as e:
                    print(f'Error eval max_abs  {e}')
                    invalid = True
            elif method == 'pca':
                
                if hps['n_components'] == 'mle':
                    n_components = 'mle'
                else:
                    if int(hps['n_components']) >= min(len(data_object.X_preprocessed), len(data_object.X_preprocessed[0])):
                        n_components = np.random.randint(low=0, high=min(len(data_object.X_preprocessed), len(data_object.X_preprocessed[0])))
                    else:
                        n_components = int(hps['n_components'])
                try: 
                    operator = PCA(n_components=n_components, whiten=hps['whiten'], svd_solver=hps['svd_solver'],  power_iteration_normalizer=hps['power_iteration_normalizer'], random_state=876)
                    data_object.X_preprocessed = operator.fit_transform(data_object.X_preprocessed)
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
                        if int(hps['n_components']) >= min(len(data_object.X_preprocessed), len(data_object.X_preprocessed[0])):
                            n_components = np.random.randint(low=0, high=min(len(data_object.X_preprocessed), len(data_object.X_preprocessed[0])))
                        else:
                            n_components = int(hps['n_components'])
                    operator = GaussianRandomProjection(n_components=n_components, eps=hps['eps'], random_state=876)
            
                    data_object.X_preprocessed = operator.fit_transform(data_object.X_preprocessed)
                except  ValueError as e:
                    print(f'Error eval grp {e}')
                    invalid = True
                except TypeError as e:
                    print(f'Error eval grp type {e}')
                    invalid = True
                    
            elif method == 'lle':
                n_neighbors = int(hps['p_neighbors'] * len(data_object.X_preprocessed))
                try:
                    if hps['n_components'] > len(data_object.X_preprocessed[0]):
                        n_components = np.random.randint(1,  len(data_object.X_preprocessed[0]))
                    else:
                        n_components=hps['n_components']
                    operator = LocallyLinearEmbedding(n_neighbors=n_neighbors,n_components=n_components, eigen_solver=hps['eigensolver'], max_iter=hps['max_iter'], method=hps['method'], neighbors_algorithm=hps['neighbors_algorithm'], random_state=876)
                    data_object.X_preprocessed = operator.fit_transform(data_object.X_preprocessed)
                except ValueError as e:
                    print(f'Error eval lle {e}')
                    invalid = True
            else:
                raise ValueError(f'Eval: Unknown method: {method}')
        
        if  np.isnan(data_object.X_preprocessed).any():
            print(f'Nans introduced {individual}')
            invalid = True
        
        if len(data_object.X_preprocessed) < data_object.n_clusters:
            print(f'To few samples s: {len(data_object.X_preprocessed)}, c: {data_object.n_clusters}')
            invalid = True
        if invalid: 
            data_object.preprocessed_ari = -0.5
            return data_object, invalid
    except Exception as e:
        print(f'Unknown error during eval {e}')
        invalid = True
        data_object.preprocessed_ari = -0.5
    eval_end = time.time()
    #print(f'eval time: {eval_end-start}')
    
    data_object.preprocessed_ari = get_ari(data_object.X_preprocessed, data_object.y_preprocessed, data_object.n_clusters)
    print(f'Len X after pp: {len(data_object.X_preprocessed)}')
    return data_object, invalid