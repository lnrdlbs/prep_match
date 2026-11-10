import warnings
from numpy import ndarray
from pymfe.mfe import MFE
from sklearn.datasets import make_blobs


def get_all_meta_features(X: ndarray)->dict:
    mfe = MFE(groups='all', features=['nr_attr', 
                                        'nr_bin',
                                        'nr_cat', 
                                        'nr_inst',
                                        'nr_num',
                                        'num_to_cat',
                                        'cor',
                                        'cov',
                                        'eigenvalues',
                                        'iq_range',
                                        'kurtosis',
                                        'mad',
                                        'max',
                                        'mean',
                                        'median',
                                        'min',
                                        'nr_cor_attr',
                                        'nr_norm',
                                        'nr_outliers',
                                        'range',
                                        'sd',
                                        'skewness',
                                        'sparsity',
                                        't_mean',
                                        'var',
                                        't2',
                                        't3',
                                        't4',
                                        'cohesiveness',
                                        'wg_dist',
                                        ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mfe.fit(X, verbose=False)

        mf = mfe.extract()
        mf_dict = dict(zip(mf[0], mf[1]))
    return mf_dict

def get_mf_by_list(X: ndarray, wanted_mf: dict) -> dict:
    mfe = MFE(groups='all', features=wanted_mf)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mfe.fit(X, verbose=False)

        mf = mfe.extract()
        mf_dict = dict(zip(mf[0], mf[1]))
    return mf_dict
def get_all_meta_features_median(X: ndarray)->dict:
    mfe = MFE(groups='all', features=['nr_attr', 
                                        'nr_bin',
                                        'nr_cat', 
                                        'nr_inst',
                                        'nr_num',
                                        'num_to_cat',
                                        'cor',
                                        'cov',
                                        'eigenvalues',
                                        'iq_range',
                                        'kurtosis',
                                        'mad',
                                        'max',
                                        'mean',
                                        'median',
                                        'min',
                                        'nr_cor_attr',
                                        'nr_norm',
                                        'nr_outliers',
                                        'range',
                                        'sd',
                                        'skewness',
                                        'sparsity',
                                        't_mean',
                                        'var',
                                        't2',
                                        't3',
                                        't4',
                                        'cohesiveness',
                                        'wg_dist',
                                        ], summary=['median', 'mean', 'sd'])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mfe.fit(X, verbose=False)

        mf = mfe.extract()
        mf_dict = dict(zip(mf[0], mf[1]))
    return mf_dict
if __name__ == '__main__':
    X, _= make_blobs(10,2, centers=-1, cluster_std=1.5)
    get_all_meta_features(X=X)