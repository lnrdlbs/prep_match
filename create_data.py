import argparse
import itertools
import os
from pathlib import Path
import pickle
import uuid
import warnings
from numpy import ndarray
from numpy.random import RandomState, SeedSequence, MT19937
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.neighbors import KNeighborsClassifier, KernelDensity
import numpy as np
import yaml
from data import Data

from helper import get_numpy_rs, replace_nan_with_constant
from meta_features import get_all_meta_features

warnings.filterwarnings("ignore")


def create_base_data(
    shape: str = "blobs",
    n_samples: int = 300,
    n_clusters: int = 3,
    n_features: int = 2,
    cluster_std: float = 1,
    noise: float = 0,
    factor: float = 0.8,
    seed: int = 213412,
) -> tuple[ndarray, ndarray]:
    """This function creates the a data array and corresponding labels based on the sklearn functions
        `make_blobs`, `make_circles`, and `make_moons`

    Parameters
    ----------
    shape: String, default: blobs
        Selection of sklearn function, valid are `blobs`, `moons`, and `circles`
    n_samples: Int, default: 300
        Number of samples (rows)
    n_clusters: Int, default=3:
       Number of clusters
    n_features: Int, default=2
        Number of features (columns)
    cluster_std: Float, default: 1
        The standard deviation of the clusters. Only applicable if the shape is `blobs`
    noise: Float, default: 0,
        Standard deviation of Gaussian noise added to the data. Only applicable if the shape is `moons` or `circles`
    factor: Float, default: 0.8,
        Scale factor between inner and outer circle in the range [0, 1). Only applicable if the shape is `circles`

    Returns
    -------
    X: ndarray,
        The generated samples
    Y: ndarray,
        The integer labels for cluster membership of each sample.
    """

    assert (
        shape == "blobs" or shape == "circles" or shape == "moons"
    ), "Shape parameter must be either blobs, moons or circles"
    assert n_samples > 0, "n_samples must be >=1"
    assert n_clusters > 1, "n_cluster must be > 1"
    assert n_features > 1, "n_features must be >1"
    assert noise >= 0, "noise must be >=0"
    assert factor >= 0 and factor < 1, "factor must be >=0 and <1"

    if shape == "blobs":
        X, y, _ = make_blobs(
            n_samples=n_samples,
            centers=n_clusters,
            n_features=n_features,
            cluster_std=cluster_std,
            random_state=seed,
            return_centers=True,
        )
    elif shape == "moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    elif shape == "circles":
        X, y = make_circles(
            n_samples=n_samples, noise=noise, factor=factor, random_state=seed
        )
    else:
        raise ValueError("Unknown shape")
    return X, y


def remove_data(
    X: ndarray,
    y: ndarray = None,
    remove_percentage: float = 0.15,
    feature_indices: ndarray = None,
    cluster_indices: ndarray = None,
    value_range: tuple = None,
    replacement_value=np.nan,
) -> ndarray:
    """Function for removing certain parts of an array (replaced with `np.nan`)

    Parameters
    ----------
    X: ArrayLike, default: none
        The array where the elements should be removed of
    y: ArrayLike, default: none
        Labels of X, only used if the `cluster_indices` is used
    remove_percentage: Float, default=0.15:
        How many percent of the data (X) should be removed (between 0 and 1, exclusive)
    feature_indices: ArrayLike, default=none
        Describes in which features the values are removed, if None then values are removed from all features
    cluster_indices: ArrayLike, default: none
        Describes from which clusters the values are to be removed, if none then the values are removed from all clusters
    value_range: tuple, default: none
        Describe the range of the to be removed values (inclusive), if none then all values are considered

    Returns
    -------
    X: ndarray,
        The array with missing data (np.nan), meeting all requirements
    """
    assert (
        remove_percentage < 1 and remove_percentage > 0
    ), "remove percentage must be between 0 and 1"
    assert len(X) > 0, "Array X must not be empty"

    number_to_remove = int(X.size * remove_percentage)

    combined_mask = np.ones(X.shape, dtype=bool)
    if not feature_indices is None:
        assert np.any(feature_indices < X.shape[1]), "Feature index out of bounds (X)"
        selected_features = X[:, feature_indices].copy()
        assert (
            selected_features.size > number_to_remove
        ), "Feature restriction leads to empty features, reduce `remove_percentage`"
        tmp_column_indices = np.tile(np.arange(X.shape[1]), (X.shape[0], 1))
        feature_mask = np.isin(tmp_column_indices, selected_features)

        combined_mask = combined_mask & feature_mask

    if not cluster_indices is None:
        assert y is not None, "y must be not none if cluster indices are provided"
        assert np.any(y < cluster_indices), "Cluster index out of bounds (y)"
        cluster_mask = np.isin(y, cluster_indices)
        selected_clusters = X[cluster_mask].copy()
        assert (
            selected_clusters.size > number_to_remove
        ), "Cluster restriction can not meet `remove_percentage`, reduce `remove_percentage`"

        combined_mask = combined_mask & cluster_mask

    if not value_range is None:
        value_mask = (X >= value_range[0]) & (X <= value_range[1])
        selected_values = X[value_mask].copy()
        assert (
            selected_values.size > number_to_remove
        ), "Value range restriction can not meet `remove percentage`, reduce `remove percentage`"
        combined_mask = combined_mask & value_mask

    allowed_indices = list(zip(np.where(combined_mask)[0], np.where(combined_mask)[1]))

    while number_to_remove > 0:
        index = allowed_indices[np.random.choice(len(allowed_indices))]

        if X[index] is not replacement_value:
            X[index] = replacement_value
            number_to_remove -= 1

    return X


def scale_data(
    X: ndarray,
    scaling_factor: int = 10,
    features: ndarray = None,
    method: str = "factor",
    seed: int = 876,
) -> ndarray:
    """
    Scales selected features by the provided `scaling_factor`.
    If no features are given, a random feature is selected for scaling

    Parameters
    ----------
    X: ArrayLike, default: none
        The array where feature(s) are scaled
    scaling_factor: Int, default: 10
        The factor by which a feature is multiplied (>0)
    features: ArrayLike, default: none
        A feature array to which the scaling factor is applied, if `None`
    method: str, default: `factor`
        Decides how the data scaled, options are `factor`, `square`, and `exp`
    seed: int, default: 876
        Used for reproducibility reasons
    Returns
    -------
    X: ndarray,
        The scaled array X
    """
    assert len(X) > 0, "Array X must not be empty"
    assert scaling_factor > 0, "Scaling factor must be >0"

    rs = get_numpy_rs(seed)

    if features is None:
        features = [rs.randint(X.shape[1])]

    for feature_index in features:
        if method == "factor":
            X[:, feature_index] = X[:, feature_index] * scaling_factor
        elif method == "exp":
            X[:, feature_index] = np.exp(X[:, feature_index])
        elif method == "square":
            X[:, feature_index] = np.square(X[:, feature_index])

    return X


def approximate_bounding_sphere(points):
    centroid = np.mean(points, axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)
    radius = np.max(distances)
    return centroid, radius


def generate_points_outside_full_random(X, centers, radii, n_points=10, seed=876):
    rs = RandomState(MT19937(SeedSequence(seed)))
    random_points = np.zeros((n_points, X.shape[1]))

    ranges = {}
    for dim in range(X.shape[1]):
        min_val = np.min(X[:, dim])
        max_val = np.max(X[:, dim])
        range_extensions = 0.2 * (max_val - min_val)
        extended_min = min_val - range_extensions
        extended_max = max_val + range_extensions
        ranges[dim] = (extended_min, extended_max)

    i = 0
    while n_points > 0:

        success = True
        random_point = np.zeros((1, X.shape[1]))
        for dim, rng in ranges.items():
            random_point[:, dim] = rs.uniform(rng[0], rng[1])

        # Check if the point is outside all spheres
        for center, radius in zip(centers, radii):
            if np.linalg.norm(random_point - center) <= radius:
                success = False
                break

        if success:
            random_points[i] = random_point

            n_points = n_points - 1
            i = i + 1

    return random_points


def generate_points_outside_distance(X, centers, radii, n_points=10):
    rs = RandomState(MT19937(SeedSequence(876)))
    random_points = np.zeros((n_points, X.shape[1]))
    i = 0
    while n_points > 0:

        success = True
        direction = rs.normal(size=centers.shape[1])
        direction /= np.linalg.norm(direction)
        random_radius = rs.uniform(1.2, 2.5) * max(radii)
        random_point = centers[rs.randint(len(centers))] + direction * random_radius

        for center, radius in zip(centers, radii):
            if np.linalg.norm(random_point - center) <= radius:
                success = False
                break

        if success:
            random_points[i] = random_point

            n_points = n_points - 1
            i = i + 1

    return np.array(random_points)


def add_outliers_sphere(
    X: ndarray, y: ndarray, p_random_points: int = 0.1, seed: int = 12341234
):

    n_points = int(len(X) * p_random_points)
    spheres = [approximate_bounding_sphere(X[y == label]) for label in np.unique(y)]
    centers = np.array([s[0] for s in spheres])
    radii = np.array([s[1] for s in spheres])

    outlier = generate_points_outside_full_random(
        X, centers, radii, n_points=n_points, seed=seed
    )

    return outlier


def add_outliers_kde(X: ndarray, y, number_outliers=10, bandwidth=0.04, level=0.01):

    cluster_kdes = {}
    for label in np.unique(y):
        cluster_data = X[y == label]
        # x_min, x_max = cluster_data[:, 0].min() - 1, cluster_data[:, 0].max() + 1
        # y_min, y_max = cluster_data[:, 1].min() - 1, cluster_data[:, 1].max() + 1
        # xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
        kde.fit(cluster_data)
        cluster_kdes[label] = kde
        # Z = np.exp(kde.score_samples(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)

    ranges = {}
    for dim in range(X.ndim):
        min_val = np.min(X[:, dim])
        max_val = np.max(X[:, dim])
        range_extensions = 0.2 * (max_val - min_val)
        extended_min = min_val - range_extensions
        extended_max = max_val + range_extensions
        ranges[dim] = (extended_min, extended_max)

    random_points = np.zeros((number_outliers, X.ndim))
    i = 0
    while number_outliers > 0:

        random_point = np.zeros((1, X.ndim))
        for dim, rng in ranges.items():
            random_point[:, dim] = np.random.uniform(rng[0], rng[1])

        for label in np.unique(y):
            kde = cluster_kdes[label]
            density = np.exp(kde.score_samples([random_point]))
            if density < level:
                random_points[i] = random_point
                number_outliers = number_outliers - 1
                i = i + 1
    return np.array(random_points)


def get_outlier_labels(X: ndarray, y: ndarray, outliers: ndarray):
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(X, y)

    labels = neigh.predict(outliers)
    return labels


def read_data_config(path: str):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def generate_data_from_config(path: str, start_id: int, output_folder: str) -> list[Data]:
    config = read_data_config(path)
    seed = config["seeds"]

    data_config_values = list(itertools.product(*config["data"].values()))
    data_config_keys = config["data"].keys()

    data_configs = [dict(zip(data_config_keys, value)) for value in data_config_values]

    data_objects_to_generate: list[Data] = list()
    data_id = 0
    total = len(data_configs)
    i = 0
    while i < len(data_configs):

        print(f"Generate Data with id {data_id} of {total}")
        dc = data_configs[i]
        print(dc)

        data_id = i+1
        if data_id < start_id:
            print("skipping")
            continue
        data = Data()
        data.id = data_id
        data.uuid = uuid.uuid1()
        data.seed = seed[0]
        data.shape = dc["shape"]
        data.n_samples = dc["n_samples"]
        data.n_clusters = dc["n_clusters"]
        data.n_features = dc["n_features"]
        data.cluster_std = dc["cluster_std"]

        X, y = create_base_data(
            shape=dc["shape"],
            n_samples=dc["n_samples"],
            n_clusters=dc["n_clusters"],
            n_features=dc["n_features"],
            cluster_std=dc["cluster_std"],
            seed=seed[0],
        )
        data.X_base = X
        data.y = y

        data.mf_base = get_all_meta_features(data.X_base)

        data.y_modified = data.y.copy()
        data.X_modified = data.X_base.copy()
        if "multiplier_methods" in dc.keys() and not dc["multiplier_methods"] == "None":
            data.X_modified = scale_data(
                data.X_base.copy(), method=dc["multiplier_methods"]
            )
            data.has_scaling = True
            data.multiplier_method = dc["multiplier_methods"]
        else:
            data.has_scaling = False
            data.multiplier_method = None
        if "p_outlier" in dc.keys() and not dc["p_outlier"] == 0:
            outliers = add_outliers_sphere(
                data.X_modified, data.y, p_random_points=dc["p_outlier"], seed=data.seed
            )
            outlier_labels = get_outlier_labels(
                data.X_modified, data.y_modified, outliers
            )

            data.X_modified = np.concatenate([data.X_modified, outliers])
            data.y_modified = np.concatenate([data.y_modified, outlier_labels])
            data.X_missing_values = data.X_modified.copy()
            data.y_missing_values = data.y_modified.copy()

            data.has_outlier = True
            data.p_outlier = dc["p_outlier"]
        else:
            data.has_outlier = False
            data.p_outlier = None

        if "p_missing_value" in dc.keys() and not dc["p_missing_value"] == 0:
            data.X_missing_values = remove_data(
                data.X_modified.copy(), remove_percentage=dc["p_missing_value"]
            )
            data.has_missing_values = True
            data.p_missing_values = dc["p_missing_value"]
            data.y_missing_values = data.y_modified.copy()
            data.X_modified = replace_nan_with_constant(
                data.X_missing_values.copy(), constant=0
            )
            data.mf_missing_values = get_all_meta_features(data.X_missing_values)
        else:
            data.has_missing_values = False
            data.p_missing_values = 0
            data.X_missing_values = None
            data.mf_missing_values = None

            data.X_missing_values = data.X_modified.copy()
            data.y_missing_values = data.y_modified.copy()

        data.mf_modified = get_all_meta_features(data.X_modified)

        assert len(data.X_modified) == len(
            data.y_modified
        ), f" {len(data.X_modified)}, {len(data.y_modified)}"
        assert len(data.X_missing_values) == len(
            data.y_missing_values
        ), f" {len(data.X_missing_values)}, {len(data.y_missing_values)}"
        assert len(data.X_missing_values) == len(
            data.X_modified
        ), f" {len(data.X_missing_values)}, {len(data.X_modified)}"

        #pdb.set_trace()
        if data.has_missing_values and np.all(np.isnan(data.X_missing_values), axis=1).any():
            print(f'retry {i}: {dc}')
        else: i +=1
        write_data_to_files([data], base_path=output_folder)
        data_objects_to_generate.append(data)

    return data_objects_to_generate


def write_data_to_files(result_data_objects: list[Data], base_path: str):
    i = 0
    Path(base_path).mkdir(parents=True, exist_ok=True)
    for d in result_data_objects:
        i = i + 1
        f_name = os.path.join(base_path, f'data_{d.id}_{d.uuid}.pickle')
        print(f'Write file {f_name}')
        with open(f_name, "wb") as f:
            pickle.dump(d, f)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate data and write to files.")
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        required=True,
                        help='Path to the config file')
    parser.add_argument('-si',
                        '--start_id',
                        type=int,
                        default=0,
                        help='Start Id for data generation (default: 0), only use when a generate process has stopped in the middle')
    parser.add_argument('-o',
                        '--output_folder',
                        type=str,
                        required=True,
                        help='Folder for output files')

    args = parser.parse_args()
    data_objects = generate_data_from_config(path=args.config, start_id=args.start_id, output_folder=args.output_folder)
