from dataclasses import dataclass, field
import uuid
from typing import Optional, List, Dict


import numpy as np

Array = np.ndarray

@dataclass
class Data:

    id: int = -1
    uuid: str = field(default_factory=lambda: str(uuid.uuid4()))

    X_base: Array = field(default_factory=lambda: np.empty((0, 0)))
    y: Array = field(default_factory=lambda: np.empty(0))

    X_modified: Array = field(default_factory=lambda: np.empty((0, 0)))
    y_modified: Array = field(default_factory=lambda: np.empty(0))

    X_missing_values: Array = field(default_factory=lambda: np.empty((0, 0)))
    y_missing_values: Array = field(default_factory=lambda: np.empty(0))

    X_preprocessed: Array = field(default_factory=lambda: np.empty((0, 0)))
    y_preprocessed: Array = field(default_factory=lambda: np.empty(0))

    # ---- meta‑features ----
    mf_base: Dict = field(default_factory=dict)
    mf_modified: Dict = field(default_factory=dict)
    mf_missing_values: Dict = field(default_factory=dict)
    mf_preprocessed: Dict = field(default_factory=dict)
    mf_modified_relevant: Dict = field(default_factory=dict)

    # ---- experiment settings ----
    seed: Optional[int] = None
    shape: Optional[str] = None
    n_samples: Optional[int] = None
    n_clusters: Optional[int] = None
    n_features: Optional[int] = None
    cluster_std: Optional[float] = None

    has_scaling: bool = False
    has_outlier: bool = False
    has_missing_values: bool = False
    multiplier_method: Optional[str] = None
    p_outlier: Optional[float] = None
    p_missing_values: Optional[float] = None

    # ---- scores ----
    base_ari: Optional[float] = None
    modified_ari: Optional[float] = None
    preprocessed_ari: Optional[float] = None

    # ---- GA logs ----
    generations: Array = field(default_factory=lambda: np.empty(0))
    deap_min: Array = field(default_factory=lambda: np.empty(0))
    deap_avg: Array = field(default_factory=lambda: np.empty(0))
    deap_max: Array = field(default_factory=lambda: np.empty(0))

    best_pipeline: List = field(default_factory=list)
    best_fitness: Optional[float] = None
    top_5_pipeline: List = field(default_factory=list)
    top_5_fitness: List = field(default_factory=list)

    # ---- GA parameters ----
    metric: Optional[str] = None
    mutpb: Optional[float] = None
    cxpb: Optional[float] = None
    indpb: Optional[float] = None
    time_budget: Optional[int] = None
    tourn_size: Optional[int] = None
    n_generation: Optional[int] = None
    population_size: Optional[int] = None
    runtime: Optional[float] = None

    run_hash: Optional[str] = None
    init_strategy: Optional[str] = None

    stopped_by_budget: bool = False
    warm_started: bool = False
