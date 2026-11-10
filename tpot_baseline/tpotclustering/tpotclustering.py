# -*- coding: utf-8 -*-

"""This file is part of the TPOT-Clustering library.

TPOT-Clustering is a fork of the TPOT library, extended to support unsupervised machine learning (clustering) tasks.

The original TPOT library was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Daniel Angell (dpa34@drexel.edu)
    - and many more generous open source contributors

TPOT-Clustering is maintained by:
    - Matheus Camilo da Silva (matheus.camilo@phd.units.it)
    - Sylvio Barbon Junior (sylvio.barbon@units.it)
    - with additional contributions from the open source community

TPOT-Clustering is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version.

TPOT-Clustering is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this program. If not, see <http://www.gnu.org/licenses/>.

Original TPOT project: https://github.com/EpistasisLab/tpot
TPOT-Clustering project: https://github.com/Mcamilo/tpot-clustering
"""

from .base import TPOTBase
from .config.clustering import (
    clustering_config_dict,
    make_clustering_config_fixed_n_clusters,
)


class TPOTClustering(TPOTBase):
    """TPOT estimator for clustering problems.

    Parameters
    ----------
    n_clusters : int or None, optional (default: None)
        When set to an integer, every clustering algorithm that accepts
        ``n_clusters`` will have that parameter fixed to this value for the
        entire optimization run.  TPOT will not explore alternative cluster
        counts, which is useful when the true number of clusters is known from
        ground-truth labels (``len(set(y_true))``).

        When ``None`` (default), the original behaviour is preserved and
        ``n_clusters`` is treated as a free hyperparameter sampled from
        ``range(2, 23)``.

    All other parameters are forwarded to :class:`TPOTBase`.
    """

    scoring_function = 'silhouette_score'  # Clustering scoring
    default_config_dict = clustering_config_dict
    clustering = True

    def __init__(
        self,
        n_clusters=None,
        generations=100,
        population_size=100,
        offspring_size=None,
        mutation_rate=0.9,
        crossover_rate=0.1,
        scoring=None,
        cv=5,
        subsample=1.0,
        n_jobs=1,
        max_time_mins=None,
        max_eval_time_mins=5,
        random_state=None,
        config_dict=None,
        template=None,
        warm_start=False,
        memory=None,
        use_dask=False,
        periodic_checkpoint_folder=None,
        early_stop=None,
        verbosity=0,
        disable_update_check=True,
        log_file=None,
    ):
        self.n_clusters = n_clusters
        if n_clusters is not None:
            self.default_config_dict = make_clustering_config_fixed_n_clusters(n_clusters)
        else:
            self.default_config_dict = clustering_config_dict
        super().__init__(
            generations=generations,
            population_size=population_size,
            offspring_size=offspring_size,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            scoring=scoring,
            cv=cv,
            subsample=subsample,
            n_jobs=n_jobs,
            max_time_mins=max_time_mins,
            max_eval_time_mins=max_eval_time_mins,
            random_state=random_state,
            config_dict=config_dict,
            template=template,
            warm_start=warm_start,
            memory=memory,
            use_dask=use_dask,
            periodic_checkpoint_folder=periodic_checkpoint_folder,
            early_stop=early_stop,
            verbosity=verbosity,
            disable_update_check=disable_update_check,
            log_file=log_file,
        )

    def _init_pretest(self, features, target=None):
        self.pretest_X = features
        self.pretest_y = target
